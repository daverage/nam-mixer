"""Optional, bounded research helpers for the local recipe assistant."""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from hybrid.services.env_file import read_saved_env_value as _env


TONE3000_BASE = "https://www.tone3000.com/api/v1"

_METADATA_STOPWORDS = frozenset({
    # Domain filler: generic enough to appear in almost any amp listing or
    # any user request, so matching on them tells you nothing about fit.
    "amp", "amps", "guitar", "model", "pack", "sound", "tone", "capture",
    "captures",
    # General English function/filler words. `rank_query` is often the
    # user's full, conversational request (see tone3000_search's docstring),
    # not a curated search term -- without a broad stopword list, words like
    # "would"/"like"/"way" from ordinary sentences were being counted as
    # genuine metadata matches, drowning out the few words that actually
    # distinguish one capture from another (see
    # test_tone3000_metadata_score_ignores_generic_query_words).
    "and", "are", "for", "from", "have", "into", "its", "need", "over",
    "that", "the", "this", "with", "your", "you", "about", "after", "all",
    "also", "any", "because", "been", "being", "but", "can", "cant",
    "could", "did", "does", "doing", "dont", "down", "each", "get", "gets",
    "getting", "going", "had", "has", "her", "here", "him", "his", "how",
    "ive", "just", "know", "like", "likes", "make", "makes", "many", "may",
    "might", "more", "most", "much", "must", "myself", "not", "now", "off",
    "one", "only", "other", "our", "out", "really", "same", "should",
    "some", "still", "such", "than", "there", "these", "they", "think",
    "those", "through", "too", "try", "trying", "use", "used", "using",
    "very", "want", "wanted", "wants", "was", "way", "well", "were",
    "what", "when", "where", "which", "while", "who", "why", "will",
    "would",
})


def _rank_tone3000_metadata(query: str, result: dict) -> tuple[int, str]:
    """Rank catalogue records from supplied metadata, never invented facts.

    This is the deterministic safety net used before the local model reads the
    shortlist. Scores represent the portion of the meaningful query terms that
    occur in the title or description; a title hit is weighted more heavily.
    """
    terms = {
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) >= 3 and term not in _METADATA_STOPWORDS
    }
    title_terms = set(re.findall(r"[a-z0-9]+", result["title"].lower()))
    description_terms = set(re.findall(r"[a-z0-9]+", result["description"].lower()))
    title_hits = sorted(terms & title_terms)
    description_hits = sorted((terms & description_terms) - set(title_hits))
    matched_weight = 2 * len(title_hits) + len(description_hits)
    score = round(100 * matched_weight / (2 * len(terms))) if terms else 0
    reason = "Metadata matches " + ", ".join(title_hits + [term for term in description_hits if term not in title_hits]) if (title_hits or description_hits) else "Limited catalogue metadata; inspect the pack description"
    return score, reason + "."


class _PageText(HTMLParser):
    """Extract readable page text without executing or trusting page markup.

    Page chrome (navigation, headers, footers, forms) is dropped, and block elements end a line,
    so menus never run together into one long fake "sentence".
    """

    IGNORED = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "button", "select", "menu", "template"}
    BLOCKS = {"p", "li", "div", "section", "article", "br", "tr", "td", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "dd", "dt"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORED:
            self._ignored += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.IGNORED and self._ignored:
            self._ignored -= 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._ignored:
            self.parts.append(data)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse redirects instead of silently re-fetching an unvalidated host.

    A search-engine result is third-party, untrusted data; if we validated
    the original hostname but then transparently followed a redirect, a
    malicious or compromised page could point us at an internal address
    (SSRF) after the safety check already passed.
    """

    def redirect_request(self, *args, **kwargs):
        return None


def _is_safe_public_host(hostname: str) -> bool:
    """Reject loopback/private/link-local/reserved targets to guard against SSRF.

    `href` values come from third-party search results, not from a trusted
    catalogue API -- unlike the TONE3000 calls in this module, which only
    ever hit a fixed, known-safe base URL.
    """
    if not hostname:
        return False
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except OSError:
        return False
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            return False
        if (
            address.is_private or address.is_loopback or address.is_link_local
            or address.is_reserved or address.is_multicast or address.is_unspecified
        ):
            return False
    return True


# Words that make a research sentence about the actual rig rather than about the web page.
_GEAR_WORDS = {
    "amp", "amps", "amplifier", "amplifiers", "head", "combo", "cabinet", "cab", "speaker", "speakers", "celestion",
    "jensen", "marshall", "fender", "vox", "mesa", "boogie", "hiwatt", "orange", "peavey", "ampeg", "dumble", "plexi",
    "twin", "deluxe", "champ", "bassman", "princeton", "stratocaster", "strat", "telecaster", "tele", "gibson",
    "sg", "explorer", "flying", "humbucker", "humbuckers", "pickup", "pickups", "p90", "pedal", "pedals", "fuzz",
    "overdrive", "distortion", "wah", "screamer", "rangemaster", "booster", "univibe", "leslie", "reverb",
    "tremolo", "echo", "delay", "echoplex", "recorded", "recording", "studio", "session", "played", "plugged",
    "used", "cranked", "gain", "valve", "tube", "tubes",
}
_FILLER_WORDS = {
    "i", "im", "i'm", "id", "i'd", "would", "like", "want", "wanted", "need", "looking", "look", "for", "find",
    "a", "an", "the", "tone", "tones", "sound", "sounds", "sounding", "that", "which", "to", "of", "on", "in",
    "with", "from", "had", "has", "have", "used", "uses", "use", "get", "got", "match", "matches", "matching",
    "similar", "please", "me", "my", "some", "is", "was", "be", "it", "and", "or", "he", "she", "they", "their",
    "his", "her", "what", "how", "can", "you", "just", "really", "exact", "exactly", "same", "as", "at", "by",
    # The Builder's focused amp-discovery question wraps the request in these words.
    "which", "specific", "guitar", "amplifier", "makes", "models", "did", "artist", "song", "era", "described",
    "this", "request",
}
# Social, video and preset-sharing sites rarely say what the artist actually used.
_SKIP_HOSTS = ("tiktok.com", "youtube.com", "youtu.be", "instagram.com", "facebook.com", "pinterest.", "twitter.com",
               "x.com", "tone.fender.com", "line6.com", "spotify.com", "apple.com", "amazon.", "ebay.", "reverb.com")
MAX_SOURCES = 4
EVIDENCE_CHARS = 700


def _topic(query: str) -> str:
    """The artist/song/gear words of a request, without conversational filler."""
    words = re.findall(r"[\w'’.-]+", query)
    kept = [w for w in words if w.lower().strip(".'’") not in _FILLER_WORDS]
    return " ".join(kept) or query.strip()


def _skip_source(href: str) -> bool:
    host = (urlparse(href).hostname or "").lower()
    return not host or any(host == pattern or host.endswith("." + pattern) or (pattern.endswith(".") and pattern in host)
                           for pattern in _SKIP_HOSTS)


def _sentence_score(sentence: str, topic_words: set) -> int:
    """0 for menus and boilerplate; otherwise topic hits (weighted) plus distinct gear words."""
    words = re.findall(r"[a-z0-9']+", sentence.lower())
    if len(words) < 7 or len(sentence) > 450:
        return 0
    if "→" in sentence or "»" in sentence or sentence.count("·") >= 2 or sentence.count("|") >= 2:
        return 0  # "related links" strips: arrows and dot/pipe separators
    capitalised = sum(1 for w in re.findall(r"[A-Za-z][\w']*", sentence) if w[0].isupper())
    if capitalised > len(words) * 0.5:  # Title Case runs are menus, headings and tag lists
        return 0
    gear = len(set(words) & _GEAR_WORDS) + (1 if "les paul" in sentence.lower() else 0)
    topic = len(set(words) & topic_words)
    return (topic * 2 + gear) if (topic and gear) or gear >= 3 else 0


def _extract_evidence(html: str, topic: str) -> str:
    """The best few sentences of a page about the requested rig, in page order."""
    parser = _PageText()
    parser.feed(html)
    topic_words = {w.lower() for w in re.findall(r"[A-Za-z0-9']{3,}", topic)}
    candidates, seen = [], set()
    for block in "".join(parser.parts).split("\n"):
        block = re.sub(r"\s+", " ", block).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            sentence = sentence.strip()
            if sentence.lower() in seen:  # pages often repeat a line (summary box + body)
                continue
            if sentence[-1:] in ".!?\"”)" and (score := _sentence_score(sentence, topic_words)):
                seen.add(sentence.lower())
                candidates.append((score, len(candidates), sentence))
    best = sorted(sorted(candidates, reverse=True)[:3], key=lambda c: c[1])
    return " ".join(c[2] for c in best)[:EVIDENCE_CHARS]


def _page_evidence(href: str, topic: str) -> str:
    """Fetch a public page (no redirects, bounded size) and extract rig evidence from it."""
    parsed = urlparse(href)
    if parsed.scheme not in ("https", "http") or not _is_safe_public_host(parsed.hostname or ""):
        return ""
    try:
        request = Request(href, headers={"User-Agent": "NAM-Mixer research/1.0"})
        opener = build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=8) as response:
            if "html" not in response.headers.get("Content-Type", ""):
                return ""
            html = response.read(750_000).decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return _extract_evidence(html, topic)


def _ddgs_search(query: str, max_results: int) -> list:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("Web research needs the 'ddgs' package (run scripts/run.sh to install it).") from exc
    with DDGS() as search:
        return search.text(query, max_results=max_results, timeout=10) or []


def web_notes(query: str, *, search=_ddgs_search, evidence=_page_evidence) -> str:
    """Return up to MAX_SOURCES notes of documented gear for the request, one "- title: text (url)" line each."""
    topic = _topic(query)
    results, seen = [], set()
    try:
        for search_query in (f"{topic} guitar rig amp used recording", f"{topic} equipment gear equipboard"):
            for result in search(search_query, 6):
                href = str(result.get("href", "")).strip()
                if href and href not in seen and not _skip_source(href):
                    seen.add(href)
                    results.append(result)
    except RuntimeError:
        raise
    except Exception as exc:  # ddgs exposes provider-specific exception types.
        raise RuntimeError(f"Web research failed: {exc}") from exc

    candidates = results[:8]
    with ThreadPoolExecutor(max_workers=len(candidates) or 1) as pool:
        extracts = list(pool.map(lambda r: evidence(str(r.get("href", "")).strip(), topic), candidates))
    topic_words = {w.lower() for w in re.findall(r"[A-Za-z0-9']{3,}", topic)}
    notes = []
    for result, extract in zip(candidates, extracts):
        title = str(result.get("title", "")).strip()
        snippet = re.sub(r"\s+", " ", str(result.get("body", ""))).strip()
        text = extract or (snippet if _sentence_score(snippet if snippet[-1:] in ".!?" else snippet + ".", topic_words) else "")
        if text:
            notes.append(f"- {title[:120]}: {text[:EVIDENCE_CHARS]} ({result['href']})")  # URL last and intact
        if len(notes) == MAX_SOURCES:
            break
    if not notes:
        raise RuntimeError("Web research found no pages describing this rig.")
    return "\n".join(notes)


def _require_tone3000_api_key(*, for_action: str) -> str:
    """Return a validated TONE3000_API_KEY, never logging its value.

    TONE3000 credentials deliberately come from the app's saved settings,
    ignoring inherited shell values that may be stale (see env_file.py).
    """
    api_key = _env("TONE3000_API_KEY")
    if not api_key:
        raise RuntimeError(f"TONE3000 {for_action} needs a server-side TONE3000_API_KEY secret key (t3k_cs_…).")
    if not api_key.startswith("t3k_cs_"):
        raise RuntimeError(
            f"TONE3000 {for_action} found a TONE3000_API_KEY, but it does not start with the expected "
            "'t3k_cs_' secret-key prefix. Replace or clear the saved key in Settings → TONE3000; "
            "inherited shell values do not override the app's saved credential."
        )
    return api_key


_TONE3000_LINK_PREFIXES = ("https://api.tone3000.com/", "https://www.tone3000.com/")


def _tone3000_link(value: object) -> str | None:
    text = str(value or "")
    return text if text.startswith(_TONE3000_LINK_PREFIXES) else None


def _tone3000_card_fields(tone: dict) -> dict:
    """Display-only pack metadata; links are kept only when they point at TONE3000 itself."""
    def count(key: str) -> int | None:
        value = tone.get(key)
        return value if isinstance(value, int) and value >= 0 else None

    images = tone.get("images") if isinstance(tone.get("images"), list) else []
    tags = tone.get("tags") if isinstance(tone.get("tags"), list) else []
    makes = tone.get("makes") if isinstance(tone.get("makes"), list) else []
    return {
        "image": next((link for link in map(_tone3000_link, images) if link), None),
        "tags": [str(tag.get("name"))[:40] for tag in tags if isinstance(tag, dict) and tag.get("name")][:12],
        "makes": [str(make.get("name") if isinstance(make, dict) else make)[:40] for make in makes if make][:6],
        "gear": str(tone.get("gear") or "")[:30] or None,
        "a2_models_count": count("a2_models_count"),
        "downloads_count": count("downloads_count"),
        "favorites_count": count("favorites_count"),
        "url": _tone3000_link(tone.get("url")),
    }


def tone3000_search(query: str, *, rig_scope: str, author: str = "", rank_query: str = "", opener=urlopen) -> list[dict]:
    """Search public TONE3000 metadata; captures themselves are never downloaded.

    `query` is the narrow amp-family term sent to the catalogue API, which
    already filters for relevance -- scoring every result against that same
    short term is nearly always 100% and tells the user nothing. `rank_query`
    is used for match scoring instead so results are actually differentiated;
    it defaults to `query` when the caller has nothing richer to offer.
    Callers should prefer passing model-curated vocabulary here (e.g. the
    assistant's own proposed search terms/source-plan roles) over the user's
    raw conversational sentence -- the scorer is a plain deterministic
    word-overlap match, not a semantic one, so it can't tell a genuinely
    distinguishing word from incidental filler on its own; see
    `_METADATA_STOPWORDS` and app.py's callsite for the reasoning.
    """
    api_key = _require_tone3000_api_key(for_action="search")
    params = {"query": query, "page": 1, "page_size": 20, "sort": "best-match", "format": "nam", "architecture": "2"}
    if rig_scope == "heads":
        params["gears"] = "amp"
    request = Request(
        f"{TONE3000_BASE}/tones/search?{urlencode(params)}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"TONE3000 search failed: {exc}") from exc
    tones = payload.get("data", []) if isinstance(payload, dict) else []
    author = author.strip().lower()
    results = []
    for tone in tones:
        # An id-less entry can never be turned into a working discuss/download
        # link downstream (both key off this id), so surfacing it would just
        # be a dead result the player can click and get nothing from.
        if not isinstance(tone.get("id"), int):
            continue
        user = tone.get("user") or {}
        creator = str(user.get("display_name") or user.get("username") or "")
        if author and author not in creator.lower():
            continue
        title = str(tone.get("title") or tone.get("name") or "Untitled capture")
        description = str(tone.get("description") or "").replace("\n", " ")
        result = {
            "id": tone.get("id"),
            "title": title,
            "creator": creator or "unknown creator",
            "description": description[:600],
            **_tone3000_card_fields(tone),
        }
        result["match_score"], result["match_reason"] = _rank_tone3000_metadata(rank_query or query, result)
        results.append(result)
    if not results:
        scope = "heads only" if rig_scope == "heads" else "any rig"
        if rig_scope == "heads":
            raise RuntimeError(
                "TONE3000 returned no heads-only matches for that wording. Try a specific amp family "
                "(for example, 'Vox AC30' or 'Marshall JCM800'), or switch the search to Anything."
            )
        raise RuntimeError(f"TONE3000 returned no {scope} matches for that search.")
    # The catalogue order is useful, but it is not the user's request-specific
    # ranking.  Rank the complete API page before shortening it: taking the
    # first eight first could discard the best AC30/JCM result simply because
    # it appeared later in TONE3000's broad best-match response.
    return sorted(
        results,
        key=lambda result: (result["match_score"], result["title"].lower()),
        reverse=True,
    )[:8]


MODELS_PAGE_SIZE = 50
MAX_MODELS = 250


def _tone3000_models_payload(tone_id: int, *, opener=urlopen) -> list[dict]:
    """Every A2 model in a pack: the API pages its results, so keep asking until a short page."""
    api_key = _require_tone3000_api_key(for_action="downloads")
    models: list[dict] = []
    page = 1
    while len(models) < MAX_MODELS:
        request = Request(
            f"{TONE3000_BASE}/models?{urlencode({'tone_id': tone_id, 'architecture': 2, 'page': page, 'page_size': MODELS_PAGE_SIZE})}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with opener(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if models:  # keep what we have rather than failing the whole pack on a later page
                break
            raise RuntimeError(f"TONE3000 pack details failed: {exc}") from exc
        batch = payload.get("data", []) if isinstance(payload, dict) else []
        models.extend(batch)
        total_pages = payload.get("total_pages") if isinstance(payload, dict) else None
        if not batch or (isinstance(total_pages, int) and page >= total_pages) or (total_pages is None and len(batch) < MODELS_PAGE_SIZE):
            break
        page += 1
    return models[:MAX_MODELS]


def tone3000_models(tone_id: int, *, opener=urlopen) -> list[dict]:
    """Return model names for one public TONE3000 tone pack, never credentials."""
    raw_models = _tone3000_models_payload(tone_id, opener=opener)
    models = []
    for model in raw_models:
        url = str(model.get("model_url") or "")
        if not url.startswith("https://"):
            continue
        models.append({
            "id": model.get("id"),
            "name": str(model.get("name") or "Unnamed NAM file"),
            "architecture": model.get("architecture_version"),
        })
    if not models:
        raise RuntimeError("TONE3000 returned no downloadable A2 NAM files in this pack.")
    return models


def tone3000_model_download(tone_id: int, model_id: int, *, opener=urlopen) -> tuple[bytes, str]:
    """Download a selected pack member server-side, keeping the API key private."""
    raw_models = _tone3000_models_payload(tone_id, opener=opener)
    model = next((item for item in raw_models if item.get("id") == model_id), None)
    url = str((model or {}).get("model_url") or "")
    if not url.startswith(f"{TONE3000_BASE}/models/"):
        raise RuntimeError("TONE3000 could not provide a safe download link for that NAM file.")
    api_key = _require_tone3000_api_key(for_action="downloads")
    try:
        with opener(Request(url, headers={"Authorization": f"Bearer {api_key}"}), timeout=30) as response:
            data = response.read()
    except Exception as exc:
        raise RuntimeError(f"TONE3000 model download failed: {exc}") from exc
    if not data:
        raise RuntimeError("TONE3000 returned an empty NAM download.")
    return data, str(model.get("name") or f"tone3000-{model_id}")
