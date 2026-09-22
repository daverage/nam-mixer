"""Optional, bounded research helpers for the local recipe assistant."""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from hybrid.env_file import read_saved_env_value as _env


TONE3000_BASE = "https://www.tone3000.com/api/v1"

_METADATA_STOPWORDS = frozenset({
    "amp", "and", "are", "for", "from", "guitar", "have", "into", "its",
    "model", "need", "over", "pack", "sound", "that", "the", "this", "tone",
    "with", "your",
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
    """Extract readable page text without executing or trusting page markup."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1

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


def _page_evidence(href: str, query: str) -> str:
    """Fetch a short, relevant text extract from a search result page."""
    parsed = urlparse(href)
    if parsed.scheme not in ("https", "http") or not _is_safe_public_host(parsed.hostname or ""):
        return ""
    try:
        request = Request(href, headers={"User-Agent": "NAM-Mixer research/1.0"})
        opener = build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=8) as response:
            if "html" not in response.headers.get("Content-Type", ""):
                return ""
            parser = _PageText()
            parser.feed(response.read(750_000).decode("utf-8", errors="ignore"))
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    keywords = {"amp", "amplifier", "vox", "mesa", "marshall", "fender", "live", "rig"}
    keywords.update(word.lower() for word in re.findall(r"[a-zA-Z]{4,}", query))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    relevant = [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 50 and any(word in sentence.lower() for word in keywords)]
    return " ".join(relevant[:2])[:600]


def web_notes(query: str) -> str:
    """Return search snippets plus small, relevant extracts from their source pages."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("Web research needs the 'ddgs' package. Run scripts/run.sh to install it.") from exc
    try:
        with DDGS() as search:
            results = search.text(f"{query} guitar amp rig settings", max_results=3, timeout=10)
    except Exception as exc:  # ddgs exposes provider-specific exception types.
        raise RuntimeError(f"Web research failed: {exc}") from exc
    notes = []
    for result in results or []:
        title = str(result.get("title", "")).strip()
        body = str(result.get("body", "")).strip()
        href = str(result.get("href", "")).strip()
        if title or body:
            evidence = _page_evidence(href, query)
            source_text = evidence or body
            notes.append(f"- {title}: {source_text} ({href})"[:750])
    if not notes:
        raise RuntimeError("Web research returned no results.")
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


def tone3000_search(query: str, *, rig_scope: str, author: str = "", rank_query: str = "", opener=urlopen) -> list[dict]:
    """Search public TONE3000 metadata; captures themselves are never downloaded.

    `query` is the narrow amp-family term sent to the catalogue API, which
    already filters for relevance -- scoring every result against that same
    short term is nearly always 100% and tells the user nothing. `rank_query`
    (typically the user's full, richer request) is used for match scoring
    instead so results are actually differentiated; it defaults to `query`
    when the caller has nothing richer to offer.
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


def _tone3000_models_payload(tone_id: int, *, opener=urlopen) -> list[dict]:
    api_key = _require_tone3000_api_key(for_action="downloads")
    request = Request(
        f"{TONE3000_BASE}/models?{urlencode({'tone_id': tone_id, 'architecture': 2})}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"TONE3000 pack details failed: {exc}") from exc
    return payload.get("data", []) if isinstance(payload, dict) else []


def tone3000_models(tone_id: int, *, opener=urlopen) -> list[dict]:
    """Return model names for one public TONE3000 tone pack, never credentials."""
    raw_models = _tone3000_models_payload(tone_id, opener=opener)
    models = []
    for model in raw_models[:30]:
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


def tone3000_notes(query: str, *, rig_scope: str, author: str = "", opener=urlopen) -> str:
    return "\n".join(
        f"- TONE3000: {result['title']} — by {result['creator']}. {result['description']}"[:600]
        for result in tone3000_search(query, rig_scope=rig_scope, author=author, opener=opener)[:4]
    )
