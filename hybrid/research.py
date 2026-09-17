"""Optional, bounded research helpers for the local recipe assistant."""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hybrid.env_file import read_env_value as _env


TONE3000_BASE = "https://www.tone3000.com/api/v1"


def _rank_tone3000_metadata(query: str, result: dict) -> tuple[int, str]:
    """Rank catalogue records from supplied metadata, never invented facts.

    This is the deterministic safety net used before the local model reads the
    shortlist: a concrete family match in a pack description is more useful
    than API ordering alone, while an undocumented pack stays conservative.
    """
    terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 3}
    title = result["title"].lower()
    description = result["description"].lower()
    title_hits = sorted(term for term in terms if term in title)
    description_hits = sorted(term for term in terms if term in description)
    score = 30 + 12 * len(title_hits) + 7 * len(description_hits)
    if description.strip():
        score += 3
    reason = "Metadata matches " + ", ".join(title_hits + [term for term in description_hits if term not in title_hits]) if (title_hits or description_hits) else "Limited catalogue metadata; inspect the pack description"
    return min(100, score), reason + "."


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


def _page_evidence(href: str, query: str) -> str:
    """Fetch a short, relevant text extract from a search result page."""
    if not href.startswith(("https://", "http://")):
        return ""
    try:
        request = Request(href, headers={"User-Agent": "NAM-Mixer research/1.0"})
        with urlopen(request, timeout=8) as response:
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


def tone3000_search(query: str, *, rig_scope: str, author: str = "", opener=urlopen) -> list[dict]:
    """Search public TONE3000 metadata; captures themselves are never downloaded."""
    api_key = _env("TONE3000_API_KEY")
    if not api_key.startswith("t3k_cs_"):
        raise RuntimeError("TONE3000 search needs a server-side TONE3000_API_KEY secret key (t3k_cs_…).")
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
        result["match_score"], result["match_reason"] = _rank_tone3000_metadata(query, result)
        results.append(result)
        if len(results) == 8:
            break
    if not results:
        scope = "heads only" if rig_scope == "heads" else "any rig"
        if rig_scope == "heads":
            raise RuntimeError(
                "TONE3000 returned no heads-only matches for that wording. Try a specific amp family "
                "(for example, 'Vox AC30' or 'Marshall JCM800'), or switch the search to Anything."
            )
        raise RuntimeError(f"TONE3000 returned no {scope} matches for that search.")
    return sorted(results, key=lambda result: result["match_score"], reverse=True)


def _tone3000_models_payload(tone_id: int, *, opener=urlopen) -> list[dict]:
    api_key = _env("TONE3000_API_KEY")
    if not api_key.startswith("t3k_cs_"):
        raise RuntimeError("TONE3000 downloads need a server-side TONE3000_API_KEY secret key (t3k_cs_…).")
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
    api_key = _env("TONE3000_API_KEY")
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
