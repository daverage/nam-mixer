"""AI helpers for the TONE3000 tab: tone planning, pack ranking and per-pack questions.

Separate from local_llm.converse on purpose: that prompt designs Amp A/B blend recipes, while this
one only describes tones, finds captures and discusses real pack files. It reuses local_llm's
provider plumbing (config, JSON-schema request, content decoding) so every provider works the same.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import local_llm
from .local_llm import LocalLlmError

_TIMEOUT_SECONDS = 90
_MAX_HISTORY = 8
_HISTORY_CHARS = 1_200


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Gear(_Model):
    kind: Literal["amp", "effect", "guitar", "pickup", "cab", "other"] = "other"
    name: str = Field(min_length=1, max_length=80)
    role: str = Field(default="", max_length=240)


class _Plan(_Model):
    summary: str = Field(min_length=1, max_length=900)
    advice: list[str] = Field(default_factory=list, max_length=8)
    gear: list[_Gear] = Field(default_factory=list, max_length=12)
    search_queries: list[str] = Field(default_factory=list, max_length=6)


class _Rank(_Model):
    id: int
    fit: int = Field(ge=0, le=100)
    why: str = Field(default="", max_length=300)


class _Ranking(_Model):
    ranking: list[_Rank] = Field(default_factory=list, max_length=40)


class _PackAnswer(_Model):
    reply: str = Field(min_length=1, max_length=1_800)
    recommended_files: list[str] = Field(default_factory=list, max_length=6)


_STYLE = (
    "You are an experienced guitar and bass tone advisor for players who use Neural Amp Modeler (NAM) "
    "captures from the TONE3000 catalogue. Be concrete and practical. Never invent facts about a "
    "specific artist's rig: when research notes are supplied, prefer them and say when something is "
    "uncertain. Only mention effects when the research or the well-documented history of the tone "
    "actually involves them. Plain text only inside JSON strings, no markdown."
)

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "advice": {"type": "array", "items": {"type": "string"}},
        "gear": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["amp", "effect", "guitar", "pickup", "cab", "other"]},
            "name": {"type": "string"}, "role": {"type": "string"}}, "required": ["kind", "name"]}},
        "search_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "advice", "gear", "search_queries"],
}
_RANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "integer"}, "fit": {"type": "integer"}, "why": {"type": "string"}},
        "required": ["id", "fit", "why"]}}},
    "required": ["ranking"],
}
_PACK_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}, "recommended_files": {"type": "array", "items": {"type": "string"}}},
    "required": ["reply", "recommended_files"],
}


def _history_messages(history: list[dict] | None) -> list[dict]:
    messages = []
    for item in (history or [])[-_MAX_HISTORY:]:
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:_HISTORY_CHARS]})
    return messages


def _ask(system: str, user: str, schema_name: str, schema: dict, model: type[_Model], *,
         history: list[dict] | None = None, max_tokens: int = 1_400, opener=None) -> _Model:
    config = local_llm._config()
    if config is None:
        raise LocalLlmError("local LLM is not configured")
    messages = [{"role": "system", "content": system}, *_history_messages(history), {"role": "user", "content": user}]
    kwargs = {"opener": opener} if opener is not None else {"opener": local_llm.urlopen}
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            content = local_llm._post_chat_completion(
                config, messages, max_tokens=max_tokens, temperature=0.3 if attempt == 0 else 0.1,
                timeout=_TIMEOUT_SECONDS, schema=(schema_name, schema), **kwargs,
            )
            return model.model_validate(local_llm._decode_json_content(content))
        except (ValueError, ValidationError, KeyError, TypeError) as exc:
            last_error = exc
            messages = [*messages[:-1], {"role": "user", "content": user + "\n\nReturn ONLY one JSON object matching the requested fields."}]
        except LocalLlmError:
            raise
        except Exception as exc:  # network / provider failure: one clear message for the UI
            raise LocalLlmError(f"AI provider request failed: {exc}") from exc
    raise LocalLlmError(f"The AI returned an answer that could not be read ({last_error}). Try again or use a larger model.")


def plan_tone(prompt: str, *, research_notes: str = "", history: list[dict] | None = None, opener=None) -> dict:
    user = (
        f"Player request: {prompt.strip()}\n\n"
        + (f"Web research notes (may be partial or noisy):\n{research_notes[:local_llm.LOCAL_LLM_RESEARCH_CHARS]}\n\n" if research_notes else "")
        + "Return JSON with:\n"
        "- summary: 2-4 sentences describing the tone in plain words (gain, EQ, feel, era).\n"
        "- advice: 3-6 short practical tips (amp settings, playing, guitar/pickup choice, and effects only if relevant).\n"
        "- gear: the specific products that define this tone, each with kind, name and a short role. Use real "
        "make and model names (for example 'Fender Vibroverb', 'Ibanez TS808 Tube Screamer', 'Fender Stratocaster'), "
        "never generic categories like 'tube amplifier' or 'overdrive pedal'. If the request names an artist, song "
        "or album, list that player's documented gear for it.\n"
        "- search_queries: 1-3 SHORT TONE3000 catalogue searches, each an amp make/model or amp family "
        "(for example 'Marshall JCM800', 'Fender Deluxe Reverb', 'Vox AC30'), matching the amps in gear. "
        "Always include the model, never a bare brand like 'Marshall'. No effects, no adjectives, no artist names."
    )
    plan = _ask(_STYLE, user, "tone_plan", _PLAN_SCHEMA, _Plan, history=history, opener=opener)
    queries = list(dict.fromkeys(q.strip()[:80] for q in plan.search_queries if q and q.strip()))[:3]
    return {
        "summary": plan.summary.strip(),
        "advice": [tip.strip() for tip in plan.advice if tip.strip()][:6],
        "gear": [gear.model_dump() for gear in plan.gear][:10],
        "search_queries": queries,
    }


def rank_packs(prompt: str, summary: str, packs: list[dict], *, opener=None) -> dict[int, dict]:
    """Score real catalogue packs against the request. Unknown ids from the model are ignored."""
    if not packs:
        return {}
    lines = []
    for pack in packs[:12]:
        tags = ", ".join(pack.get("tags") or [])
        lines.append(json.dumps({
            "id": pack["id"], "title": pack.get("title", ""), "gear": pack.get("gear"),
            "tags": tags, "description": (pack.get("description") or "")[:280],
        }))
    user = (
        f"Player request: {prompt.strip()}\nTone summary: {summary}\n\nCandidate TONE3000 packs (one JSON per line):\n"
        + "\n".join(lines)
        + "\n\nScore EVERY candidate: fit 0-100 for how well the pack could produce the requested tone, and "
        "why in one short sentence that names the deciding detail. Use only the ids given."
    )
    ranking = _ask(_STYLE, user, "pack_ranking", _RANK_SCHEMA, _Ranking, max_tokens=1_800, opener=opener)
    known = {pack["id"] for pack in packs}
    return {item.id: {"fit": item.fit, "why": item.why.strip()} for item in ranking.ranking if item.id in known}


def ask_about_pack(question: str, pack: dict, file_names: list[str], *, tone_goal: str = "",
                   history: list[dict] | None = None, opener=None) -> dict:
    files = "\n".join(f"- {name}" for name in file_names[:30])
    user = (
        (f"The player is looking for this tone: {tone_goal.strip()}\n\n" if tone_goal.strip() else "")
        + f"TONE3000 pack: {pack.get('title', '')} by {pack.get('creator', '')}\n"
        f"Tags: {', '.join(pack.get('tags') or [])}\nDescription: {(pack.get('description') or '')[:600]}\n"
        f"NAM files in this pack:\n{files}\n\nQuestion: {question.strip()}\n\n"
        "Answer the question using the pack information. File names often encode gain, channel or mic "
        "settings; explain what they suggest and say when you are inferring. In recommended_files list "
        "only exact file names from the list above that you recommend (can be empty)."
    )
    answer = _ask(_STYLE, user, "pack_answer", _PACK_SCHEMA, _PackAnswer, history=history, opener=opener)
    picks = [name for name in (_match_file(rec, file_names) for rec in answer.recommended_files) if name]
    return {"reply": answer.reply.strip(), "recommended_files": list(dict.fromkeys(picks))}


def _match_file(recommended: str, file_names: list[str]) -> str | None:
    """Exact name, or a shortened one ('EDGY') that identifies exactly one real file."""
    wanted = recommended.strip().lower()
    if not wanted:
        return None
    exact = [name for name in file_names if name.lower() == wanted]
    if exact:
        return exact[0]
    partial = [name for name in file_names if wanted in name.lower()]
    return partial[0] if len(partial) == 1 else None
