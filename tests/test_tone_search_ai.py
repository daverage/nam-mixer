from __future__ import annotations

import json

import pytest

import app as app_module
from hybrid.services import local_llm, tone_search_ai
from hybrid.services.local_llm import AiConfig, LocalLlmError
from hybrid.services.research import _tone3000_card_fields


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    replies: list[object] = []
    sent: list[dict] = []
    monkeypatch.setattr(local_llm, "_config", lambda **_: AiConfig(provider="local", base_url="http://x", model="m", api_key=""))

    def post(config, messages, **kwargs):
        sent.append({"messages": messages, "schema": kwargs.get("schema")})
        return replies.pop(0)

    monkeypatch.setattr(local_llm, "_post_chat_completion", post)
    return replies, sent


def test_plan_uses_its_own_schema_and_trims_queries(fake_llm):
    replies, sent = fake_llm
    replies.append(json.dumps({
        "summary": "Edge of breakup blues.", "advice": ["Roll the volume back", " "],
        "gear": [{"kind": "amp", "name": "Fender Deluxe", "role": "core"}],
        "search_queries": ["Fender Deluxe", "Fender Deluxe", "Vox AC30", "Marshall Bluesbreaker", "extra"],
    }))
    plan = tone_search_ai.plan_tone("warm blues", research_notes="notes about a Deluxe")
    assert sent[0]["schema"][0] == "tone_plan"
    assert "notes about a Deluxe" in sent[0]["messages"][-1]["content"]
    assert plan["search_queries"] == ["Fender Deluxe", "Vox AC30", "Marshall Bluesbreaker"]
    assert plan["advice"] == ["Roll the volume back"]


def test_plan_retries_once_on_unreadable_json_then_fails_clearly(fake_llm):
    replies, _ = fake_llm
    replies.extend(["not json", "still not json"])
    with pytest.raises(LocalLlmError, match="could not be read"):
        tone_search_ai.plan_tone("anything")


def test_rank_ignores_ids_the_model_invented(fake_llm):
    replies, _ = fake_llm
    replies.append({"ranking": [{"id": 1, "fit": 90, "why": "Deluxe"}, {"id": 999, "fit": 100, "why": "made up"}]})
    scores = tone_search_ai.rank_packs("blues", "summary", [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}])
    assert scores == {1: {"fit": 90, "why": "Deluxe"}}


def test_pack_answer_only_recommends_real_file_names(fake_llm):
    replies, _ = fake_llm
    replies.append({"reply": "Use CLEAN.", "recommended_files": ["CLEAN - x", "IMAGINARY"]})
    answer = tone_search_ai.ask_about_pack("which?", {"title": "Pack"}, ["CLEAN - x", "HOT - x"])
    assert answer == {"reply": "Use CLEAN.", "recommended_files": ["CLEAN - x"]}


def test_card_fields_keep_only_tone3000_links():
    fields = _tone3000_card_fields({
        "images": ["https://evil.example/x.jpg", "https://api.tone3000.com/storage/a.jpg"],
        "url": "javascript:alert(1)", "tags": [{"name": "blues"}], "gear": "amp", "a2_models_count": 3,
        "downloads_count": -1,
    })
    assert fields["image"] == "https://api.tone3000.com/storage/a.jpg"
    assert fields["url"] is None and fields["downloads_count"] is None
    assert fields["tags"] == ["blues"] and fields["a2_models_count"] == 3


def test_ai_search_route_ranks_real_packs(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(app_module, "plan_tone", lambda prompt, **_: {
        "summary": "s", "advice": [], "gear": [], "search_queries": ["Fender Deluxe", "Vox AC30"]})
    catalogue = {"Fender Deluxe": [{"id": 1, "title": "Deluxe", "match_score": 50}],
                 "Vox AC30": [{"id": 2, "title": "AC30", "match_score": 90}, {"id": 1, "title": "Deluxe", "match_score": 50}]}
    monkeypatch.setattr(app_module, "tone3000_search", lambda q, **_: catalogue[q])
    monkeypatch.setattr(app_module, "rank_packs", lambda *a, **k: {1: {"fit": 95, "why": "d"}, 2: {"fit": 40, "why": "a"}})
    data = client.post("/api/tone3000/ai_search", json={"prompt": "blues"}).get_json()
    assert [p["id"] for p in data["results"]] == [1, 2]
    assert data["results"][0]["ai_fit"] == 95 and data["results"][0]["query"] == "Fender Deluxe"


def test_ai_search_route_keeps_results_when_ranking_fails(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(app_module, "plan_tone", lambda prompt, **_: {"summary": "s", "advice": [], "gear": [], "search_queries": ["x"]})
    monkeypatch.setattr(app_module, "tone3000_search", lambda q, **_: [{"id": 1, "title": "t", "match_score": 1}])

    def boom(*a, **k):
        raise LocalLlmError("timeout")
    monkeypatch.setattr(app_module, "rank_packs", boom)
    data = client.post("/api/tone3000/ai_search", json={"prompt": "x"}).get_json()
    assert len(data["results"]) == 1 and "ranking unavailable" in data["warnings"][0]


def test_pack_chat_route_validates_and_passes_real_file_names(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(app_module, "tone3000_models", lambda tone_id: [{"id": 5, "name": "CLEAN"}])
    seen = {}

    def answer(question, pack, names, **kwargs):
        seen.update(pack=pack, names=names)
        return {"reply": "ok", "recommended_files": ["CLEAN"]}
    monkeypatch.setattr(app_module, "ask_about_pack", answer)
    assert client.post("/api/tone3000/ai_pack_chat", json={"tone_id": "1", "question": "q"}).status_code == 400
    data = client.post("/api/tone3000/ai_pack_chat", json={"tone_id": 1, "question": "q", "pack": {"title": "T", "tags": ["a"]}}).get_json()
    assert seen["names"] == ["CLEAN"] and seen["pack"]["title"] == "T"
    assert data["models"][0]["id"] == 5 and data["recommended_files"] == ["CLEAN"]


def test_pack_answer_accepts_a_short_name_only_when_it_is_unambiguous(fake_llm):
    replies, _ = fake_llm
    files = ["EDGY - Deluxe 65", "CLEAN - Deluxe 65", "CLEANEST - Deluxe 65"]
    replies.append({"reply": "r", "recommended_files": ["EDGY", "CLEAN"]})
    assert tone_search_ai.ask_about_pack("q", {}, files)["recommended_files"] == ["EDGY - Deluxe 65"]
