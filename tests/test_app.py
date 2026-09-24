"""API-level tests for the render-stage/blend-stage cost split enforced by
app.py: /api/render_pair is the only route allowed to touch NAM inference
(render() is faked out here, same as tests/test_pipeline_render.py, so any
route that accidentally invoked it would still "work" but this suite proves
the cache actually gets replaced correctly and that blend-stage routes never
need a fresh render).
"""
from __future__ import annotations

import io
import json as jsonlib
import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import app as app_module
from hybrid.core.render import SLIM_FULL, SLIM_LITE
import hybrid.modes.blend_training_target as blend_training_target
import hybrid.modes.cab_embed_training_target as cab_embed_training_target
import hybrid.core.pipeline as pipeline
import hybrid.modes.training_target as training_target

# Smallest bundled DI fixture (17.75s) -- keeps these tests fast since the
# causal envelope follower is a real (if cheap) per-sample computation.
DI_FILE = "high_thrash.wav"


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(pipeline, "render", fake_render)
    monkeypatch.setattr(training_target, "render", fake_render)
    monkeypatch.setattr(blend_training_target, "render", fake_render)
    monkeypatch.setattr(cab_embed_training_target, "render", fake_render)
    # Bypass the official-V3-file MD5 check for synthetic training-input
    # fixtures in these tests -- we don't ship the real ~27MB official file.
    monkeypatch.setattr(training_target, "_md5_file", lambda path: training_target.OFFICIAL_V3_INPUT_MD5)


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
    # Don't leak a rendered pair into unrelated tests/sessions.
    app_module._rendered_pair_cache["pair"] = None
    app_module._rendered_pair_cache["snapshot"] = None
    app_module._comparison_cache.clear()


def _write_fake_nam(path, input_level_dbu=None):
    data = {"architecture": "Test", "config": {}, "sample_rate": 48000}
    if input_level_dbu is not None:
        data["input_level_dbu"] = input_level_dbu
    path.write_text(jsonlib.dumps(data))


def test_selected_cab_warns_for_explicit_amp_cab_source_metadata(tmp_path):
    from hybrid.core.cab_ir import CabDesign
    source = tmp_path / "source.nam"
    source.write_text(jsonlib.dumps({"architecture": "Test", "config": {}, "sample_rate": 48000,
                                     "metadata": {"gear_type": "amp_cab"}}))
    warning = app_module._source_cabinet_warning(str(source), cab=CabDesign(selected=True))
    assert warning and "double-cabinet" in warning and source.name in warning


def test_setup_status_reports_a_configured_non_local_ai_provider_as_ready(client, monkeypatch):
    # provider != "local" never gets a `reachable` probe (see hybrid/services/local_llm.py's
    # status()) -- the checklist must not fall through to a bogus "start ollama
    # serve" message for a fully-configured Cloudflare/custom provider.
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True, "provider": "cloudflare", "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast"})
    items = client.get("/api/setup/status").get_json()["items"]
    llm_item = next(i for i in items if i["id"] == "local_llm")
    assert llm_item["ready"] is True
    assert "ollama" not in llm_item["detail"].lower()
    assert "Cloudflare Workers AI" in llm_item["detail"]


def test_local_llm_recipe_is_unavailable_until_a_model_is_configured(client, monkeypatch, tmp_path):
    # Isolate from a real .env on this machine (hybrid/services/env_file.py's deliberate fallback), same as tests/test_local_llm.py.
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "unused.env"))
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "")
    assert client.get("/api/local_llm/status").get_json()["enabled"] is False
    response = client.post("/api/local_llm/recipe", json={"prompt": "a clean crunch blend"})
    assert response.status_code == 503


def test_local_llm_recipe_rejects_oversized_conversation_history(client, monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "test-model")
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "a clean crunch blend",
        "history": [{"role": "user", "content": "x"}] * 9,
    })
    assert response.status_code == 400
    assert "at most 8" in response.get_json()["error"]


def test_local_llm_recipe_accepts_detailed_history_and_trims_it_for_the_model(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    seen = {}

    def fake_converse(_prompt, history, _research_notes="", **_kwargs):
        seen["history"] = history
        return SimpleNamespace(recipe=None, tone3000_queries=[], to_dict=lambda: {"reply": "Understood."})

    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which file should I use?",
        "history": [{"role": "assistant", "content": "x" * 2_500}],
        "research": {"web": False, "tone3000": False, "rig_scope": "anything", "author": ""},
    })

    assert response.status_code == 200
    # The endpoint permits the browser's full answer; local_llm.converse caps
    # it before sending to the model.
    assert len(seen["history"][0]["content"]) == 2_500


def test_local_llm_uses_ai_amp_queries_for_tone3000_research(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    calls = []

    def fake_converse(prompt, history, research_notes="", **kwargs):
        calls.append({"prompt": prompt, "research_notes": research_notes, **kwargs})
        if len(calls) == 1:
            return SimpleNamespace(
                tone3000_queries=["Vox AC30", "Marshall JCM800"],
                source_plan={"ampA": "Vox AC30", "ampB": "Marshall JCM800"},
                to_dict=lambda: {"reply": "Tell me your amps."},
            )
        return SimpleNamespace(
            tone3000_queries=None,
            to_dict=lambda: {"reply": "Try the Vox AC30 capture -- it fits the clean side."},
        )

    def fake_search(query, *, rig_scope, author, rank_query=""):
        return [{"id": query, "title": query + " capture", "creator": "tester", "description": "head"}]

    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    monkeypatch.setattr(app_module, "tone3000_search", fake_search)
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Give me Foo Fighters live clean and dirt tones",
        "research": {"tone3000": True, "web": False, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    # First call only proposes search terms; the real catalog matches are fed
    # back in on a second call so the FINAL reply can reference them by name.
    assert calls[0]["request_tone3000_queries"] is True
    assert calls[1]["request_tone3000_queries"] is False
    assert calls[0]["known_source_plan"] is None
    assert calls[1]["known_source_plan"] == {"ampA": "Vox AC30", "ampB": "Marshall JCM800"}
    assert "Vox AC30 capture" in calls[1]["research_notes"]
    assert "Marshall JCM800 capture" in calls[1]["research_notes"]
    data = response.get_json()
    assert data["reply"] == "Try the Vox AC30 capture -- it fits the clean side."
    assert [result["query"] for result in data["tone3000_results"]] == ["Vox AC30", "Marshall JCM800"]


def test_local_llm_debug_trace_includes_ai_context_and_research_without_secrets(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {
        "enabled": True, "provider": "cloudflare", "model": "test-model",
    })
    calls = 0

    def fake_converse(prompt, history, research_notes="", debug_trace=None, **_kwargs):
        nonlocal calls
        calls += 1
        if debug_trace is not None:
            debug_trace.update({
                "provider": "cloudflare",
                "model": "test-model",
                "messages": [{"role": "user", "content": prompt + "\n" + research_notes}],
            })
        if calls == 1:
            return SimpleNamespace(
                tone3000_queries=["Vox AC30"], source_plan=None,
                to_dict=lambda: {"reply": "Searching."},
            )
        return SimpleNamespace(
            tone3000_queries=None, source_plan=None,
            to_dict=lambda: {"reply": "Use the catalogue match."},
        )

    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    monkeypatch.setattr(app_module, "tone3000_search", lambda *_args, **_kwargs: [{
        "id": 7, "title": "Vox AC30 Clean", "creator": "tester", "description": "Clean head",
        "match_score": 90, "match_reason": "Metadata match.",
    }])

    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Find a clean Vox source",
        "history": [{"role": "assistant", "content": "Previous context"}],
        "research": {"tone3000": True, "web": False, "rig_scope": "heads", "author": ""},
        "include_debug": True,
    })

    assert response.status_code == 200
    debug = response.get_json()["debug"]
    assert [call["stage"] for call in debug["ai_calls"]] == ["initial", "final_with_tone3000_matches"]
    assert debug["request"]["history_received"] == [{"role": "assistant", "content": "Previous context"}]
    assert debug["research"]["tone3000_queries"] == ["Vox AC30"]
    assert debug["research"]["tone3000_results"][0]["title"] == "Vox AC30 Clean"
    serialized = jsonlib.dumps(debug)
    assert '"api_key"' not in serialized.lower()
    assert "Bearer " not in serialized


def test_local_llm_runs_focused_amp_discovery_when_first_pass_finds_no_amps(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True, "provider": "local"})
    web_queries = []
    catalogue_queries = []
    ai_prompts = []

    def fake_web_notes(query):
        web_queries.append(query)
        return "Blink-182 rig evidence: Mesa/Boogie Triple Rectifier and Marshall JCM900."

    def fake_converse(prompt, _history, _research_notes="", **_kwargs):
        ai_prompts.append(prompt)
        if prompt.startswith("Identify concrete amp-family search terms"):
            return SimpleNamespace(
                tone3000_queries=["Mesa/Boogie Triple Rectifier", "Marshall JCM900"],
                source_plan={"ampA": "Marshall JCM900 clean", "ampB": "Mesa/Boogie Triple Rectifier driven"},
                to_dict=lambda: {"reply": "Found the amp families."},
            )
        if len(ai_prompts) == 1:
            return SimpleNamespace(
                tone3000_queries=[], source_plan=None,
                to_dict=lambda: {"reply": "Which captures do you want?"},
            )
        return SimpleNamespace(
            tone3000_queries=None,
            source_plan={"ampA": "Marshall JCM900 clean", "ampB": "Mesa/Boogie Triple Rectifier driven"},
            to_dict=lambda: {"reply": "Use the researched captures."},
        )

    def fake_search(query, **_kwargs):
        catalogue_queries.append(query)
        return [{
            "id": len(catalogue_queries), "title": query + " capture", "creator": "tester",
            "description": "head", "match_score": 90, "match_reason": "Metadata match.",
        }]

    monkeypatch.setattr(app_module, "web_notes", fake_web_notes)
    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    monkeypatch.setattr(app_module, "tone3000_search", fake_search)

    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Clean Blink-182 All the Small Things into boosted distortion",
        "research": {"tone3000": True, "web": True, "rig_scope": "heads", "author": ""},
        "include_debug": True,
    })

    assert response.status_code == 200
    assert len(web_queries) == 2
    assert web_queries[1].startswith("Which specific guitar amplifier makes and models")
    assert catalogue_queries == ["Mesa/Boogie Triple Rectifier", "Marshall JCM900"]
    data = response.get_json()
    assert data["source_plan"]["ampB"] == "Mesa/Boogie Triple Rectifier driven"
    assert data["debug"]["research"]["focused_amp_discovery"]["queries_returned"] == catalogue_queries


def test_local_llm_uses_the_players_description_when_no_catalogue_term_is_proposed(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    searched = []

    def fake_converse(*_args, **_kwargs):
        return SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "A starting point."})

    def fake_search(query, *, rig_scope, author, rank_query=""):
        searched.append(query)
        return []

    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    monkeypatch.setattr(app_module, "tone3000_search", fake_search)
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "I want Dave Grohl clean to distorted tones",
        "research": {"tone3000": True, "web": False, "rig_scope": "anything", "author": ""},
    })

    assert response.status_code == 200
    assert searched == ["I want Dave Grohl clean to distorted tones"]


def test_local_llm_uses_model_queries_without_hard_coded_amp_overrides(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    searched = []

    def fake_converse(*_args, **_kwargs):
        return SimpleNamespace(tone3000_queries=["Marshall JCM800"], recipe=None, to_dict=lambda: {"reply": "Starting point."})

    def fake_search(query, *, rig_scope, author, rank_query=""):
        searched.append(query)
        return []

    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    monkeypatch.setattr(app_module, "tone3000_search", fake_search)
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Start with a clean Vox AC30, then move to a JCM800.",
        "research": {"tone3000": True, "web": False, "rig_scope": "anything", "author": ""},
    })

    assert response.status_code == 200
    assert searched == ["Marshall JCM800"]


def test_local_llm_accepts_a_selected_tone3000_pack_outside_the_user_prompt_limit(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    captured = {}

    def fake_converse(prompt, _history, research_notes="", **_kwargs):
        captured["prompt"] = prompt
        captured["notes"] = research_notes
        return SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "Try Clean.nam."})

    monkeypatch.setattr(app_module, "converse_with_local_llm", fake_converse)
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which model should I use?",
        "tone3000_context": {
            "id": 42, "title": "Example pack", "creator": "tester", "description": "A useful pack",
            "models": ["Clean.nam"] * 30,
        },
        "research": {"web": False, "tone3000": False, "rig_scope": "anything", "author": ""},
    })

    assert response.status_code == 200
    assert captured["prompt"] == "Which model should I use?"
    assert "Selected TONE3000 pack" in captured["notes"]
    assert "Model: Clean.nam" in captured["notes"]


def test_selected_tone3000_pack_without_a_role_requests_one_before_recommending(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(
        app_module, "converse_with_local_llm",
        lambda *_args, **_kwargs: SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "generic"}),
    )
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which specific model should be Amp A or Amp B?",
        "tone3000_context": {
            "id": 42, "title": "AC30", "creator": "tester", "description": "", 
            "models": ["AC30_CLEAN", "AC30_EDGE", "AC30_CRUNCH", "AC30_HOT"],
        },
        "research": {"web": False, "tone3000": True, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    assert "clean foundation (Amp A)" in response.get_json()["reply"]
    assert response.get_json()["tone3000_results"] == []


def test_selected_amp_a_pack_does_not_replace_the_existing_amp_b(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(
        app_module, "converse_with_local_llm",
        lambda *_args, **_kwargs: SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "generic"}),
    )
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which file should I use as Amp A or Amp B?",
        "history": [{"role": "assistant", "content": "Use Vox AC30 CC2 as Amp A (clean source) and Mesa Dual Rectifier as Amp B (driven source)."}],
        "tone3000_context": {
            "id": 42, "title": "Vox AC30 CC2", "creator": "tester", "description": "",
            "models": ["AC30_CLEAN", "AC30_EDGE", "AC30_CRUNCH", "AC30_HOT"],
        },
        "research": {"web": False, "tone3000": True, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    reply = response.get_json()["reply"]
    assert "`AC30_CLEAN`" in reply
    assert "Do not use another file from this pack as Amp B" in reply


def test_selected_amp_b_pack_keeps_the_existing_amp_a(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(
        app_module, "converse_with_local_llm",
        lambda *_args, **_kwargs: SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "generic"}),
    )
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which file should I use?",
        "history": [{"role": "assistant", "content": (
            "**Source Roles:**\n* **Amp A (Clean):** `1973 Fender Twin Reverb by mick8187`\n"
            "* **Amp B (Driven):** `Marshall - JCM800 (Kerry King Signature) [18dBu] by kenazmusic`"
        )}],
        "tone3000_context": {
            "id": 42, "title": "Marshall - JCM800 (Kerry King Signature) [18dBu]", "creator": "kenazmusic", "description": "",
            "models": ["Marshall - JCM800 KKS EOB [18dBu]", "Marshall - JCM800 KKS (TS9) HG [18dBu]"],
        },
        "research": {"web": False, "tone3000": True, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    reply = response.get_json()["reply"]
    assert "already-chosen **Amp B**" in reply
    assert "Keep the previously selected clean amp pack as Amp A" in reply


def test_related_amp_variant_in_the_source_plan_keeps_its_amp_b_role(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(
        app_module, "converse_with_local_llm",
        lambda *_args, **_kwargs: SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "generic"}),
    )
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which specific file should I use?",
        "history": [{"role": "assistant", "content": (
            "The complete identity shifts from the Fender source (Amp A) to the Marshall source (Amp B) as guitar volume increases.\n"
            "* **Amp A (Low Volume):** Fender Deluxe Reverb '65 Reissue (Clean Source)\n"
            "* **Amp B (High Volume):** Marshall JCM 800 1959 100W Pack (Driven Source)"
        )}],
        "tone3000_context": {
            "id": 1071, "title": "Marshall JCM 800 2203", "creator": "arthm", "description": "",
            "models": ["JCM800 2203 - P5 B5 M5 T5 MV6 G5 - AZG - 700", "JCM800 2203 - P5 B5 M5 T5 MV5 G9 - AZG - 700"],
        },
        "research": {"web": False, "tone3000": True, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    assert "already-chosen **Amp B**" in response.get_json()["reply"]


def test_durable_source_plan_tags_a_discussed_pack_without_using_history(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(
        app_module, "converse_with_local_llm",
        lambda *_args, **_kwargs: SimpleNamespace(recipe=None, tone3000_queries=[], to_dict=lambda: {"reply": "generic"}),
    )
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which file should I use?",
        "source_plan": {"ampA": "Fender Deluxe Reverb", "ampB": "Marshall JCM 800"},
        "tone3000_context": {
            "id": 1071, "title": "Marshall JCM 800 2203", "creator": "arthm", "description": "Flat Marshall capture",
            "models": ["JCM800 2203 G1", "JCM800 2203 G9"],
        },
        "research": {"web": False, "tone3000": True, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["selected_source_role"] == "b"
    assert data["source_plan"] == {"ampA": "Fender Deluxe Reverb", "ampB": "Marshall JCM 800"}


def test_ambiguous_selected_pack_asks_for_its_role_instead_of_using_it_twice(client, monkeypatch):
    monkeypatch.setattr(app_module, "local_llm_status", lambda: {"enabled": True})
    monkeypatch.setattr(
        app_module, "converse_with_local_llm",
        lambda *_args, **_kwargs: SimpleNamespace(tone3000_queries=[], recipe=None, to_dict=lambda: {"reply": "generic"}),
    )
    response = client.post("/api/local_llm/recipe", json={
        "prompt": "Which file should I use?",
        "tone3000_context": {
            "id": 42, "title": "Example pack", "creator": "tester", "description": "",
            "models": ["Clean", "Crunch"],
        },
        "research": {"web": False, "tone3000": True, "rig_scope": "heads", "author": ""},
    })

    assert response.status_code == 200
    assert "clean foundation (Amp A)" in response.get_json()["reply"]


def test_tone3000_pack_models_are_exposed_as_direct_downloads(client, monkeypatch):
    monkeypatch.setattr(app_module, "tone3000_models", lambda tone_id: [{
        "id": 12, "name": "Crunch 6.nam", "architecture": 2,
    }] if tone_id == 42 else [])

    response = client.get("/api/tone3000/tones/42/models")

    assert response.status_code == 200
    assert response.get_json()["models"][0]["name"] == "Crunch 6.nam"


def test_tone3000_model_download_proxies_the_selected_pack_member(client, monkeypatch):
    monkeypatch.setattr(app_module, "tone3000_model_download", lambda tone_id, model_id: (b"nam-data", "Clean.nam"))

    response = client.get("/api/tone3000/tones/42/models/12/download")

    assert response.status_code == 200
    assert response.data == b"nam-data"
    assert "attachment" in response.headers["Content-Disposition"]


def _write_tool_nam(path):
    path.write_text(jsonlib.dumps({
        "architecture": "SlimmableContainer",
        "config": {"submodels": [{"model": {"config": {"head_scale": 0.0051461088670930214, "weights": [1]}, "metadata": {"loudness": -22.8}}}]},
        "metadata": {"loudness": -22.7, "gain": 3.0},
    }))


def _render_body(amp_a, amp_b, **overrides):
    body = {
        "amp_a_path": str(amp_a),
        "amp_b_path": str(amp_b),
        "di_file": DI_FILE,
        "instrument_type": "guitar",
        "input_profile_id": "vintage_humbucker",
        "calibration_mode": "auto",
    }
    body.update(overrides)
    return body


def _current_render_id():
    return app_module._rendered_pair_cache["snapshot"]["render_id"]


def test_preview_without_render_pair_first_returns_400(client):
    app_module._rendered_pair_cache["pair"] = None
    resp = client.post("/api/preview", json={"source": "a"})
    assert resp.status_code == 400
    assert "render" in resp.get_json()["error"].lower()


def test_renderer_readiness_reports_missing_binary_without_attempting_inference(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_nam_render_exe", lambda: (_ for _ in ()).throw(app_module.NamRenderError("not found")))
    response = client.get("/api/renderer/readiness")
    assert response.status_code == 200
    assert response.get_json() == {"found": False, "verified": False, "error": "not found"}


def test_renderer_readiness_accepts_namcore_usage_exit(client, monkeypatch):
    class Result:
        returncode, stdout, stderr = 1, "Usage: render <model.nam> <input.wav>", ""
    monkeypatch.setattr(app_module, "find_nam_render_exe", lambda: Path("/tmp/nam_render"))
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: Result())
    assert client.get("/api/renderer/readiness").get_json() == {
        "found": True, "verified": True, "path": "/tmp/nam_render",
    }


def test_renderer_readiness_distinguishes_found_but_unusable_binary(client, monkeypatch):
    class Result:
        returncode, stdout, stderr = 126, "", "permission denied"
    monkeypatch.setattr(app_module, "find_nam_render_exe", lambda: Path("/tmp/broken-renderer"))
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: Result())
    assert client.get("/api/renderer/readiness").get_json() == {
        "found": True, "verified": False, "path": "/tmp/broken-renderer", "error": "permission denied",
    }


def test_renderer_readiness_can_recover_on_retry_without_server_restart(client, monkeypatch):
    calls = 0
    def find_renderer():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise app_module.NamRenderError("not installed")
        return Path("/tmp/nam_render")
    class Result:
        returncode, stdout, stderr = 1, "Usage: render <model.nam> <input.wav>", ""
    monkeypatch.setattr(app_module, "find_nam_render_exe", find_renderer)
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: Result())

    assert client.get("/api/renderer/readiness").get_json()["verified"] is False
    recovered = client.get("/api/renderer/readiness").get_json()
    assert recovered["verified"] is True
    assert recovered["path"] == "/tmp/nam_render"


def test_oversized_upload_returns_a_clear_json_error(client, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 1)
    response = client.post(
        "/api/nam/upload",
        data={"file": (io.BytesIO(b"{}"), "too-large.nam")},
    )
    assert response.status_code == 413
    assert response.get_json()["error"] == "upload exceeds the 256 MiB limit"


def test_file_backed_session_embeds_nam_for_download_and_tools(client, tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    model_dir = session_dir / "models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    nam_bytes = jsonlib.dumps({"architecture": "WaveNet", "config": {"head_scale": 1.0}}).encode()
    session = {
        "type": "nam-mixer-session", "version": 1,
        "id": "saved-session", "name": "Saved session", "savedAt": "2026-09-09T10:00:00Z",
        "settings": {"mode": "hybrid"},
        "artifact": {"filename": "saved.nam", "nam_base64": base64.b64encode(nam_bytes).decode()},
    }
    saved = client.post("/api/sessions", json=session)
    assert saved.status_code == 201
    listed = client.get("/api/sessions").get_json()
    assert listed[0]["artifact"]["downloadUrl"] == "/api/sessions/saved-session/nam/download"
    assert listed[0]["artifact"]["toolPath"] == str(model_dir / "saved-session.nam")
    assert listed[0]["artifact"]["sha256"] == hashlib.sha256(nam_bytes).hexdigest()
    assert "nam_base64" not in listed[0]["artifact"]
    assert client.get("/api/sessions/saved-session/nam/download").data == nam_bytes
    tool_inspect = client.post("/api/nam/tools/inspect", json={"path": listed[0]["artifact"]["toolPath"]})
    assert tool_inspect.status_code == 200
    assert client.delete("/api/sessions/saved-session").status_code == 204
    assert not (model_dir / "saved-session.nam").exists()


def test_deleting_a_session_with_an_active_kaggle_job_requires_explicit_confirmation(client, tmp_path, monkeypatch):
    """The first DELETE must fail with a machine-readable flag (not just a
    string the frontend has to pattern-match) so it can show a SEPARATE,
    specific warning about cancelling the Kaggle job -- then actually
    cancel it once the caller passes ?cancel_active_jobs=1, rather than
    leaving an orphaned Kaggle kernel/dataset with no session managing it."""
    from hybrid.training.kaggle_training import KaggleJob, save_job

    session_dir = tmp_path / "sessions"
    model_dir = session_dir / "models"
    a2_dir = tmp_path / "a2"
    model_dir.mkdir(parents=True)
    a2_dir.mkdir()
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module._kaggle_manager, "a2_output_dir", a2_dir)

    session = {
        "type": "nam-mixer-session", "version": 1, "id": "active-kaggle-session",
        "name": "Active Kaggle session", "savedAt": "2026-09-18T10:00:00Z",
        "settings": {"mode": "hybrid"}, "designId": "design-1",
    }
    assert client.post("/api/sessions", json=session).status_code == 201

    job = KaggleJob(job_id="job-1", design_id="design-1", state="running",
                     dataset_ref="user/dataset-1", kernel_ref="user/kernel-1")
    save_job(a2_dir, job)

    calls = []
    monkeypatch.setattr(app_module._kaggle_manager.cli, "datasets_delete",
                         lambda ref: (calls.append(("dataset", ref)), SimpleNamespace(ok=True, stdout="", stderr=""))[1])
    monkeypatch.setattr(app_module._kaggle_manager.cli, "kernels_delete",
                         lambda ref: (calls.append(("kernel", ref)), SimpleNamespace(ok=True, stdout="", stderr=""))[1])

    blocked = client.delete("/api/sessions/active-kaggle-session")
    assert blocked.status_code == 409
    assert blocked.get_json()["active_kaggle_job"] is True
    assert calls == []  # must not touch Kaggle resources without the explicit flag

    confirmed = client.delete("/api/sessions/active-kaggle-session?cancel_active_jobs=1")
    assert confirmed.status_code == 204
    assert ("dataset", "user/dataset-1") in calls
    assert ("kernel", "user/kernel-1") in calls
    assert not (session_dir / "active-kaggle-session.nam-mixer.json").exists()


def test_session_validation_report_must_match_embedded_nam(client, tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    model_dir = session_dir / "models"
    a2_output_dir = tmp_path / "a2"
    model_dir.mkdir(parents=True)
    a2_output_dir.mkdir()
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_output_dir)
    nam_bytes = b'{"architecture":"WaveNet","config":{}}'
    sha256 = hashlib.sha256(nam_bytes).hexdigest()
    base = {
        "type": "nam-mixer-session", "version": 1, "id": "bound-report",
        "name": "Bound report", "savedAt": "2026-09-10T10:00:00Z",
        "settings": {"mode": "hybrid"},
        "artifact": {"filename": "model.nam", "nam_base64": base64.b64encode(nam_bytes).decode()},
    }
    mismatched = {
        **base,
        "validationReport": {"schema_version": 2, "model_sha256": "0" * 64, "state": "passed"},
    }
    rejected = client.post("/api/sessions", json=mismatched)
    assert rejected.status_code == 400
    assert "different NAM artifact" in rejected.get_json()["error"]

    matching = {
        **base,
        "validationReport": {"schema_version": 2, "model_sha256": sha256, "state": "passed"},
    }
    saved = client.post("/api/sessions", json=matching)
    assert saved.status_code == 201
    restored = client.get("/api/sessions").get_json()[0]
    assert restored["artifact"]["sha256"] == sha256
    assert restored["validationReport"]["model_sha256"] == sha256


def _isolate_uploads(tmp_path, monkeypatch):
    nam_dir, cab_dir = tmp_path / "uploaded_nam", tmp_path / "uploaded_cab"
    nam_dir.mkdir(); cab_dir.mkdir()
    monkeypatch.setattr(app_module, "NAM_UPLOAD_DIR", nam_dir)
    monkeypatch.setattr(app_module, "CAB_UPLOAD_DIR", cab_dir)
    return nam_dir, cab_dir


def _age(path: Path, seconds: float) -> None:
    import os
    now = __import__("time").time()
    os.utime(path, (now - seconds, now - seconds))


def test_deleting_a_session_sweeps_its_orphaned_uploads_but_keeps_shared_and_recent_ones(client, tmp_path, monkeypatch):
    """The exact bug this guards: work/uploaded_nam and work/uploaded_cab accumulate forever because nothing ever ties them
    to session lifecycle. Deleting the last session pointing at a file must now free it -- but never a file another surviving
    session still uses, and never one uploaded too recently to have been saved into a session yet."""
    session_dir = tmp_path / "sessions"; model_dir = session_dir / "models"; a2_dir = tmp_path / "a2"
    model_dir.mkdir(parents=True); a2_dir.mkdir()
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    nam_dir, cab_dir = _isolate_uploads(tmp_path, monkeypatch)

    only_a = nam_dir / "only-used-by-a.nam"; only_a.write_text("{}")
    shared = nam_dir / "shared-by-a-and-b.nam"; shared.write_text("{}")
    cab = cab_dir / "cab-used-by-a.wav"; cab.write_text("RIFF")
    recent_orphan = nam_dir / "just-uploaded-not-saved-yet.nam"; recent_orphan.write_text("{}")
    for f in (only_a, shared, cab, recent_orphan):
        _age(f, 4000)  # old enough to sweep, except recent_orphan which we re-age below
    _age(recent_orphan, 5)  # uploaded 5s ago: must survive even though no session references it yet

    session_a = {"type": "nam-mixer-session", "version": 1, "id": "a", "name": "A", "savedAt": "2026-09-22T00:00:00Z",
                 "settings": {"mode": "hybrid", "ampA": {"path": str(only_a)}, "ampB": {"path": str(shared)},
                             "cab": {"path": str(cab)}}}
    session_b = {"type": "nam-mixer-session", "version": 1, "id": "b", "name": "B", "savedAt": "2026-09-22T00:00:00Z",
                 "settings": {"mode": "hybrid", "ampA": {"path": str(shared)}, "ampB": {"path": ""}}}
    assert client.post("/api/sessions", json=session_a).status_code == 201
    assert client.post("/api/sessions", json=session_b).status_code == 201

    assert client.delete("/api/sessions/a").status_code == 204
    assert not only_a.exists(), "no remaining session references it -> swept"
    assert not cab.exists(), "no remaining session references the cab either -> swept"
    assert shared.exists(), "session b still references it -> kept"
    assert recent_orphan.exists(), "uploaded moments ago, unreferenced by design -> kept (grace period)"

    assert client.delete("/api/sessions/b").status_code == 204
    assert not shared.exists(), "the last session referencing it is gone -> swept"


def test_render_sources_sweep_keeps_only_folders_a_remaining_bundle_references(tmp_path, monkeypatch):
    """work/render_sources holds a copy per render/preview -- most never become a saved bundle, so this is swept both at app
    startup and after a session delete, not tied to any single session's own lifecycle the way uploads are."""
    a2_dir = tmp_path / "a2"; design_dir = a2_dir / "design-1"; design_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module, "WORK_DIR", tmp_path)
    rs = tmp_path / "render_sources"; rs.mkdir()
    used = rs / "aaa111" / "amp-a.nam"; used.parent.mkdir(); used.write_text("{}")
    di_only = rs / "bbb222" / "some-di.wav"; di_only.parent.mkdir(); di_only.write_text("RIFF")  # DI copies are never persisted
    for f in (used, di_only):
        _age(f.parent, 4000); _age(f, 4000)
    (design_dir / "training_manifest.json").write_text(jsonlib.dumps(
        {"amp_a": {"path": str(used)}, "amp_b": {"path": ""}}))
    removed = app_module._sweep_orphaned_render_sources()
    assert used.parent.exists() and used.exists()
    assert not di_only.parent.exists()
    assert str(di_only.parent) in removed


def test_deleting_a_session_also_sweeps_render_sources_its_bundle_owned(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"; model_dir = session_dir / "models"; a2_dir = tmp_path / "a2"
    model_dir.mkdir(parents=True); a2_dir.mkdir()
    design_dir = a2_dir / "design-1"; design_dir.mkdir()
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module, "WORK_DIR", tmp_path)
    _isolate_uploads(tmp_path, monkeypatch)
    rs = tmp_path / "render_sources"; rs.mkdir()
    owned = rs / "ccc333" / "amp.nam"; owned.parent.mkdir(); owned.write_text("{}")
    _age(owned.parent, 4000); _age(owned, 4000)
    (design_dir / "training_manifest.json").write_text(jsonlib.dumps({"amp_a": {"path": str(owned)}, "amp_b": {"path": ""}}))
    session = {"type": "nam-mixer-session", "version": 1, "id": "design-1", "name": "Gen", "savedAt": "2026-09-22T00:00:00Z",
               "settings": {"mode": "hybrid"}, "designId": "design-1"}
    (design_dir / "nam-mixer-session.json").write_text(jsonlib.dumps(session))
    c = app_module.app.test_client()
    assert c.delete("/api/sessions/design-1").status_code == 204
    assert not owned.parent.exists(), "the owning bundle is gone -> its render source is swept too"


def test_upload_sweep_also_checks_generated_sessions(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"; a2_dir = tmp_path / "a2"; design_dir = a2_dir / "design-1"
    session_dir.mkdir(); design_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    nam_dir, _cab_dir = _isolate_uploads(tmp_path, monkeypatch)
    used = nam_dir / "used-by-generated-session.nam"; used.write_text("{}"); _age(used, 4000)
    unused = nam_dir / "unused.nam"; unused.write_text("{}"); _age(unused, 4000)
    (design_dir / "nam-mixer-session.json").write_text(jsonlib.dumps(
        {"settings": {"mode": "hybrid", "ampA": {"path": str(used)}, "ampB": {"path": ""}}}))
    removed = app_module._sweep_orphaned_uploads()
    assert str(used) not in removed and used.exists()
    assert str(unused) in removed and not unused.exists()


def test_update_check_route_reports_an_available_update(client, monkeypatch):
    from hybrid.services.update_check import UpdateCheckResult
    monkeypatch.setattr(app_module, "check_for_update", lambda current: UpdateCheckResult(
        current_version=current, latest_version="v9.9.9", update_available=True,
        release_url="https://github.com/daverage/nam-mixer/releases/tag/v9.9.9", asset_url="https://example.invalid/asset.dmg"))
    data = client.get("/api/update/check").get_json()
    assert data == {"ok": True, "current_version": app_module.APP_VERSION, "latest_version": "v9.9.9", "update_available": True,
                    "release_url": "https://github.com/daverage/nam-mixer/releases/tag/v9.9.9",
                    "asset_url": "https://example.invalid/asset.dmg", "is_packaged": False}


def test_update_check_route_reports_a_ui_safe_error_without_raising(client, monkeypatch):
    from hybrid.services.update_check import UpdateCheckError
    def boom(current):
        raise UpdateCheckError("Could not reach GitHub to check for updates: no route to host")
    monkeypatch.setattr(app_module, "check_for_update", boom)
    response = client.get("/api/update/check")
    assert response.status_code == 200  # never a 500 -- the frontend shows response.error directly
    assert response.get_json() == {"ok": False, "error": "Could not reach GitHub to check for updates: no route to host"}


def test_update_check_route_never_hits_the_real_network(client):
    """Guards against a future edit accidentally removing the monkeypatch seam: this route is the app's one
    deliberate exception to "no network unless the user asks", so it must go through urlopen exactly once per
    call, never as a side effect of import/app-startup/collection -- proven here by having urlopen itself fail
    and confirming the route still only reports a UI-safe error, rather than the test suite having quietly made
    a real network call before this point."""
    import hybrid.services.update_check as update_check_module
    with patch.object(update_check_module, "urlopen", side_effect=OSError("must not hit the real network")) as mock_urlopen:
        response = client.get("/api/update/check")
    assert mock_urlopen.call_count == 1
    assert response.status_code == 200
    assert response.get_json() == {"ok": False, "error": "Could not reach GitHub to check for updates: must not hit the real network"}


def test_session_rejects_declared_artifact_hash_mismatch(client, tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    model_dir = session_dir / "models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(app_module, "SESSION_MODEL_DIR", model_dir)
    session = {
        "type": "nam-mixer-session", "version": 1, "id": "bad-hash",
        "name": "Bad hash", "savedAt": "2026-09-10T10:00:00Z",
        "settings": {},
        "artifact": {"filename": "model.nam", "nam_base64": base64.b64encode(b"model").decode(), "sha256": "f" * 64},
    }
    response = client.post("/api/sessions", json=session)
    assert response.status_code == 400
    assert not (model_dir / "bad-hash.nam").exists()


def test_generated_session_is_stored_in_and_deletes_its_bundle(client, tmp_path, monkeypatch):
    a2_dir = tmp_path / "a2"
    bundle = a2_dir / "demo"
    bundle.mkdir(parents=True)
    (bundle / "training_manifest.json").write_text(jsonlib.dumps({"mode": "hybrid", "model_name": "Demo", "amp_a": {}, "amp_b": {}, "design": {}}))
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    listed = client.get("/api/sessions").get_json()
    assert listed[0]["generated"] is True
    assert (bundle / "nam-mixer-session.json").is_file()
    assert client.delete(f"/api/sessions/{listed[0]['id']}").status_code == 204
    assert not bundle.exists()


def test_renaming_loaded_generated_session_updates_untrained_manifest(client, tmp_path, monkeypatch):
    a2_dir = tmp_path / "a2"
    bundle = a2_dir / "demo"
    bundle.mkdir(parents=True)
    manifest_path = bundle / "training_manifest.json"
    manifest_path.write_text(jsonlib.dumps({
        "mode": "hybrid", "model_name": "Demo", "artifact_stem": "Demo",
        "artifact_filename": "Demo.nam", "amp_a": {}, "amp_b": {}, "design": {},
    }))
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    session = client.get("/api/sessions").get_json()[0]
    session["settings"]["modelName"] = "Renamed Final Build"
    session["name"] = "Renamed Final Build"

    saved = client.post("/api/sessions", json=session)

    assert saved.status_code == 201
    manifest = jsonlib.loads(manifest_path.read_text())
    assert manifest["model_name"] == "Renamed Final Build"
    assert manifest["artifact_stem"] == "Renamed_Final_Build"
    assert manifest["artifact_filename"] == "Renamed_Final_Build.nam"
    assert saved.get_json()["name"] == "Renamed Final Build"


def test_rejected_training_start_preserves_running_bundle_protection(client, tmp_path, monkeypatch):
    from hybrid.training.local_training import LocalTrainingManager

    a2_dir = tmp_path / "a2"
    for design in ("running-A", "other-B"):
        bundle = a2_dir / design
        bundle.mkdir(parents=True)
        (bundle / "training_manifest.json").write_text(jsonlib.dumps({"mode": "hybrid", "amp_a": {}, "amp_b": {}, "design": {}}))
    manager = LocalTrainingManager(tmp_path, a2_dir)
    manager.manifest_path = a2_dir / "running-A" / "training_manifest.json"
    manager.process = SimpleNamespace(poll=lambda: None)
    manager.state = "training"
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module, "_local_training_manager", manager)
    sessions = client.get("/api/sessions").get_json()
    running_session = next(item for item in sessions if item["designId"] == "running-A")

    response = client.post("/api/local_training/start", json={"design_id": "other-B"})
    assert response.status_code == 400
    assert manager.design_id == "running-A"
    response = client.delete(f"/api/sessions/{running_session['id']}")
    assert response.status_code == 409
    assert manager.manifest_path.is_file()


def test_nam_volume_tool_writes_only_a_new_validated_file(client, tmp_path):
    source = tmp_path / "Mesa.nam"
    _write_tool_nam(source)
    uploaded = client.post("/api/nam/upload", data={"file": (io.BytesIO(source.read_bytes()), "Mesa.nam")}).get_json()
    response = client.post("/api/nam/tools/volume", json={"path": uploaded["path"], "db_change": 6})
    assert response.status_code == 200
    data = response.get_json()
    assert data["changed_paths"] == [
        "config.submodels[0].model.config.head_scale",
        "config.submodels[0].model.metadata.loudness",
        "metadata.loudness",
    ]
    assert data["filename"] == "Mesa_+6dB.nam"
    downloaded = client.get(data["download_url"])
    assert downloaded.status_code == 200
    edited = jsonlib.loads(downloaded.data)
    assert edited["metadata"]["gain"] == 3.0
    assert edited["config"]["submodels"][0]["model"]["config"]["weights"] == [1]


def test_nam_volume_validation_ignores_json_object_key_order(client, tmp_path):
    source = tmp_path / "MetadataFirst.nam"
    source.write_text(jsonlib.dumps({
        "architecture": "SlimmableContainer",
        # Some real exporters put metadata before config. The approved paths
        # are identical after saving even though their traversal order differs.
        "metadata": {"loudness": -20.5, "name": "Metadata First"},
        "config": {"submodels": [{
            "model": {
                "metadata": {"loudness": -20.5},
                "config": {"head_scale": 0.01, "weights": [1]},
            },
        }]},
    }))
    uploaded = client.post(
        "/api/nam/upload",
        data={"file": (io.BytesIO(source.read_bytes()), source.name)},
    ).get_json()

    response = client.post(
        "/api/nam/tools/volume",
        json={"path": uploaded["path"], "db_change": 3},
    )

    assert response.status_code == 200, response.get_json()
    assert set(response.get_json()["changed_paths"]) == {
        "metadata.loudness",
        "config.submodels[0].model.metadata.loudness",
        "config.submodels[0].model.config.head_scale",
    }


def test_nam_metadata_tool_edits_descriptive_fields_only(client, tmp_path):
    source = tmp_path / "Meta.nam"
    _write_tool_nam(source)
    uploaded = client.post("/api/nam/upload", data={"file": (io.BytesIO(source.read_bytes()), "Meta.nam")}).get_json()
    response = client.post("/api/nam/tools/metadata", json={"path": uploaded["path"], "metadata": {"name": "Battery", "modeled_by": "Test", "gear_model": "Mark IIC+"}})
    assert response.status_code == 200
    edited = jsonlib.loads(client.get(response.get_json()["download_url"]).data)
    assert edited["metadata"]["name"] == "Battery"
    assert edited["metadata"]["modeled_by"] == "Test"
    assert edited["metadata"]["gear_model"] == "Mark IIC+"
    assert edited["metadata"]["gain"] == 3.0


def test_nam_tools_inspect_works_for_embedded_cab_sequential_export(client, tmp_path):
    # Regression test: an embedded-cab export's architecture is "Sequential"
    # (hybrid/training/sequential_nam.py), which find_output_scalers() can't find a
    # head_scale for -- that must not block the metadata editor from
    # loading at all (it previously raised a hard error, see the NAM Tools
    # UI bug report this test guards against).
    source = tmp_path / "model-embedded-experimental.nam"
    source.write_text(jsonlib.dumps({
        "architecture": "Sequential",
        "config": {"models": []},
        "metadata": {"name": "British American High Gain + Cab", "gear_type": "amp_cab", "loudness": -16.3},
    }), encoding="utf-8")
    uploaded = client.post("/api/nam/upload", data={"file": (io.BytesIO(source.read_bytes()), source.name)}).get_json()
    response = client.post("/api/nam/tools/inspect", json={"path": uploaded["path"]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["architecture"] == "Sequential"
    assert data["head_scales"] == []
    assert "Sequential" in data["volume_unsupported_reason"]
    assert data["metadata"]["name"] == "British American High Gain + Cab"


def test_wizard_insight_requires_a_rendered_pair(client):
    app_module._rendered_pair_cache["pair"] = None
    resp = client.post("/api/wizard/insight", json={})
    assert resp.status_code == 400
    assert "render" in resp.get_json()["error"].lower()


def test_wizard_insight_describes_a_rendered_pair(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    resp = client.post("/api/wizard/insight", json={"render_id": _current_render_id()})
    assert resp.status_code == 200
    data = resp.get_json()
    assert {"level_text", "tone_text", "feel_text", "amp_a", "amp_b"} <= data.keys()


def test_live_blend_stems_returns_trimmed_stereo_pair_without_rerender(client, tmp_path):
    """The live-audition route must reuse the cached NAM output and leave the
    browser to perform only the final, adjustable linear mix."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    response = client.post("/api/live_blend_stems", json={"render_id": _current_render_id(),
        "mix_b": 0.5, "auto_level": True, "manual_b_trim_db": 0.0,
    })

    assert response.status_code == 200
    assert response.headers["X-Live-Audition"] == "fixed-blend-stems"
    assert response.headers["X-Effective-Trim-Db"] == "0.000"
    audio, sample_rate = sf.read(io.BytesIO(response.data), dtype="float32", always_2d=True)
    assert sample_rate == 48_000
    assert audio.shape[1] == 2
    assert np.allclose(audio[:, 0], audio[:, 1])


def test_render_pair_cache_reflects_the_newest_profile_not_the_old_one(client, tmp_path):
    """Re-rendering with a different profile must REPLACE the cached pair --
    a stale /api/preview response here would mean the server is serving an
    old render instead of the one that was just requested."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp0 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="vintage_humbucker"))
    assert resp0.status_code == 200
    data0 = resp0.get_json()
    assert data0["input_profile_gain_db"] == 0.0
    audio_at_0db = client.post("/api/preview", json={"source": "a", "render_id": data0["render_id"]}).data

    resp1 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="hot_humbucker"))
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert data1["input_profile_gain_db"] == 4.5
    assert data1["input_peak_dbfs"] > data0["input_peak_dbfs"]
    audio_at_hot = client.post("/api/preview", json={"source": "a", "render_id": data1["render_id"]}).data

    assert audio_at_0db != audio_at_hot


def test_stale_render_id_cannot_preview_or_generate_a_newer_pair(client, tmp_path, isolated_training_paths):
    training_path, _ = isolated_training_paths
    _write_training_wav(training_path)
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    old = client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).get_json()
    new = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db=2.0)).get_json()
    assert old["render_id"] != new["render_id"]
    preview = client.post("/api/preview", json={"source": "a", "render_id": old["render_id"]})
    generate = client.post("/api/generate", json={"render_id": old["render_id"], "model_name": "must-not-exist"})
    assert preview.status_code == generate.status_code == 409
    assert preview.get_json()["code"] == generate.get_json()["code"] == "stale_render"


def test_render_pair_applies_test_gain_db_as_real_additional_gain(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp0 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db=0.0))
    assert resp0.status_code == 200
    data0 = resp0.get_json()
    assert data0["test_gain_db"] == 0.0

    resp1 = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db=12.0))
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert data1["test_gain_db"] == 12.0
    assert data1["input_peak_dbfs"] > data0["input_peak_dbfs"]


def test_render_and_generation_retain_identical_source_bytes(client, tmp_path, isolated_training_paths, monkeypatch):
    training_path, _ = isolated_training_paths
    _write_training_wav(training_path)
    monkeypatch.setattr(app_module, "WORK_DIR", tmp_path / "work")
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    original = amp_a.read_bytes()
    observed_paths = []

    def changing_original(model, audio, sample_rate):
        # Simulate replacement during native inference. The retained model
        # is already separate, and later generation must consume those bytes.
        amp_a.write_text("replaced during inference")
        assert model.path.read_bytes() == original
        observed_paths.append(model.path)
        return np.asarray(audio, dtype=np.float32).copy()

    monkeypatch.setattr(pipeline, "render", changing_original)
    monkeypatch.setattr(training_target, "render", changing_original)
    rendered = client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    assert rendered.status_code == 200
    identity = rendered.get_json()
    expected_hash = hashlib.sha256(original).hexdigest()
    assert identity["source_hashes"]["amp_a"] == expected_hash
    generated = client.post("/api/generate", json={
        "render_id": identity["render_id"], "model_name": "Frozen source regression",
    })
    assert generated.status_code == 200
    manifest = jsonlib.loads(Path(generated.get_json()["manifest_path"]).read_text())
    provenance = manifest["preview_render_provenance"]
    assert provenance["source_hashes"] == identity["source_hashes"]
    assert Path(provenance["source_paths"]["amp_a"]).read_bytes() == original
    assert len(observed_paths) == 4
    assert observed_paths[:2] == observed_paths[2:]


def test_render_pair_rejects_non_numeric_test_gain_db(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, test_gain_db="loud"))
    assert resp.status_code == 400
    assert "test_gain_db" in resp.get_json()["error"]


def test_rerendering_same_profile_repeatedly_does_not_stack_gain(client, tmp_path):
    """Calling /api/render_pair twice with the SAME profile must reproduce
    the exact same input_peak_dbfs each time -- proves the profile gain is
    always applied fresh from the original source DI, never compounded onto
    a previous render's already-gained signal."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    body = _render_body(amp_a, amp_b, input_profile_id="extreme_passive")

    peaks = [client.post("/api/render_pair", json=body).get_json()["input_peak_dbfs"] for _ in range(3)]
    assert peaks[0] == pytest.approx(peaks[1]) == pytest.approx(peaks[2])


def test_blend_stage_routes_never_require_a_fresh_render(client, tmp_path):
    """Once rendered, crossover/transition/trim changes must be servable
    purely from the cached RenderedPair -- this is the entire point of
    splitting render_pair()/build_hybrid() apart. Sweeping crossover here
    must succeed and change the reported trim without ever touching
    /api/render_pair again."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    r1 = client.post("/api/blend_info", json={"render_id": _current_render_id(), "crossover_dbfs": -35.0, "transition_width_db": 8.0})
    r2 = client.post("/api/blend_info", json={"render_id": _current_render_id(), "crossover_dbfs": -5.0, "transition_width_db": 8.0})
    assert r1.status_code == 200
    assert r2.status_code == 200

    # amp_a/amp_b are identical fakes here, so auto_trim_db is always 0 --
    # the real proof that crossover moves the blend without a rerender is
    # the blend WEIGHT curve itself, which depends only on the envelope and
    # crossover config, not on which amps were rendered.
    curve1 = client.post("/api/blend_curve", json={"render_id": _current_render_id(), "crossover_dbfs": -35.0, "transition_width_db": 8.0, "max_points": 200})
    curve2 = client.post("/api/blend_curve", json={"render_id": _current_render_id(), "crossover_dbfs": -5.0, "transition_width_db": 8.0, "max_points": 200})
    assert curve1.status_code == 200 and curve2.status_code == 200
    assert curve1.get_json()["blend_weight"] != curve2.get_json()["blend_weight"]


def test_input_profile_gain_reaches_the_actual_rendered_audio(client, tmp_path):
    """End-to-end check through the real Flask route (not just the pipeline
    unit test): a hotter profile must make what /api/preview actually
    returns for source=a louder, proving the gain reaches the served audio,
    not just the diagnostics JSON."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    quiet_render = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="vintage_single")).get_json()
    quiet_wav = client.post("/api/preview", json={"source": "a", "render_id": quiet_render["render_id"]}).data

    loud_render = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="extreme_passive")).get_json()
    loud_wav = client.post("/api/preview", json={"source": "a", "render_id": loud_render["render_id"]}).data

    # Both are valid WAVs of the same nominal duration; the hot one must
    # simply contain larger sample magnitudes since render() is an identity
    # pass-through here.
    assert len(quiet_wav) == len(loud_wav)
    assert quiet_wav != loud_wav


def test_calibration_warning_surfaces_through_render_pair_response(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a, input_level_dbu=8.0)
    _write_fake_nam(amp_b)  # uncalibrated

    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, calibration_mode="auto"))
    data = resp.get_json()
    assert data["calibration_applied"] is False
    assert any("input_level_dbu" in w for w in data["warnings"])


def test_active_profile_without_custom_gain_is_rejected(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, input_profile_id="active_buffered"))
    assert resp.status_code == 400


@pytest.mark.parametrize("field", ["reference_input_level_dbu", "test_gain_db", "amp_a_input_gain_db", "amp_b_input_gain_db"])
def test_render_pair_rejects_non_numeric_gain_controls(client, tmp_path, field):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)

    resp = client.post("/api/render_pair", json=_render_body(amp_a, amp_b, **{field: "not-a-number"}))

    assert resp.status_code == 400
    assert resp.is_json


def test_preview_rejects_invalid_output_gain(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    resp = client.post("/api/preview", json={
        "source": "hybrid", "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "manual_output_gain_db": "not-a-number",
    })

    assert resp.status_code == 400
    assert resp.is_json


@pytest.fixture
def isolated_training_paths(tmp_path, monkeypatch):
    """Redirect app.py's training-input/bundle paths into tmp_path so these
    tests never touch the real work/ directory."""
    training_path = tmp_path / "training_input" / "input.wav"
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    monkeypatch.setattr(app_module, "TRAINING_INPUT_PATH", training_path)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    return training_path, a2_dir


def _write_training_wav(path, n=4800, sample_rate=48000):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.uniform(-1, 1, n)).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path


def _comparison_bundle(tmp_path, monkeypatch):
    a2_dir, di_dir = tmp_path / "a2", tmp_path / "di"
    bundle = a2_dir / "design-1"
    bundle.mkdir(parents=True)
    di_dir.mkdir()
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setattr(app_module, "DI_DIR", di_dir)
    _write_training_wav(di_dir / "held-out.wav", n=1000)
    amp_a, amp_b, model = bundle / "a.nam", bundle / "b.nam", bundle / "model.nam"
    for path in (amp_a, amp_b, model):
        _write_fake_nam(path)
    from hybrid.modes.design import HybridDesign
    HybridDesign(
        str(amp_a), str(amp_b), crossover_dbfs=-20.0, calibration_mode="raw",
    ).write_json(bundle / "hybrid_design.json")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "mode": "hybrid",
        "amp_a": {"path": str(amp_a), "sha256": sha(amp_a)},
        "amp_b": {"path": str(amp_b), "sha256": sha(amp_b)},
        "cab": {"selected": False},
        "output_gain": {"applied_gain_db": 0.0},
        "target": {"global_safety_gain_reduction_db": 0.0},
        "training": {"output_nam_path": str(model), "output_nam_sha256": sha(model)},
    }
    (bundle / "training_manifest.json").write_text(jsonlib.dumps(manifest))
    return bundle, amp_a, amp_b, model


def test_comparison_builds_synchronised_teacher_full_lite_stems(client, tmp_path, monkeypatch):
    _bundle, _a, _b, model = _comparison_bundle(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(app_module, "render_processed_reference", lambda design, manifest, dry, sr: SimpleNamespace(hybrid=dry * 2.0))
    def fake_model(path, dry, sr, slim=None):
        seen.append((Path(path), slim, float(dry[0])))
        return dry * (2.0 if slim == SLIM_FULL else 1.5)
    monkeypatch.setattr(app_module, "render_trained_a2", fake_model)

    response = client.post("/api/comparison", json={
        "design_id": "design-1", "di_file": "held-out.wav", "input_gain_db": -6,
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["actual_output_levels"] is True
    assert [variant["id"] for variant in data["variants"]] == ["teacher", "full", "lite"]
    assert data["variants"][1]["metrics"]["raw_esr"] == pytest.approx(0.0)
    assert [entry[1] for entry in seen] == [SLIM_FULL, SLIM_LITE]
    assert all(entry[0] == model for entry in seen)
    audio_response = client.get(data["audio_url"])
    audio, sample_rate = sf.read(io.BytesIO(audio_response.data), dtype="float32", always_2d=True)
    assert sample_rate == 48000
    assert audio.shape == (1000, 3)
    np.testing.assert_allclose(audio[:, 0], audio[:, 1], atol=1e-6)


def test_comparison_missing_teacher_source_keeps_model_available(client, tmp_path, monkeypatch):
    _bundle, amp_a, _b, _model = _comparison_bundle(tmp_path, monkeypatch)
    amp_a.unlink()
    called = False
    def should_not_render(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("inference must not start")
    monkeypatch.setattr(app_module, "render_trained_a2", should_not_render)

    response = client.post("/api/comparison", json={"design_id": "design-1", "di_file": "held-out.wav"})

    assert response.status_code == 409
    assert response.get_json()["code"] == "teacher_sources_unavailable"
    assert response.get_json()["model_available"] is True
    assert called is False


def test_imported_model_without_original_bundle_explains_teacher_is_unavailable(client, tmp_path, monkeypatch):
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    model = a2_dir / "embedded-copy.nam"
    _write_fake_nam(model)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)

    response = client.post("/api/comparison", json={
        "design_id": "missing-design", "model_path": str(model), "di_file": "anything.wav",
    })

    assert response.status_code == 409
    data = response.get_json()
    assert data["code"] == "teacher_sources_unavailable"
    assert data["model_available"] is True
    assert "embedded NAM remains usable" in data["error"]


def test_comparison_cache_identity_changes_with_gain_and_lite_can_be_unavailable(client, tmp_path, monkeypatch):
    _comparison_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "render_processed_reference", lambda design, manifest, dry, sr: SimpleNamespace(hybrid=dry))
    def fake_model(path, dry, sr, slim=None):
        if slim == SLIM_LITE:
            raise RuntimeError("export has no Lite branch")
        return dry
    monkeypatch.setattr(app_module, "render_trained_a2", fake_model)
    body = {"design_id": "design-1", "di_file": "held-out.wav", "input_gain_db": 0}

    first = client.post("/api/comparison", json=body).get_json()
    second = client.post("/api/comparison", json=body).get_json()
    quieter = client.post("/api/comparison", json={**body, "input_gain_db": -24}).get_json()

    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert first["comparison_id"] == second["comparison_id"]
    assert quieter["identity"] != first["identity"]
    assert first["variants"][2]["state"] == "unavailable"
    assert first["variants"][1]["metrics"]["raw_esr"] == pytest.approx(0.0)


def test_training_input_status_missing_by_default(client, isolated_training_paths):
    resp = client.get("/api/training_input/status")
    assert resp.status_code == 200
    assert resp.get_json()["ready"] is False


def test_training_input_upload_accepts_valid_wav(client, isolated_training_paths, tmp_path):
    src = _write_training_wav(tmp_path / "src.wav")
    with open(src, "rb") as f:
        resp = client.post("/api/training_input/upload", data={"file": (f, "input.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ready"] is True

    status = client.get("/api/training_input/status").get_json()
    assert status["ready"] is True


def test_training_input_upload_rejects_wrong_sample_rate(client, isolated_training_paths, tmp_path):
    src = _write_training_wav(tmp_path / "src.wav", sample_rate=44100)
    with open(src, "rb") as f:
        resp = client.post("/api/training_input/upload", data={"file": (f, "input.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert client.get("/api/training_input/status").get_json()["ready"] is False


def test_generate_requires_rendered_pair(client, isolated_training_paths):
    app_module._rendered_pair_cache["pair"] = None
    app_module._rendered_pair_cache["snapshot"] = None
    resp = client.post("/api/generate", json={})
    assert resp.status_code == 400


def test_generate_requires_training_input(client, isolated_training_paths, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={"render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0})
    assert resp.status_code == 400
    assert resp.get_json()["training_input_ready"] is False


def test_generate_end_to_end_produces_bundle(client, isolated_training_paths, tmp_path):
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)

    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={
        "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "auto_level": False, "manual_b_trim_db": 1.5,
        "model_name": "My Hybrid Rig",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["implemented"] is True
    assert Path(data["target_path"]).is_file()
    assert Path(data["manifest_path"]).is_file()

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    # The design was auditioned at input_profile_id=vintage_humbucker (0 dB
    # gain, per _render_body's default) -- but this proves the field exists
    # and generation never re-applies ANY profile gain to the training input.
    assert manifest["design"]["pickup_profile_applied_to_training_input"] is False
    assert manifest["design"]["frozen_effective_b_trim_db"] == pytest.approx(1.5)
    assert manifest["mode"] == "hybrid"  # omitted mode defaults to Hybrid
    assert manifest["model_name"] == "My Hybrid Rig"
    assert manifest["artifact_filename"] == "My_Hybrid_Rig.nam"
    assert data["download_filename"] == "My_Hybrid_Rig.nam"


def test_mix_info_defaults_to_hybrid_mode(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    resp = client.post("/api/mix_info", json={"render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0})
    assert resp.status_code == 200
    assert resp.get_json()["mode"] == "hybrid"


def test_mix_info_blend_mode_accepts_mix_b(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))
    resp = client.post("/api/mix_info", json={"render_id": _current_render_id(), "mode": "blend", "mix_b": 0.75})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode"] == "blend"
    assert data["mix_b"] == pytest.approx(0.75)
    assert data["mix_a"] == pytest.approx(0.25)


def test_preview_blend_source_accepts_mix_and_is_independent_of_crossover_params(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/preview", json={"source": "blend", "render_id": _current_render_id(), "mix_b": 0.2, "auto_level": False})
    assert resp.status_code == 200
    assert resp.headers.get("X-Mix-B") is not None
    assert float(resp.headers["X-Mix-B"]) == pytest.approx(0.2)


def test_preview_invalid_mix_b_is_clamped(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/preview", json={"source": "blend", "render_id": _current_render_id(), "mix_b": 5.0, "auto_level": False})
    assert resp.status_code == 200
    assert float(resp.headers["X-Mix-B"]) == pytest.approx(1.0)


def test_cab_upload_validation_rejects_non_wav(client, tmp_path):
    bogus = tmp_path / "not_a_wav.txt"
    bogus.write_text("nope")
    with open(bogus, "rb") as f:
        resp = client.post("/api/cab/upload", data={"file": (f, "cab.txt")}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_cab_upload_validation_rejects_silent_ir(client, tmp_path):
    import soundfile as sf
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
    with open(silent, "rb") as f:
        resp = client.post("/api/cab/upload", data={"file": (f, "cab.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_cab_upload_accepts_valid_ir(client, tmp_path):
    import soundfile as sf
    ir = tmp_path / "ir.wav"
    ir_data = np.zeros(200, dtype=np.float32)
    ir_data[0] = 1.0
    sf.write(ir, ir_data, 48000, subtype="FLOAT")
    with open(ir, "rb") as f:
        resp = client.post("/api/cab/upload", data={"file": (f, "cab.wav")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["original_sample_rate"] == 48000
    assert "path" in data


def test_nam_tool_learned_cab_creates_shared_training_bundle(client, isolated_training_paths, tmp_path, monkeypatch):
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)
    nam_dir = tmp_path / "uploaded_nam"
    nam_dir.mkdir()
    monkeypatch.setattr(app_module, "NAM_UPLOAD_DIR", nam_dir)
    source = nam_dir / "source.nam"
    source.write_text(jsonlib.dumps({
        "architecture": "Test", "config": {}, "sample_rate": 48000,
        "metadata": {"name": "Source Amp", "tone_type": "clean", "input_level_dbu": -12.0},
    }))
    ir_path = tmp_path / "cab.wav"
    sf.write(ir_path, np.array([0.75, 0.25], dtype=np.float32), 48000, subtype="FLOAT")

    response = client.post("/api/nam/tools/cab-embed", json={
        "path": str(source), "cab_path": str(ir_path), "mode": "learned",
        "model_name": "Source Amp Studio", "cab_display_name": "Studio 2x12",
    })

    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    manifest = jsonlib.loads(Path(data["manifest_path"]).read_text())
    assert data["design_id"] == "Source_Amp_Studio"
    assert manifest["mode"] == "cab_embed"
    assert manifest["amp_a"]["filename"] == "source.nam"
    assert manifest["cab"]["export_mode"] == "learned"
    assert manifest["cab"]["display_name"] == "Studio 2x12"
    assert manifest["artifact_filename"] == "Source_Amp_Studio.nam"
    assert (a2_dir / data["design_id"] / "hybrid_target.wav").is_file()


def test_nam_tool_exact_cab_embed_packages_supported_a2(client, tmp_path, monkeypatch):
    nam_dir = tmp_path / "uploaded_nam"
    tool_dir = tmp_path / "nam_tools"
    nam_dir.mkdir()
    tool_dir.mkdir()
    monkeypatch.setattr(app_module, "NAM_UPLOAD_DIR", nam_dir)
    monkeypatch.setattr(app_module, "NAM_TOOL_OUTPUT_DIR", tool_dir)
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "settings.env"))
    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "true")
    source = nam_dir / "source.nam"
    wave = {"version": "0.7.0", "architecture": "WaveNet", "config": {}, "weights": [1], "sample_rate": 48000}
    source.write_text(jsonlib.dumps({
        "version": "0.7.0", "architecture": "SlimmableContainer", "sample_rate": 48000,
        "config": {"submodels": [{"max_value": 1.0, "model": wave}]}, "weights": [],
    }))
    ir_path = tmp_path / "cab.wav"
    sf.write(ir_path, np.array([1.0, -0.25], dtype=np.float32), 48000, subtype="FLOAT")

    response = client.post("/api/nam/tools/cab-embed", json={
        "path": str(source), "cab_path": str(ir_path), "mode": "embedded",
        "model_name": "Source Amp", "cab_display_name": "Exact Cab",
    })

    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    packaged = jsonlib.loads((tool_dir / data["filename"]).read_text())
    assert packaged["architecture"] == "Sequential"
    assert [child["architecture"] for child in packaged["config"]["models"]] == ["WaveNet", "Linear"]
    assert data["prepared_ir_tap_count"] == 2


def test_preview_with_cab_applies_same_ir_to_a_result_and_b(client, tmp_path):
    import soundfile as sf
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    ir_path = tmp_path / "ir.wav"
    ir_data = np.zeros(10, dtype=np.float32)
    ir_data[0] = 0.5
    ir_data[1] = 0.5
    sf.write(ir_path, ir_data, 48000, subtype="FLOAT")

    for source in ("a", "b", "hybrid"):
        resp = client.post("/api/preview", json={
                "source": source, "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
            "cab_path": str(ir_path), "cab_preview_enabled": True,
        })
        assert resp.status_code == 200, (source, resp.get_json() if resp.data else resp.status_code)


def test_generate_blend_mode_produces_bundle_with_mode_blend(client, isolated_training_paths, tmp_path):
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)

    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    resp = client.post("/api/generate", json={"render_id": _current_render_id(), "mode": "blend", "mix_b": 0.4, "auto_level": False, "manual_b_trim_db": 0.5})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode"] == "blend"
    assert Path(data["target_path"]).is_file()

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    assert manifest["mode"] == "blend"
    assert manifest["design"]["mix_b"] == pytest.approx(0.4)
    assert manifest["artifact_filename"] == "a_and_b_Parallel_Blend.nam"
    assert data["download_filename"] == "a_and_b_Parallel_Blend.nam"


def test_generate_baked_cab_records_provenance(client, isolated_training_paths, tmp_path):
    import soundfile as sf
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)

    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    ir_path = tmp_path / "ir.wav"
    ir_data = np.zeros(10, dtype=np.float32)
    ir_data[0] = 0.6
    ir_data[1] = 0.4
    sf.write(ir_path, ir_data, 48000, subtype="FLOAT")

    resp = client.post("/api/generate", json={
        "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "cab_path": str(ir_path), "cab_preview_enabled": True, "cab_export_mode": "learned",
        "cab_display_name": "Modern Boutique 4x12", "model_name": "British American High Gain",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["cab_summary"]["baked"] is True
    assert data["cab_summary"]["sha256"]
    assert data["cab_summary"]["display_name"] == "Modern Boutique 4x12"

    with open(data["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    assert manifest["cab"]["baked"] is True
    assert manifest["cab"]["selected"] is True
    assert manifest["cab"]["display_name"] == "Modern Boutique 4x12"

    from hybrid.training.a2_training_settings import user_metadata_kwargs
    assert user_metadata_kwargs(manifest)["name"] == "British American High Gain + Modern Boutique 4x12 [Learned Cab]"


def test_generate_embedded_cab_keeps_the_training_target_head_only(
    client, isolated_training_paths, tmp_path, monkeypatch
):
    """An exact cabinet derivative never changes what the A2 trains/tests."""
    import soundfile as sf
    training_path, a2_dir = isolated_training_paths
    _write_training_wav(training_path)
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    client.post("/api/render_pair", json=_render_body(amp_a, amp_b))

    ir_path = tmp_path / "ir.wav"
    ir_data = np.zeros(10, dtype=np.float32)
    ir_data[0] = 0.6
    sf.write(ir_path, ir_data, 48000, subtype="FLOAT")

    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "true")

    resp = client.post("/api/generate", json={
        "render_id": _current_render_id(), "crossover_dbfs": -20.0, "transition_width_db": 8.0,
        "cab_path": str(ir_path), "cab_preview_enabled": True, "cab_export_mode": "embedded",
    })
    assert resp.status_code == 200
    with open(resp.get_json()["manifest_path"]) as f:
        manifest = jsonlib.load(f)
    assert manifest["cab"]["export_mode"] == "embedded"
    assert manifest["cab"]["baked"] is False
    assert manifest["output_gain"]["embedded_final"]["final_linear_scalar"] > 0


def test_generate_embedded_cab_requires_advanced_setting(client, tmp_path, monkeypatch):
    ir_path = tmp_path / "ir.wav"
    sf.write(ir_path, np.array([0.6, 0.0], dtype=np.float32), 48000, subtype="FLOAT")
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "unused.env"))
    monkeypatch.delenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", raising=False)

    with pytest.raises(ValueError, match="experimental NAM architecture"):
        app_module._resolve_cab_design({
            "cab_path": str(ir_path),
            "cab_export_mode": "embedded",
        }, 48000)


def test_settings_get_and_save_round_trip(client, tmp_path, monkeypatch):
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.delenv("NAM_RENDER_EXE", raising=False)

    resp = client.get("/api/settings")
    assert resp.status_code == 200
    names = {field["name"] for field in resp.get_json()["settings"]}
    assert "NAM_RENDER_EXE" in names

    resp = client.post("/api/settings", json={"values": {"NAM_RENDER_EXE": "/opt/nam_render"}})
    assert resp.status_code == 200
    assert resp.get_json()["saved"] == ["NAM_RENDER_EXE"]

    resp = client.get("/api/settings")
    saved = {field["name"]: field["value"] for field in resp.get_json()["settings"]}
    assert saved["NAM_RENDER_EXE"] == "/opt/nam_render"

    # save_settings() sets os.environ directly (by design, for immediate
    # in-process effect). Clean up with a plain os.environ.pop, not
    # monkeypatch.delenv: monkeypatch would instead restore this value at
    # teardown (it saves "current value" to undo its own delenv), leaking it
    # into later tests.
    import os as _os
    _os.environ.pop("NAM_RENDER_EXE", None)


def test_settings_api_rejects_tone3000_publishable_key(client, tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(env_path))

    resp = client.post("/api/settings", json={"values": {"TONE3000_API_KEY": "t3k_pub_wrong-kind"}})

    assert resp.status_code == 400
    assert "t3k_cs_" in resp.get_json()["error"]
    assert not env_path.exists()


def test_local_llm_pull_route_reports_backend_error_as_json(client, monkeypatch):
    from hybrid.services.ollama_pull import OllamaPullError

    def fake_start_pull(model=None):
        raise OllamaPullError("ollama not found")

    monkeypatch.setattr(app_module, "start_ollama_pull", fake_start_pull)
    resp = client.post("/api/local_llm/pull")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is False
    assert "ollama not found" in data["error"]


def test_local_llm_models_route_returns_provider_models(client, monkeypatch):
    monkeypatch.setattr(app_module, "available_ai_models", lambda: {
        "ok": True, "provider": "local", "models": ["gemma4:e4b", "qwen3:8b"],
    })

    response = client.get("/api/local_llm/models")

    assert response.status_code == 200
    assert response.get_json()["models"] == ["gemma4:e4b", "qwen3:8b"]


def test_local_llm_pull_status_route_returns_current_state(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_ollama_pull_status", lambda: {"status": "idle", "model": None})
    resp = client.get("/api/local_llm/pull_status")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "idle"


def test_renderer_download_route_reports_backend_error_as_json(client, monkeypatch):
    from hybrid.core.render_bootstrap import NamRenderDownloadError

    def fake_download():
        raise NamRenderDownloadError("no prebuilt binary for this platform")

    monkeypatch.setattr(app_module, "download_prebuilt_nam_render", fake_download)
    resp = client.post("/api/renderer/download")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is False
    assert "no prebuilt binary" in data["error"]


def test_character_low_level_check_applies_per_amp_input_gain(client, tmp_path, monkeypatch):
    """The preview sweep must feed each amp the same input the bundle gate does
    (calibration + that amp's input trim), or preview and generation disagree."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    body = _render_body(amp_a, amp_b, calibration_mode="raw", amp_b_input_gain_db=-12.0)
    assert client.post("/api/render_pair", json=body).status_code == 200

    peaks = {"a.nam": [], "b.nam": []}

    def recording_render(model, audio, sample_rate):
        peaks[Path(model.path).name].append(float(np.max(np.abs(audio))))
        return np.asarray(audio, dtype=np.float32).copy()

    monkeypatch.setattr(app_module, "render", recording_render)
    resp = client.post("/api/character/low_level_check", json={"render_id": _current_render_id()})
    assert resp.status_code == 200
    assert peaks["a.nam"] and len(peaks["a.nam"]) == len(peaks["b.nam"])
    for peak_a, peak_b in zip(peaks["a.nam"], peaks["b.nam"]):
        assert peak_b == pytest.approx(peak_a * 10 ** (-12.0 / 20.0), rel=1e-4)


def test_render_sources_sweep_keeps_the_live_renders_sources(tmp_path, monkeypatch):
    """A render auditioned for longer than the grace period is still in use:
    deleting any session must not sweep the amp copies it will generate from."""
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", tmp_path / "a2")
    monkeypatch.setattr(app_module, "WORK_DIR", tmp_path)
    rs = tmp_path / "render_sources"; rs.mkdir()
    amp_a = rs / "aaa111" / "amp-a.nam"; amp_a.parent.mkdir(); amp_a.write_text("{}")
    amp_b = rs / "bbb222" / "amp-b.nam"; amp_b.parent.mkdir(); amp_b.write_text("{}")
    for f in (amp_a, amp_b):
        _age(f.parent, 4000); _age(f, 4000)
    monkeypatch.setitem(app_module._rendered_pair_cache, "snapshot",
                        {"amp_a_path": str(amp_a), "amp_b_path": str(amp_b), "source_paths": {}})
    assert app_module._sweep_orphaned_render_sources() == []
    assert amp_a.exists() and amp_b.exists()


def test_profile_coverage_rejects_a_non_numeric_custom_gain(client, tmp_path):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200
    resp = client.post("/api/profile_coverage", json={"render_id": _current_render_id(), "crossover_dbfs": -20,
                                                      "transition_width_db": 6, "custom_input_gain_db": "abc"})
    assert resp.status_code == 400 and "custom_input_gain_db" in resp.get_json()["error"]


def test_character_low_level_check_reports_renderer_failure_as_json(client, tmp_path, monkeypatch):
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200

    def failing_render(*_a, **_k):
        raise app_module.NamRenderError("model failed to load")

    monkeypatch.setattr(app_module, "render", failing_render)
    resp = client.post("/api/character/low_level_check", json={"render_id": _current_render_id()})
    assert resp.status_code == 500 and "model failed to load" in resp.get_json()["error"]


def test_character_analyses_are_computed_once_per_render(client, tmp_path, monkeypatch):
    """The wizard, the low-level check and Character previews for one render
    share one analysis per amp: no re-hashing or re-analysis per request."""
    amp_a, amp_b = tmp_path / "a.nam", tmp_path / "b.nam"
    _write_fake_nam(amp_a)
    _write_fake_nam(amp_b)
    monkeypatch.setattr(app_module, "WORK_DIR", tmp_path / "work")
    assert client.post("/api/render_pair", json=_render_body(amp_a, amp_b)).status_code == 200
    monkeypatch.setattr(app_module, "render", lambda model, audio, sr, **k: np.asarray(audio, dtype=np.float32).copy())
    calls = []
    real = app_module.analyse_rendered_audio
    monkeypatch.setattr(app_module, "analyse_rendered_audio", lambda *a, **k: calls.append(1) or real(*a, **k))
    rid = _current_render_id()
    assert client.post("/api/wizard/insight", json={"render_id": rid}).status_code == 200
    first = len(calls)
    assert 1 <= first <= 2   # at most once per amp (identical fake amps share the disk cache entry)
    assert client.post("/api/wizard/insight", json={"render_id": rid}).status_code == 200
    assert client.post("/api/character/low_level_check", json={"render_id": rid}).status_code == 200
    assert len(calls) == first   # later requests for the same render analyse nothing


def test_packaged_app_keeps_the_training_venv_in_the_writable_data_dir(tmp_path):
    """In the packaged app TRAINING_ROOT is inside the installed bundle; the
    1.4 GB venv must go to the per-user data dir. From source it stays the
    README's <repo>/.venv-a2."""
    root, work = tmp_path / "bundle" / "training_support", tmp_path / "data"
    assert app_module._training_venv_dir(root, work, frozen=True) == work / ".venv-a2"
    assert app_module._training_venv_dir(root, work, frozen=False) == root / ".venv-a2"


def test_backend_bundle_spec_includes_the_kaggle_worker_and_excludes_personal_captures():
    spec = (Path(__file__).resolve().parent.parent / "packaging" / "backend" / "nam_mixer_backend.spec").read_text()
    assert '"cloud" / "kaggle" / "train_a2_cloud.py"' in spec and "training_support/cloud/kaggle" in spec
    assert '"assets" / "nam_models"), "assets/nam_models"' not in spec
    assert "raise SystemExit" in spec   # no renderer, no bundle


def test_backend_exit_stops_a_running_local_training(monkeypatch):
    """The desktop shell SIGTERMs the backend on quit; local training runs in its
    own session, so the backend must stop it on the way out."""
    stopped = []

    class Manager:
        def cancel(self):
            stopped.append(True)

    monkeypatch.setattr(app_module, "_local_training_manager", Manager())
    app_module._stop_local_training_on_exit()
    assert stopped == [True]

    class Idle:
        def cancel(self):
            raise RuntimeError("No local setup or training process is running.")

    monkeypatch.setattr(app_module, "_local_training_manager", Idle())
    app_module._stop_local_training_on_exit()   # nothing running: no error
    with pytest.raises(SystemExit):
        app_module._exit_on_sigterm(15, None)



def test_shutdown_endpoint_needs_the_shells_token(client, monkeypatch):
    stopped, exited = [], []
    monkeypatch.setattr(app_module, "_stop_local_training_on_exit", lambda: stopped.append(1))

    class ImmediateTimer:
        def __init__(self, _delay, fn):
            self.fn = fn
        def start(self):
            self.fn()

    monkeypatch.setattr(app_module.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(app_module, "_exit_process", lambda: exited.append(1))

    monkeypatch.delenv("NAM_MIXER_SHUTDOWN_TOKEN", raising=False)
    assert client.post("/api/shutdown").status_code == 404          # source runs: no shutdown endpoint
    monkeypatch.setenv("NAM_MIXER_SHUTDOWN_TOKEN", "s3cret-token")
    assert client.post("/api/shutdown").status_code == 403
    assert client.post("/api/shutdown", headers={"X-NAM-Mixer-Shutdown-Token": "wrong"}).status_code == 403
    assert stopped == [] and exited == []
    resp = client.post("/api/shutdown", headers={"X-NAM-Mixer-Shutdown-Token": "s3cret-token"})
    assert resp.status_code == 200 and stopped == [1] and exited == [1]


@pytest.mark.parametrize("enabled, status", [(False, 409), (True, 200)])
def test_embedded_nam_download_requires_experimental_architectures(client, tmp_path, monkeypatch, enabled, status):
    a2_dir = tmp_path / "a2"; (a2_dir / "d1").mkdir(parents=True)
    monkeypatch.setattr(app_module, "A2_OUTPUT_DIR", a2_dir)
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "test.env"))
    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "true" if enabled else "false")
    sequential = a2_dir / "d1" / "model-with-cab.nam"; sequential.write_text("{}")
    head = a2_dir / "d1" / "model.nam"; head.write_text("{}")
    (a2_dir / "d1" / "training_manifest.json").write_text(jsonlib.dumps({"training": {
        "output_nam_path": str(head),
        "embedded_artifact": {"state": "validated", "artifacts": {"sequential_nam_path": str(sequential)}}}}))
    resp = client.get("/api/local_training/download?design_id=d1&artifact=embedded")
    assert resp.status_code == status
    if not enabled:
        assert "experimental" in resp.get_json()["error"]
