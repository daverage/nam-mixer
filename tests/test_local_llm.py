from __future__ import annotations

import json

import pytest

from hybrid import local_llm


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_local_llm_is_disabled_without_a_model(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "")
    assert local_llm.status()["enabled"] is False


def test_status_reports_reachable_when_host_responds(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "gemma3:4b")
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    class _Ok(_Response):
        status = 200

    monkeypatch.setattr(local_llm, "urlopen", lambda *a, **k: _Ok({}))
    status = local_llm.status()
    assert status["enabled"] is True
    assert status["reachable"] is True


def test_status_reports_unreachable_when_host_does_not_respond(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "gemma3:4b")
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(local_llm, "urlopen", _raise)
    status = local_llm.status()
    assert status["enabled"] is True
    assert status["reachable"] is False


def test_local_llm_accepts_only_a_schema_valid_recipe(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "qwen2.5:3b")
    seen = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _Response({"choices": [{"message": {"content": json.dumps({
            "mode": "character", "tone": 0, "feel": 0, "drive": 0,
            "driveLow": 0, "driveMid": 35, "driveHigh": 100, "explanation": "Vox clean to Marshall crunch.",
        })}}]})

    recipe = local_llm.suggest_recipe("Vox to Marshall", opener=fake_open)
    assert recipe.to_dict()["driveHigh"] == 100
    assert seen == {"url": "http://127.0.0.1:11434/v1/chat/completions", "timeout": 60}


def test_local_llm_rejects_invalid_model_values(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "qwen2.5:3b")

    def fake_open(_request, timeout):
        return _Response({"choices": [{"message": {"content": '{"mode":"blend","mixB":200}'}}]})

    with pytest.raises(local_llm.LocalLlmError, match="invalid mixB"):
        local_llm.suggest_recipe("constant mix", opener=fake_open)


def test_local_llm_conversation_can_ask_a_clarifying_question(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "qwen2.5:3b")
    seen = {}

    def fake_open(request, timeout):
        seen["body"] = json.loads(request.data.decode())
        return _Response({"choices": [{"message": {"content": json.dumps({
            "reply": "Should Amp A or Amp B provide the crunch?", "recipe": None,
        })}}]})

    reply = local_llm.converse(
        "Make it more aggressive",
        [{"role": "assistant", "content": "The current recipe is a balanced blend."}],
        opener=fake_open,
    )

    assert reply.recipe is None
    assert reply.reply == "Should Amp A or Amp B provide the crunch?"
    assert [message["role"] for message in seen["body"]["messages"]] == ["system", "assistant", "user"]


def test_local_llm_teaches_mode_selection_from_signal_behaviour(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "qwen2.5:3b")
    seen = {}

    def fake_open(request, timeout):
        seen["system"] = json.loads(request.data.decode())["messages"][0]["content"]
        return _Response({"choices": [{"message": {"content": json.dumps({
            "reply": "Use Dynamic Hybrid.",
            "recipe": {"mode": "hybrid", "switchKnob": 6, "width": 3, "explanation": "A full voice handoff."},
        })}}]})

    local_llm.converse("Use one whole amp at low guitar volume and another at high volume.", opener=fake_open)

    assert "constant mixture" in seen["system"]
    assert "entire amp to become the other" in seen["system"]
    assert "stable tonal foundation" in seen["system"]


def test_local_llm_compacts_prose_but_preserves_a_structured_source_plan(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_HISTORY_MESSAGES", "6")
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_HISTORY_MESSAGE_CHARS", "900")
    seen = {}

    def fake_open(request, timeout):
        seen["body"] = json.loads(request.data.decode())
        return _Response({"choices": [{"message": {"content": json.dumps({
            "reply": "Keep the assigned sources.", "recipe": None,
            "source_plan": {"ampA": "Fender Deluxe Reverb", "ampB": "Marshall JCM800"},
        })}}]})

    reply = local_llm.converse(
        "Which file should I use?",
        [{"role": "assistant", "content": "x" * 2_000}] * 8,
        research_notes="r" * 9_000,
        opener=fake_open,
    )

    messages = seen["body"]["messages"]
    assert len(messages) == 8  # system, six compact history turns, user
    assert all(len(message["content"]) <= 900 for message in messages[1:-1])
    assert len(messages[-1]["content"]) <= len("Which file should I use?") + 5_200
    assert seen["body"]["max_tokens"] == 1400
    assert reply.source_plan == {"ampA": "Fender Deluxe Reverb", "ampB": "Marshall JCM800"}


def test_local_llm_uses_low_drive_when_a_model_omits_legacy_base_drive():
    recipe = local_llm._recipe_from_json({
        "mode": "character", "tone": 0, "feel": 0,
        "driveLow": 15, "driveMid": 50, "driveHigh": 90,
    })

    assert recipe.drive == 15


def test_local_llm_keeps_a_detailed_but_bounded_explanation(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MAX_EXPLANATION_CHARS", str(local_llm.MAX_LOCAL_RECIPE_EXPLANATION_LENGTH))
    recipe = local_llm._recipe_from_json({
        "mode": "blend", "mixB": 50, "explanation": "x" * 2000,
    })

    assert len(recipe.explanation) == local_llm.MAX_LOCAL_RECIPE_EXPLANATION_LENGTH
