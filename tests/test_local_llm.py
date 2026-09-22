from __future__ import annotations

import json
import socket

import pytest

from hybrid import local_llm


@pytest.fixture(autouse=True)
def isolated_env_file(tmp_path, monkeypatch):
    """These tests exercise NAM_MIXER_AI_*/NAM_MIXER_LOCAL_LLM_* purely via monkeypatched os.environ, so a real .env on the
    developer's machine (hybrid/env_file.py's fallback, used deliberately in production) must never leak in -- point it at a
    file that doesn't exist. Tests that want the .env fallback itself set NAM_MIXER_ENV_FILE to a real tmp_path file explicitly."""
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "unused.env"))
    for name in local_llm.PROVIDER_AI_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


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
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "gemma4:e4b")
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    class _Ok(_Response):
        status = 200

    monkeypatch.setattr(local_llm, "urlopen", lambda *a, **k: _Ok({}))
    status = local_llm.status()
    assert status["enabled"] is True
    assert status["reachable"] is True


def test_status_reports_unreachable_when_host_does_not_respond(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "gemma4:e4b")
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(local_llm, "urlopen", _raise)
    status = local_llm.status()
    assert status["enabled"] is True
    assert status["reachable"] is False


def test_provider_scoped_config_does_not_mix_hosts(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_LOCAL_MODEL", "gemma4:e4b")
    monkeypatch.setenv("NAM_MIXER_AI_LOCAL_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("NAM_MIXER_AI_CUSTOM_MODEL", "remote-chat")
    monkeypatch.setenv("NAM_MIXER_AI_CUSTOM_BASE_URL", "https://models.example.com/v1")
    monkeypatch.setenv("NAM_MIXER_AI_CUSTOM_API_KEY", "remote-secret")
    monkeypatch.setattr(local_llm.socket, "getaddrinfo", lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])

    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "local")
    assert local_llm._config() == local_llm.AiConfig(
        "local", "http://localhost:11434/v1", "gemma4:e4b", None,
    )
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "custom")
    assert local_llm._config() == local_llm.AiConfig(
        "custom", "https://models.example.com/v1", "remote-chat", "remote-secret",
    )


def test_available_models_uses_openai_compatible_models_endpoint(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "local")
    monkeypatch.setenv("NAM_MIXER_AI_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1")
    seen = {}

    def fake_open(request, timeout):
        seen.update(url=request.full_url, timeout=timeout)
        return _Response({"data": [{"id": "qwen3:8b"}, {"id": "gemma4:e4b"}]})

    result = local_llm.available_models(opener=fake_open)

    assert result == {"ok": True, "provider": "local", "models": ["gemma4:e4b", "qwen3:8b"]}
    assert seen == {"url": "http://127.0.0.1:11434/v1/models", "timeout": 10}


def test_available_models_uses_cloudflare_model_search(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "cloudflare")
    monkeypatch.setenv("NAM_MIXER_AI_CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("NAM_MIXER_AI_CLOUDFLARE_API_KEY", "cloudflare-secret")
    seen = {}

    def fake_open(request, timeout):
        seen.update(url=request.full_url, authorization=request.headers.get("Authorization"))
        return _Response({"result": [{"name": "@cf/meta/llama-3.3-70b-instruct-fp8-fast"}]})

    result = local_llm.available_models(opener=fake_open)

    assert result["models"] == ["@cf/meta/llama-3.3-70b-instruct-fp8-fast"]
    assert "/ai/models/search?" in seen["url"]
    assert seen["authorization"] == "Bearer cloudflare-secret"


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
    assert "2-18 dB" in seen["system"]
    assert "Changes as you play harder" in seen["system"]
    assert "Parallel Blend" not in seen["system"]


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
    # Computed from the real formula (hybrid.local_llm._default_max_tokens applied to the actual char-limit constants) rather
    # than a hardcoded number: a hardcoded 1400 here silently went stale after the char limits changed and only "passed" by
    # coincidence on machines whose .env happened to pin NAM_MIXER_AI_MAX_TOKENS/NAM_MIXER_LOCAL_LLM_MAX_TOKENS to 1400.
    expected_max_tokens = local_llm._default_max_tokens(
        local_llm.MAX_LOCAL_RECIPE_EXPLANATION_LENGTH, local_llm.MAX_LOCAL_CONVERSATION_REPLY_LENGTH
    )
    assert seen["body"]["max_tokens"] == expected_max_tokens
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


def test_cloudflare_constructs_fixed_url_and_sends_bearer_token(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "cloudflare")
    monkeypatch.setenv("NAM_MIXER_AI_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("NAM_MIXER_AI_API_KEY", "secret-token")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "@cf/openai/gpt-oss-20b")
    seen = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        return _Response({"choices": [{"message": {"content": '{"reply":"ok","recipe":null}'}}]})

    assert local_llm.converse("hello", opener=fake_open).reply == "ok"
    assert seen == {
        "url": "https://api.cloudflare.com/client/v4/accounts/" + "a" * 32 + "/ai/v1/chat/completions",
        "authorization": "Bearer secret-token",
    }


def test_conversation_debug_records_exact_messages_without_credentials(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "cloudflare")
    monkeypatch.setenv("NAM_MIXER_AI_ACCOUNT_ID", "b" * 32)
    monkeypatch.setenv("NAM_MIXER_AI_API_KEY", "super-secret-token")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    debug = {}

    def fake_open(_request, timeout):
        return _Response({"choices": [{"message": {"content": '{"reply":"Use the researched source.","recipe":null}'}}]})

    local_llm.converse(
        "Find a clean source.",
        history=[{"role": "assistant", "content": "Which era?"}],
        research_notes="Web research:\n- Example evidence",
        debug_trace=debug,
        opener=fake_open,
    )

    assert debug["provider"] == "cloudflare"
    assert debug["model"] == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert debug["messages"][-2] == {"role": "assistant", "content": "Which era?"}
    assert "<research_notes>" in debug["messages"][-1]["content"]
    assert debug["parsed_response"]["reply"] == "Use the researched source."
    assert "super-secret-token" not in __import__("json").dumps(debug)


def test_cloudflare_and_local_receive_the_same_conversation_payload(monkeypatch):
    bodies = {}

    def fake_open(request, timeout):
        bodies[request.full_url.split("/")[2]] = json.loads(request.data.decode())
        return _Response({"choices": [{"message": {"content": '{"reply":"ok","recipe":null}'}}]})

    common = {
        "prompt": "Keep the Vox EQ, then move to Marshall crunch as I play harder.",
        "history": [{"role": "assistant", "content": "Use a broad hybrid."}],
        "research_notes": "TONE3000: Vox AC30 and Marshall JVM captures.",
    }
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "local")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "gemma4:e4b")
    local_llm.converse(opener=fake_open, **common)

    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "cloudflare")
    monkeypatch.setenv("NAM_MIXER_AI_ACCOUNT_ID", "d" * 32)
    monkeypatch.setenv("NAM_MIXER_AI_API_KEY", "token")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    local_llm.converse(opener=fake_open, **common)

    local_body = bodies["127.0.0.1:11434"]
    cloudflare_body = bodies["api.cloudflare.com"]
    assert {key: local_body[key] for key in ("messages", "temperature", "max_tokens")} == {
        key: cloudflare_body[key] for key in ("messages", "temperature", "max_tokens")
    }
    assert local_body["response_format"]["type"] == "json_object"
    assert cloudflare_body["response_format"]["type"] == "json_schema"


def test_local_provider_never_sends_authorization(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "local")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "test")
    monkeypatch.setenv("NAM_MIXER_AI_API_KEY", "should-not-be-used")
    seen = {}

    def fake_open(request, timeout):
        seen["authorization"] = request.get_header("Authorization")
        return _Response({"choices": [{"message": {"content": '{"reply":"ok","recipe":null}'}}]})

    local_llm.converse("hello", opener=fake_open)
    assert seen["authorization"] is None


def test_provider_json_mode_accepts_fenced_or_typed_content(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "cloudflare")
    monkeypatch.setenv("NAM_MIXER_AI_ACCOUNT_ID", "b" * 32)
    monkeypatch.setenv("NAM_MIXER_AI_API_KEY", "token")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "@cf/openai/gpt-oss-20b")

    def fake_open(request, timeout):
        fenced = "```json\n" + json.dumps({"reply": "ok", "recipe": None}) + "\n```"
        return _Response({"choices": [{"message": {"content": [{"type": "text", "text": fenced}]}}]})

    assert local_llm.test_connection(opener=fake_open)["ok"] is True


def test_connection_returns_safe_shape_diagnostics_for_malformed_content(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "cloudflare")
    monkeypatch.setenv("NAM_MIXER_AI_ACCOUNT_ID", "c" * 32)
    monkeypatch.setenv("NAM_MIXER_AI_API_KEY", "token")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "@cf/meta/llama-3-8b-instruct")

    def fake_open(request, timeout):
        return _Response({"choices": [{"message": {"content": "not JSON"}}]})

    result = local_llm.test_connection(opener=fake_open)
    assert result["ok"] is False
    assert result["error"] == "AI provider did not return valid JSON"
    assert result["diagnostics"] == {
        "response_keys": ["choices"],
        "choice_keys": ["message"],
        "message_keys": ["content"],
        "content_type": "str",
        "content_length": 8,
        "exception": "JSONDecodeError",
    }


def test_custom_remote_requires_https_and_public_resolution(monkeypatch):
    monkeypatch.setenv("NAM_MIXER_AI_PROVIDER", "custom")
    monkeypatch.setenv("NAM_MIXER_AI_MODEL", "test")
    monkeypatch.setenv("NAM_MIXER_AI_BASE_URL", "http://example.com/v1")
    assert "HTTPS" in local_llm.status()["error"]

    monkeypatch.setenv("NAM_MIXER_AI_BASE_URL", "https://example.com/v1")
    monkeypatch.setattr(local_llm.socket, "getaddrinfo", lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(local_llm.LocalLlmError, match="disallowed"):
        local_llm._config()


def test_legacy_settings_remain_usable_when_new_names_are_absent(monkeypatch):
    monkeypatch.delenv("NAM_MIXER_AI_MODEL", raising=False)
    monkeypatch.delenv("NAM_MIXER_AI_PROVIDER", raising=False)
    monkeypatch.setenv("NAM_MIXER_LOCAL_LLM_MODEL", "legacy-model")
    assert local_llm._config().model == "legacy-model"
