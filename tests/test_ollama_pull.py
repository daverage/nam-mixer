"""Tests for hybrid/services/ollama_pull.py -- the Settings page's one-click "pull
gemma4:e4b via Ollama" button backend. Subprocess/PATH access is mocked so
these never actually invoke Ollama."""
import time
from unittest.mock import patch

import pytest

from hybrid.services import ollama_pull


@pytest.fixture(autouse=True)
def reset_state():
    with ollama_pull._lock:
        ollama_pull._state.update({"status": "idle", "model": None, "error": None, "log_tail": ""})
    yield
    with ollama_pull._lock:
        ollama_pull._state.update({"status": "idle", "model": None, "error": None, "log_tail": ""})


def test_start_pull_raises_clear_error_when_ollama_missing():
    with patch.object(ollama_pull.shutil, "which", return_value=None):
        with pytest.raises(ollama_pull.OllamaPullError, match="not found on PATH"):
            ollama_pull.start_pull()


def test_start_pull_uses_recommended_model_by_default():
    class FakeProcess:
        stdout = iter(["pulling manifest\n", "success\n"])
        def wait(self):
            return 0

    with patch.object(ollama_pull.shutil, "which", return_value="/usr/bin/ollama"), \
         patch.object(ollama_pull.subprocess, "Popen", return_value=FakeProcess()) as popen:
        state = ollama_pull.start_pull()
        assert state["model"] == ollama_pull.RECOMMENDED_LOCAL_MODEL
        for _ in range(50):
            if ollama_pull.get_pull_status()["status"] != "running":
                break
            time.sleep(0.02)
        assert ollama_pull.get_pull_status()["status"] == "done"
        popen.assert_called_once_with(
            ["ollama", "pull", ollama_pull.RECOMMENDED_LOCAL_MODEL],
            stdout=ollama_pull.subprocess.PIPE, stderr=ollama_pull.subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )


def test_start_pull_reports_nonzero_exit_as_error():
    class FakeProcess:
        stdout = iter(["error: no such model\n"])
        def wait(self):
            return 1

    with patch.object(ollama_pull.shutil, "which", return_value="/usr/bin/ollama"), \
         patch.object(ollama_pull.subprocess, "Popen", return_value=FakeProcess()):
        ollama_pull.start_pull("bogus-model")
        for _ in range(50):
            if ollama_pull.get_pull_status()["status"] != "running":
                break
            time.sleep(0.02)
        state = ollama_pull.get_pull_status()
        assert state["status"] == "error"
        assert "exited with code 1" in state["error"]


def test_start_pull_returns_existing_state_when_already_running():
    with ollama_pull._lock:
        ollama_pull._state.update({"status": "running", "model": "gemma4:e4b", "error": None, "log_tail": ""})
    with patch.object(ollama_pull.shutil, "which", return_value="/usr/bin/ollama"), \
         patch.object(ollama_pull.subprocess, "Popen") as popen:
        state = ollama_pull.start_pull("other-model")
        popen.assert_not_called()
        assert state["status"] == "running"
        assert state["model"] == "gemma4:e4b"


def test_unexpected_error_in_the_pull_thread_is_reported_and_allows_a_retry():
    """Any failure while reading output (e.g. a decode error) must end in
    status 'error', not leave the pull stuck on 'running' forever."""
    def broken_output():
        yield "pulling manifest\n"
        raise UnicodeDecodeError("cp1252", b"\x81", 0, 1, "character maps to <undefined>")

    class BrokenProcess:
        stdout = broken_output()
        def wait(self):
            return 0

    class FakeProcess:
        stdout = iter(["success\n"])
        def wait(self):
            return 0

    with patch.object(ollama_pull.shutil, "which", return_value="/usr/bin/ollama"), \
         patch.object(ollama_pull.subprocess, "Popen", side_effect=[BrokenProcess(), FakeProcess()]) as popen:
        ollama_pull.start_pull("gemma4:e4b")
        for _ in range(50):
            if ollama_pull.get_pull_status()["status"] != "running":
                break
            time.sleep(0.02)
        assert ollama_pull.get_pull_status()["status"] == "error"

        ollama_pull.start_pull("gemma4:e4b")  # a second attempt actually starts
        for _ in range(50):
            if ollama_pull.get_pull_status()["status"] != "running":
                break
            time.sleep(0.02)
        assert popen.call_count == 2
        assert ollama_pull.get_pull_status()["status"] == "done"


def test_recommended_model_is_the_one_the_settings_page_shows():
    from hybrid.services import local_llm

    assert ollama_pull.RECOMMENDED_LOCAL_MODEL is local_llm.RECOMMENDED_LOCAL_MODEL

