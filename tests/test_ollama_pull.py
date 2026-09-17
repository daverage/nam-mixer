"""Tests for hybrid/ollama_pull.py -- the Settings page's one-click "pull
gemma4:e4b via Ollama" button backend. Subprocess/PATH access is mocked so
these never actually invoke Ollama."""
import time
from unittest.mock import patch

import pytest

from hybrid import ollama_pull


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
