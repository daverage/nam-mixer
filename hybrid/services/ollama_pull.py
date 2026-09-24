"""One-click "pull the recommended local model" for the Settings page.

Any local LLM host that speaks the OpenAI-compatible /v1 API works with the
AI Assistant tab (Ollama, LM Studio, llama.cpp server, ...) -- see
hybrid/services/local_llm.py. Ollama is the one we can actually automate a model
download for, since it has a simple CLI (`ollama pull <model>`); this is not
a claim that Ollama is required, just the easiest path for someone who
doesn't already have a preferred host running.

Pulling a model can take minutes, so this runs the CLI in a background
thread and exposes poll-able state -- mirrors the shape of
hybrid/training/kaggle_training.py's job manager, at a much smaller scale (one global
job, no persistence needed across restarts).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Optional

from .local_llm import RECOMMENDED_LOCAL_MODEL  # one definition, shared with the Settings page

_lock = threading.Lock()
_state: dict = {"status": "idle", "model": None, "error": None, "log_tail": ""}


class OllamaPullError(RuntimeError):
    pass


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def _run(model: str) -> None:
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model],
            # Decode progress output as UTF-8 regardless of the OS locale, and
            # never let an undecodable byte kill the reader thread.
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        tail_lines: list[str] = []
        for line in process.stdout or []:
            tail_lines.append(line.rstrip("\n"))
            tail_lines[:] = tail_lines[-20:]
            with _lock:
                _state["log_tail"] = "\n".join(tail_lines)
        returncode = process.wait()
        with _lock:
            if returncode == 0:
                _state["status"] = "done"
            else:
                _state["status"] = "error"
                _state["error"] = f"ollama pull exited with code {returncode}"
    except Exception as exc:  # noqa: BLE001 -- anything else would leave status stuck at "running" forever
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(exc) or type(exc).__name__


def start_pull(model: Optional[str] = None) -> dict:
    """Start (or report the already-running) pull of `model` (default:
    RECOMMENDED_LOCAL_MODEL) via the `ollama` CLI. Non-blocking."""
    model = model or RECOMMENDED_LOCAL_MODEL
    if not ollama_available():
        raise OllamaPullError(
            "The 'ollama' command was not found on PATH. Install Ollama from "
            "https://ollama.com, or run this model on any other local "
            "OpenAI-compatible host and set its URL/model above directly."
        )
    with _lock:
        if _state["status"] == "running":
            return dict(_state)
        _state.update({"status": "running", "model": model, "error": None, "log_tail": ""})
    thread = threading.Thread(target=_run, args=(model,), daemon=True)
    thread.start()
    return get_pull_status()


def get_pull_status() -> dict:
    with _lock:
        return dict(_state)
