from __future__ import annotations

import sys
import time

from hybrid.local_training import LocalTrainingManager


def test_cancel_stops_an_app_owned_process_and_reports_cancelled(tmp_path):
    manager = LocalTrainingManager(tmp_path, tmp_path / "work" / "a2")
    manager._start([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"], "training")

    manager.cancel()
    deadline = time.monotonic() + 2
    while manager.status()["state"] == "cancelling" and time.monotonic() < deadline:
        time.sleep(0.01)

    status = manager.status()
    assert status["state"] == "cancelled"
    assert status["exit_code"] is not None
    assert "Cancellation requested" in status["log_tail"]
