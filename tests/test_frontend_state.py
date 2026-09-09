"""Execute asynchronous UI-state regressions without requiring a browser."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="requires Node.js")
def test_frontend_request_invalidation():
    script = Path(__file__).with_name("frontend_state.test.cjs")
    result = subprocess.run(
        ["node", "--test", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
