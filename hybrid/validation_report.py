"""Versioned, backend-neutral exported-model validation reports."""
from __future__ import annotations

from typing import Optional

VALIDATION_REPORT_SCHEMA_VERSION = 1


def build_validation_report(model_sha256: str, variants: dict, *, quiet_playing: Optional[dict] = None, cabinet: Optional[dict] = None) -> dict:
    """Normalize Full/Lite validation into an honest aggregate outcome.

    Metrics are measurements, not a claim of perceptual equivalence.  A
    rendering failure is a failed check; absent/legacy evidence is explicitly
    unavailable, never a green pass.
    """
    checks = []
    for variant in ("full", "lite"):
        result = variants.get(variant)
        if result is None:
            checks.append({"id": f"{variant}_render", "state": "unavailable", "reason": "check was not run"})
        elif result.get("rendered_ok"):
            checks.append({"id": f"{variant}_render", "state": "passed", "metrics": result.get("metrics", {})})
        else:
            checks.append({"id": f"{variant}_render", "state": "failed", "reason": result.get("error", "invalid render")})

    if quiet_playing is None:
        checks.append({"id": "quiet_playing", "state": "unavailable", "reason": "no equivalent validation reference in this bundle"})
    elif quiet_playing.get("pass") is True:
        checks.append({"id": "quiet_playing", "state": "passed", "metrics": quiet_playing})
    elif quiet_playing.get("pass") is False:
        checks.append({"id": "quiet_playing", "state": "failed", "reason": quiet_playing.get("reason") or "quiet-playing response differs from the frozen teacher", "metrics": quiet_playing})
    else:
        checks.append({"id": "quiet_playing", "state": "unavailable", "reason": quiet_playing.get("reason", "check unavailable"), "metrics": quiet_playing})

    states = {check["state"] for check in checks}
    state = "needs_attention" if "failed" in states else "unavailable" if "unavailable" in states else "passed"
    return {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "state": state,
        "checks": checks,
        "cabinet": cabinet or {"baked": False, "approximation": None},
        "summary": (
            "Validation found one or more issues; inspect the failed checks and listen to the export."
            if state == "needs_attention" else
            "Validation is incomplete because one or more checks are unavailable."
            if state == "unavailable" else
            "Required technical checks passed. This does not establish perceptual equivalence."
        ),
    }
