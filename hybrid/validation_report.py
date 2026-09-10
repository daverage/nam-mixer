"""Versioned, backend-neutral exported-model validation reports."""
from __future__ import annotations

import math
from typing import Optional

VALIDATION_REPORT_SCHEMA_VERSION = 2
VALIDATION_POLICY_VERSION = 1
DEFAULT_VALIDATION_POLICY = {
    "max_raw_esr": 0.25,
    "max_gain_normalized_esr": 0.15,
    "max_absolute_level_error_db": 6.0,
    "max_response_shape_error_db": 3.0,
    "max_extra_quiet_attenuation_db": 6.0,
    "silence_floor_dbfs": -100.0,
}


def _variant_quality_check(variant: str, result: Optional[dict], policy: dict) -> dict:
    check_id = f"{variant}_quality"
    if result is None or not result.get("rendered_ok"):
        return {"id": check_id, "state": "unavailable", "reason": "valid render metrics are unavailable"}
    metrics = result.get("metrics") or {}
    required = ("raw_esr", "gain_normalized_esr")
    if not all(name in metrics for name in required):
        return {"id": check_id, "state": "unavailable", "reason": "raw and gain-normalized ESR were not both recorded", "metrics": metrics}
    if not all(isinstance(metrics[name], (int, float)) and math.isfinite(metrics[name]) for name in required):
        return {"id": check_id, "state": "failed", "reason": "ESR metrics are non-finite", "metrics": metrics}
    failures = []
    if metrics["raw_esr"] > policy["max_raw_esr"]:
        failures.append(f"raw ESR {metrics['raw_esr']:.4g} exceeds {policy['max_raw_esr']:.4g}")
    if metrics["gain_normalized_esr"] > policy["max_gain_normalized_esr"]:
        failures.append(
            f"gain-normalized ESR {metrics['gain_normalized_esr']:.4g} exceeds "
            f"{policy['max_gain_normalized_esr']:.4g}"
        )
    return {
        "id": check_id,
        "state": "failed" if failures else "passed",
        "reason": "; ".join(failures) if failures else "ESR metrics are within the documented policy",
        "metrics": metrics,
    }


def build_validation_report(model_sha256: str, variants: dict, *, quiet_playing: Optional[dict] = None, cabinet: Optional[dict] = None, policy: Optional[dict] = None) -> dict:
    """Normalize Full/Lite validation into an honest aggregate outcome.

    Metrics are measurements, not a claim of perceptual equivalence.  A
    rendering failure is a failed check; absent/legacy evidence is explicitly
    unavailable, never a green pass.
    """
    effective_policy = {**DEFAULT_VALIDATION_POLICY, **(policy or {})}
    checks = []
    for variant in ("full", "lite"):
        result = variants.get(variant)
        if result is None:
            checks.append({"id": f"{variant}_render", "state": "unavailable", "reason": "check was not run"})
        elif result.get("rendered_ok"):
            checks.append({"id": f"{variant}_render", "state": "passed", "metrics": result.get("metrics", {})})
        elif result.get("state") == "unavailable":
            checks.append({"id": f"{variant}_render", "state": "unavailable", "reason": result.get("error", "render check unavailable")})
        else:
            checks.append({"id": f"{variant}_render", "state": "failed", "reason": result.get("error", "invalid render")})
        checks.append(_variant_quality_check(variant, result, effective_policy))

    quiet_variants = quiet_playing if quiet_playing and any(k in quiet_playing for k in ("full", "lite")) else {"full": quiet_playing}
    for variant in ("full", "lite"):
        quiet = quiet_variants.get(variant)
        check_id = f"{variant}_quiet_playing"
        if quiet is None:
            checks.append({"id": check_id, "state": "unavailable", "reason": "no equivalent validation reference result"})
        elif quiet.get("pass") is True:
            checks.append({"id": check_id, "state": "passed", "reason": "quiet response is within the documented policy", "metrics": quiet})
        elif quiet.get("pass") is False:
            checks.append({"id": check_id, "state": "failed", "reason": quiet.get("reason") or "quiet-playing response differs from the frozen teacher", "metrics": quiet})
        else:
            checks.append({"id": check_id, "state": "unavailable", "reason": quiet.get("reason", "check unavailable"), "metrics": quiet})

    states = {check["state"] for check in checks}
    state = "needs_attention" if "failed" in states else "unavailable" if "unavailable" in states else "passed"
    cabinet_record = dict(cabinet or {"baked": False, "approximation": None})
    approximation = cabinet_record.get("approximation", cabinet_record.get("cab_requires_approximation"))
    cabinet_record["approximation"] = approximation
    cabinet_record["note"] = (
        "The baked cabinet exceeds the destination temporal field and is an approximation; inspect metrics and listen."
        if cabinet_record.get("baked") and approximation is True else
        "The baked cabinet fits the recorded destination temporal field."
        if cabinet_record.get("baked") and approximation is False else
        "Cabinet approximation status is unavailable."
        if cabinet_record.get("baked") else
        "No cabinet was baked into the trained target."
    )
    return {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "policy_version": VALIDATION_POLICY_VERSION,
        "policy": effective_policy,
        "model_sha256": model_sha256,
        "state": state,
        "checks": checks,
        "cabinet": cabinet_record,
        "summary": (
            "Validation found one or more issues; inspect the failed checks and listen to the export."
            if state == "needs_attention" else
            "Validation is incomplete because one or more checks are unavailable."
            if state == "unavailable" else
            "Required technical checks passed. This does not establish perceptual equivalence."
        ),
    }
