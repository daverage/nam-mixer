"""Loading .nam model files and exposing their calibration metadata.

A .nam file is JSON (architecture + config + weights + optional metadata). Parsing
that JSON and pulling out calibration fields does NOT require torch or the
`neural-amp-modeler` package, so it is fully implemented here. Actually running the
model (turning `NamModel.weights`/`config` into audio) DOES require torch + the NAM
architectures and is deliberately NOT implemented in this module -- see render.py for
the current status of that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Calibration metadata has appeared under a few different keys across NAM trainer
# versions. We check all of them rather than assuming one fixed schema.
_CALIBRATION_KEY_CANDIDATES = (
    ("input_level_dbu",),
    ("metadata", "input_level_dbu"),
    ("metadata", "loudness", "input_level_dbu"),
)
_OUTPUT_CALIBRATION_KEY_CANDIDATES = (
    ("output_level_dbu",),
    ("metadata", "output_level_dbu"),
    ("metadata", "loudness", "output_level_dbu"),
)


def _dig(d: dict, path: tuple) -> Optional[Any]:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_present(d: dict, candidates: tuple) -> Optional[float]:
    for path in candidates:
        val = _dig(d, path)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class NamModel:
    """A loaded .nam file: raw JSON plus convenience accessors.

    `weights`/`config`/`architecture` are kept as raw dict/list data exactly as
    parsed -- no torch tensors are constructed here. See render.py for why.
    """

    path: Path
    raw: dict = field(repr=False)

    @property
    def architecture(self) -> Optional[str]:
        return self.raw.get("architecture")

    @property
    def config(self) -> dict:
        return self.raw.get("config", {})

    @property
    def sample_rate(self) -> Optional[float]:
        return self.raw.get("sample_rate")

    @property
    def input_level_dbu(self) -> Optional[float]:
        return _first_present(self.raw, _CALIBRATION_KEY_CANDIDATES)

    @property
    def output_level_dbu(self) -> Optional[float]:
        return _first_present(self.raw, _OUTPUT_CALIBRATION_KEY_CANDIDATES)

    @property
    def is_calibrated(self) -> bool:
        """True only when BOTH input and output calibration levels are present.

        A NAM file with only one of the two is treated as uncalibrated for our
        purposes, since a one-sided calibration can't be used to level-match
        against another model.
        """
        return self.input_level_dbu is not None and self.output_level_dbu is not None

    @property
    def calibration_status(self) -> str:
        return "Calibrated NAM" if self.is_calibrated else "Calibration metadata unavailable"

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "architecture": self.architecture,
            "sample_rate": self.sample_rate,
            "input_level_dbu": self.input_level_dbu,
            "output_level_dbu": self.output_level_dbu,
            "calibration_status": self.calibration_status,
        }


def load_nam(path: str | Path) -> NamModel:
    """Parse a .nam file's JSON and return a NamModel.

    Does not reject older/uncalibrated files -- calibration_status simply reports
    "Calibration metadata unavailable" for them. Raises the underlying JSON/IO
    error unchanged if the file can't be read/parsed, since there's no sensible
    fallback for a .nam file that isn't valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return NamModel(path=path, raw=raw)
