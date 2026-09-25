"""Epoch-progress extraction from a trainer log -- UX only, never fatal.

The trainers (scripts/train_a2.py and cloud/kaggle/train_a2_cloud.py) add a
Lightning callback that prints ``NAM Mixer: epoch N of M`` (1-based) as each
epoch starts. That line is the authoritative source, because Lightning's own
progress bar is not usable for this:

- its Rich bar only renders live on a terminal; through a pipe (the local UI)
  or a Kaggle kernel log it prints nothing until training ends;
- both its Rich (``Epoch 19/19``) and tqdm (``Epoch 19:``) bars are ZERO-based
  and the Rich total is ``max_epochs - 1``, so a 20-epoch run ends on
  ``Epoch 19/19``.

The Lightning forms are still parsed (converted to 1-based) as a fallback for
logs from trainers that predate the marker.
"""
from __future__ import annotations

import re
from typing import Optional

EPOCH_MARKER = "NAM Mixer: epoch"

_MARKER_RE = re.compile(r"NAM Mixer: epoch\s+(\d+)\s+of\s+(\d+)")
_RICH_RE = re.compile(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)")
_TQDM_RE = re.compile(r"[Ee]poch\s+(\d+)\s*[:|]")
_TOTAL_RE = re.compile(r"(?:epoch(?:s)?|for)\D{0,20}(\d+)\s+epochs?", re.IGNORECASE)


def parse_epoch_progress(log_text: Optional[str]) -> Optional[dict]:
    """``{"epoch": <1-based epoch running>, "total_epochs": N}`` or None."""
    try:
        text = log_text or ""
        marker = _MARKER_RE.findall(text)
        if marker:
            current, total = marker[-1]
            return {"epoch": int(current), "total_epochs": int(total)}
        rich = _RICH_RE.findall(text)
        if rich:
            current, last = rich[-1]
            return {"epoch": int(current) + 1, "total_epochs": int(last) + 1}
        current = _TQDM_RE.findall(text)
        totals = _TOTAL_RE.findall(text)
        if current and totals:
            return {"epoch": int(current[-1]) + 1, "total_epochs": int(totals[-1])}
        return None
    except (ValueError, TypeError):
        return None
