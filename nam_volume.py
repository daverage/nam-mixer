#!/usr/bin/env python3
"""Safely create a level-adjusted copy of a Neural Amp Modeler .nam file.

Examples: ``python nam_volume.py Mesa_Boogie.nam +6`` or
``python nam_volume.py Mesa_Boogie.nam -3 --dry-run``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from hybrid.nam_tools import NamToolError, apply_volume_change, compare_changes, load_nam, save_nam


def _default_output(source: Path, db: float) -> Path:
    return source.with_name(f"{source.stem}_{db:+g}dB{source.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a safely volume-adjusted copy of a NAM file.")
    parser.add_argument("input", type=Path, help="source .nam file")
    parser.add_argument("db_change", type=float, help="output change in dB, e.g. +6 or -3.5")
    parser.add_argument("--output", type=Path, help="destination .nam path")
    parser.add_argument("--overwrite", action="store_true", help="replace the source NAM (otherwise never overwrite it)")
    parser.add_argument("--dry-run", action="store_true", help="show approved changes without writing")
    parser.add_argument("--verbose", action="store_true", help="show all changed JSON paths")
    args = parser.parse_args(argv)
    if args.input.suffix.lower() != ".nam":
        parser.error("input must be a .nam file")
    output = args.output or (_default_output(args.input, args.db_change))
    if output.resolve() == args.input.resolve() and not args.overwrite:
        parser.error("refusing to overwrite the original; use --output or explicitly pass --overwrite")
    if output.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"output already exists: {output} (use --overwrite to replace it)")
    try:
        original = load_nam(args.input)
        edited, expected_paths, multiplier = apply_volume_change(original, args.db_change)
    except (OSError, ValueError, NamToolError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"File: {args.input}\nArchitecture: {original.get('architecture')}\nRequested change: {args.db_change:+.2f} dB\nLinear multiplier: {multiplier:.6f}")
    print("\nChanges:")
    for path in expected_paths:
        print(f"  {path}")
    print("metadata.gain: unchanged")
    if args.db_change > 12:
        print("Warning: boosts above +12 dB may clip in the host or target hardware.", file=sys.stderr)
    if args.dry_run:
        return 0
    try:
        save_nam(edited, output)
        persisted = load_nam(output)
        if compare_changes(original, persisted) != expected_paths:
            output.unlink(missing_ok=True)
            raise NamToolError("saved file failed approved-path validation")
    except (OSError, ValueError, NamToolError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"\nWrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
