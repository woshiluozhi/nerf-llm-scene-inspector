#!/usr/bin/env python
"""Validate an exported portfolio pack before sharing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        default="results/portfolio_pack",
        help="Exported portfolio pack directory or .zip archive.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path. Defaults to <pack>/portfolio_pack_validation.json.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on warnings as well as errors.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_portfolio_pack(args.pack)
    output = Path(args.output) if args.output else _default_output_path(Path(args.pack))
    report.to_json(output)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


def _default_output_path(pack: Path) -> Path:
    if pack.suffix.lower() == ".zip":
        return pack.with_name(f"{pack.stem}_validation.json")
    return pack / "portfolio_pack_validation.json"


if __name__ == "__main__":
    raise SystemExit(main())
