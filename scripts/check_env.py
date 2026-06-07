#!/usr/bin/env python
"""Check local and optional upstream environment readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.utils.env_check import build_env_report, format_report_table  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Write structured JSON to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Show successful optional checks too.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument(
        "--check-upstream",
        action="store_true",
        help="Treat Nerfstudio/LERF/OpenNeRF shell tools and methods as required.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_env_report(require_gpu=args.require_gpu, check_upstream=args.check_upstream)
    if args.json:
        print(report.to_json())
    else:
        print(format_report_table(report, verbose=args.verbose))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
