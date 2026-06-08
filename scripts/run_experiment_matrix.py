#!/usr/bin/env python
"""Run or summarize a configured experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.experiment_matrix import run_experiment_matrix  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment matrix config.")
    parser.add_argument("--output", help="Output directory for matrix artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Force all experiments into dry-run mode.")
    parser.add_argument("--real-run", action="store_true", help="Force all experiments into real mode.")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Do not launch pipeline runs; summarize existing run directories under the output.",
    )
    parser.add_argument("--limit", type=int, help="Run or collect only the first N experiments.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run and args.real_run:
        print("run_experiment_matrix failed: choose only one of --dry-run or --real-run", file=sys.stderr)
        return 2
    dry_run = True if args.dry_run else False if args.real_run else None
    try:
        report = run_experiment_matrix(
            config_path=args.config,
            output_dir=args.output,
            dry_run=dry_run,
            collect_only=args.collect_only,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"run_experiment_matrix failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    output = Path(report.output_dir)
    print(f"\nWrote {output / 'experiment_matrix_summary.json'}")
    print(f"Wrote {output / 'experiment_matrix_table.csv'}")
    print(f"Wrote {output / 'experiment_matrix_report.md'}")
    return 0 if report.successful_experiments or args.collect_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
