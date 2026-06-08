#!/usr/bin/env python
"""Analyze prompt robustness for open-vocabulary scene queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.prompt_sensitivity import (  # noqa: E402
    analyze_prompt_sensitivity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Prompt sensitivity YAML suite.")
    parser.add_argument("--results", required=True, help="Directory containing query_result.json files.")
    parser.add_argument("--output", default="results/prompt_sensitivity", help="Output directory.")
    parser.add_argument("--scene-name", help="Override scene name in the report.")
    parser.add_argument("--min-mean-confidence", type=float, default=0.55)
    parser.add_argument("--min-box-consistency-iou", type=float, default=0.25)
    parser.add_argument("--min-view-agreement", type=float, default=0.67)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic prompt results if --results has no query_result.json files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = analyze_prompt_sensitivity(
            suite_path=args.suite,
            results_dir=args.results,
            output_dir=args.output,
            scene_name=args.scene_name,
            dry_run=args.dry_run,
            min_mean_confidence=args.min_mean_confidence,
            min_box_consistency_iou=args.min_box_consistency_iou,
            min_view_agreement=args.min_view_agreement,
        )
    except Exception as exc:
        print(f"analyze_prompt_sensitivity failed: {exc}", file=sys.stderr)
        return 1
    output = Path(args.output)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output / 'prompt_sensitivity_summary.json'}")
    print(f"Wrote {output / 'prompt_sensitivity_table.csv'}")
    print(f"Wrote {output / 'prompt_sensitivity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
