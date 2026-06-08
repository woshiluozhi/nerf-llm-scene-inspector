#!/usr/bin/env python
"""Merge an offline annotation workbench export into evaluation annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.annotation_merge import merge_workbench_annotations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="Original annotation_template.json file.")
    parser.add_argument("--filled", required=True, help="JSON exported from annotation_workbench.html.")
    parser.add_argument("--output", default="results/annotations_merged.json", help="Merged annotation JSON.")
    parser.add_argument(
        "--report-output",
        default="results/annotation_merge_report.json",
        help="Structured merge report JSON.",
    )
    parser.add_argument("--queries", help="Optional query YAML for validation coverage checks.")
    parser.add_argument("--results", help="Optional query results directory for view-id validation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    parser.add_argument("--strict", action="store_true", help="Fail on validation warnings as well as errors.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = merge_workbench_annotations(
            template_path=args.template,
            filled_path=args.filled,
            output_path=args.output,
            report_path=args.report_output,
            queries_path=args.queries,
            results_dir=args.results,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"merge_annotation_workbench failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.report_output}")
    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
