#!/usr/bin/env python
"""Validate manual annotations before query evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.annotation_validation import validate_annotations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Annotation JSON file.")
    parser.add_argument("--queries", help="Optional query YAML for coverage checks.")
    parser.add_argument("--results", help="Optional query results directory for view-id checks.")
    parser.add_argument("--output", default="results/annotation_validation.json")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings as well as errors.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = validate_annotations(
            args.annotations,
            queries_path=args.queries,
            results_dir=args.results,
        )
    except Exception as exc:
        print(f"validate_annotations failed: {exc}", file=sys.stderr)
        return 1
    output = Path(args.output)
    report.to_json(output)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
