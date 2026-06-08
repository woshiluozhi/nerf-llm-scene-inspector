#!/usr/bin/env python
"""Generate visual QA artifacts for manual query annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.annotation_review import build_annotation_review  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Annotation JSON file.")
    parser.add_argument("--results", required=True, help="Directory containing query_result.json files.")
    parser.add_argument("--output", default="results/evaluation", help="Output directory.")
    parser.add_argument("--json-output", help="Defaults to OUTPUT/annotation_review.json.")
    parser.add_argument("--markdown-output", help="Defaults to OUTPUT/annotation_review.md.")
    parser.add_argument("--max-sheet-columns", type=int, default=2)
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Exit 0 even when view fallbacks, missing boxes, or missing results are reported.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    json_output = Path(args.json_output) if args.json_output else output / "annotation_review.json"
    markdown_output = (
        Path(args.markdown_output) if args.markdown_output else output / "annotation_review.md"
    )
    try:
        report = build_annotation_review(
            annotations_path=args.annotations,
            results_dir=args.results,
            output_dir=output,
            max_sheet_columns=args.max_sheet_columns,
        )
        report.to_json(json_output)
        report.to_markdown(markdown_output)
    except Exception as exc:
        print(f"review_annotations failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {json_output}")
    print(f"Wrote {markdown_output}")
    if report.contact_sheet:
        print(f"Wrote {output / report.contact_sheet}")
    return 0 if report.ok or args.allow_warnings else 1


if __name__ == "__main__":
    raise SystemExit(main())
