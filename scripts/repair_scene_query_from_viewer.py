#!/usr/bin/env python
"""Repair a scene_query_report.json with manually saved viewer outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.querying.viewer_import import (  # noqa: E402
    repair_scene_query_report_from_viewer_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Existing scene_query_report.json to repair.")
    parser.add_argument(
        "--viewer-root",
        required=True,
        help="Directory containing one subdirectory per query, named by query slug.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON report path. Defaults to overwriting --report.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Output Markdown report path. Defaults to output path with .md suffix.",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument(
        "--no-create-overlays",
        action="store_true",
        help="Do not generate side-by-side overlays when RGB and relevancy files are present.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit non-zero unless every query in the report has a matching viewer-output directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report, summary = repair_scene_query_report_from_viewer_outputs(
            report_path=args.report,
            viewer_root=args.viewer_root,
            output_report_path=args.output,
            markdown_report_path=args.markdown_output,
            threshold_quantile=args.threshold_quantile,
            create_missing_overlays=not args.no_create_overlays,
            require_all=args.require_all,
        )
    except Exception as exc:
        print(f"repair_scene_query_from_viewer failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"scene_query_report": report.to_dict(), "summary": summary.to_dict()}, indent=2))
    print(f"\nWrote {summary.output_report_path}")
    print(f"Wrote {summary.markdown_report_path}")
    if args.require_all and not summary.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
