#!/usr/bin/env python
"""Finalize a run after editing annotation workbench labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.annotation_finalize import finalize_workbench_annotations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--filled", required=True, help="JSON downloaded from annotation_workbench.html.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "real-run", "portfolio"],
        default="smoke",
        help="Quality-gate profile to refresh after evaluation.",
    )
    parser.add_argument("--pack", help="Optional portfolio pack directory for quality/claim/submission checks.")
    parser.add_argument("--export-pack", action="store_true", help="Export a fresh portfolio pack after refreshing reports.")
    parser.add_argument("--zip-pack", action="store_true", help="Also create PACK.zip when --export-pack is set.")
    parser.add_argument("--dry-run-eval", action="store_true", help="Allow evaluate_queries.py to synthesize missing dry-run results.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k localization threshold passed to evaluate_queries.py.")
    parser.add_argument("--repo-url", default="", help="Repository URL to include in refreshed submission packet.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue non-destructive refresh steps after a failure.")
    parser.add_argument("--output", help="JSON report path. Defaults to RUN_DIR/annotation_finalize_report.json.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown report path. Defaults to RUN_DIR/annotation_finalize_report.md.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = finalize_workbench_annotations(
            run_dir=args.run_dir,
            filled_path=args.filled,
            profile=args.profile,
            pack_dir=args.pack,
            export_pack=args.export_pack,
            zip_pack=args.zip_pack,
            dry_run_eval=args.dry_run_eval,
            top_k=args.top_k,
            repo_url=args.repo_url,
            continue_on_error=args.continue_on_error,
            report_output=args.output,
            markdown_output=args.markdown_output,
        )
    except Exception as exc:
        print(f"finalize_annotations failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    run_dir = Path(args.run_dir)
    print(f"\nWrote {Path(args.output) if args.output else run_dir / 'annotation_finalize_report.json'}")
    print(f"Wrote {Path(args.markdown_output) if args.markdown_output else run_dir / 'annotation_finalize_report.md'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
