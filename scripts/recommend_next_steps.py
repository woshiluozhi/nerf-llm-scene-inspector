#!/usr/bin/env python
"""Generate actionable next-step recommendations for a pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.run_recommendations import build_run_recommendations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/run_recommendations.json.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/run_recommendations.md.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the run is portfolio-ready.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "run_recommendations.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else run_dir / "run_recommendations.md"
    try:
        report = build_run_recommendations(run_dir)
        report.to_json(output)
        report.to_markdown(markdown_output)
    except Exception as exc:
        print(f"recommend_next_steps failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    if report.readiness_level == "blocked":
        return 1
    if args.strict and report.readiness_level != "ready_for_portfolio":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
