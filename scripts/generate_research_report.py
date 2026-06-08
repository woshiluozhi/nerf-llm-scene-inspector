#!/usr/bin/env python
"""Generate a research-style Markdown report from a pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.research_report import write_research_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--output", help="Markdown output path. Defaults to RUN_DIR/research_report.md.")
    parser.add_argument("--json-output", help="JSON output path. Defaults to RUN_DIR/research_report.json.")
    parser.add_argument("--matrix-summary", help="Optional experiment_matrix_summary.json to reference.")
    parser.add_argument("--title", default="NeRF-LLM Scene Inspector Research Report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = write_research_report(
            args.run_dir,
            output=args.output,
            json_output=args.json_output,
            matrix_summary_path=args.matrix_summary,
            title=args.title,
        )
    except Exception as exc:
        print(f"generate_research_report failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    run_dir = Path(args.run_dir)
    print(f"\nWrote {Path(args.output) if args.output else run_dir / 'research_report.md'}")
    print(f"Wrote {Path(args.json_output) if args.json_output else run_dir / 'research_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
