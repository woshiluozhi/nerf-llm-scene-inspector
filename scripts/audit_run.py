#!/usr/bin/env python
"""Audit a pipeline run directory for completeness and portfolio readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.run_audit import audit_pipeline_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory to audit.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/run_audit.json.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/run_audit.md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for warning-level findings as well as blockers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "run_audit.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else run_dir / "run_audit.md"

    try:
        report = audit_pipeline_run(run_dir)
        report.to_json(output)
        report.to_markdown(markdown_output)
    except Exception as exc:
        print(f"audit_run failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    if report.status == "blocked":
        return 1
    if args.strict and report.status != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
