#!/usr/bin/env python
"""Diagnose failed or degraded pipeline runs from saved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.failure_diagnostics import (  # noqa: E402
    write_failure_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory to diagnose.")
    parser.add_argument(
        "--output",
        help="JSON output path. Defaults to RUN_DIR/failure_diagnostics.json.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/failure_diagnostics.md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for warning-level diagnostics as well as blockers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "failure_diagnostics.json"
    markdown_output = (
        Path(args.markdown_output) if args.markdown_output else run_dir / "failure_diagnostics.md"
    )
    try:
        report = write_failure_diagnostics(
            run_dir,
            output=output,
            markdown_output=markdown_output,
        )
    except Exception as exc:
        print(f"diagnose_run_failures failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    if report.status == "blocked":
        return 1
    if args.strict and report.status != "clear":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
