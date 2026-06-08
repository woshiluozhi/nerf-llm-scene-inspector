#!/usr/bin/env python
"""Create a run-level readiness gate report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.run_readiness import write_run_readiness  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/run_readiness.json.")
    parser.add_argument("--markdown-output", help="Markdown output path. Defaults to RUN_DIR/run_readiness.md.")
    parser.add_argument("--pack", help="Optional portfolio pack directory or zip to include in readiness gates.")
    parser.add_argument(
        "--pack-validation",
        help="Optional existing portfolio_pack_validation.json. Overrides live pack validation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = write_run_readiness(
            args.run_dir,
            output=args.output,
            markdown_output=args.markdown_output,
            pack_dir=args.pack,
            pack_validation_path=args.pack_validation,
        )
    except Exception as exc:
        print(f"create_run_readiness failed: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else Path(args.run_dir) / "run_readiness.json"
    markdown = Path(args.markdown_output) if args.markdown_output else Path(args.run_dir) / "run_readiness.md"
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown}")
    return 0 if report.fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
