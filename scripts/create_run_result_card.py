#!/usr/bin/env python
"""Create a concise reviewer-facing result card for one pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.run_result_card import write_run_result_card  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--output", help="Markdown output path. Defaults to RUN_DIR/run_result_card.md.")
    parser.add_argument("--json-output", help="JSON output path. Defaults to RUN_DIR/run_result_card.json.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        card = write_run_result_card(
            args.run_dir,
            output=args.output,
            json_output=args.json_output,
        )
    except Exception as exc:
        print(f"create_run_result_card failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(card.to_dict(), indent=2))
    run_dir = Path(args.run_dir)
    print(f"\nWrote {Path(args.output) if args.output else run_dir / 'run_result_card.md'}")
    print(f"Wrote {Path(args.json_output) if args.json_output else run_dir / 'run_result_card.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
