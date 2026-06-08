#!/usr/bin/env python
"""Create a portfolio evidence scorecard for a pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.evidence_scorecard import build_evidence_scorecard  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/evidence_scorecard.json.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/evidence_scorecard.md.",
    )
    parser.add_argument("--json", action="store_true", help="Print only JSON payload.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    scorecard = build_evidence_scorecard(run_dir)
    json_path = Path(args.output) if args.output else run_dir / "evidence_scorecard.json"
    markdown_path = (
        Path(args.markdown_output) if args.markdown_output else run_dir / "evidence_scorecard.md"
    )
    scorecard.to_json(json_path)
    scorecard.to_markdown(markdown_path)
    payload = scorecard.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
        print(f"\nWrote {json_path}")
        print(f"Wrote {markdown_path}")
    return 0 if scorecard.evidence_level != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
