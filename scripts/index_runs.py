#!/usr/bin/env python
"""Index pipeline runs for experiment tracking and portfolio review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.run_index import index_pipeline_runs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/pipeline_runs", help="Directory containing run subdirectories.")
    parser.add_argument("--output", help="JSON output path. Defaults to ROOT/run_index.json.")
    parser.add_argument("--markdown-output", help="Markdown output path. Defaults to ROOT/run_index.md.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "run_index.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else root / "run_index.md"
    try:
        index = index_pipeline_runs(root)
        index.to_json(output)
        index.to_markdown(markdown_output)
    except Exception as exc:
        print(f"index_runs failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(index.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
