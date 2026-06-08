#!/usr/bin/env python
"""Create an offline HTML workbench for manual bbox annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.annotation_workbench import build_annotation_workbench  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Annotation template JSON.")
    parser.add_argument("--results", required=True, help="Query results directory.")
    parser.add_argument("--output", default="results/annotation_workbench", help="Output directory.")
    parser.add_argument("--title", default="NeRF-LLM Annotation Workbench")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        workbench = build_annotation_workbench(
            annotations_path=args.annotations,
            results_dir=args.results,
            output_dir=args.output,
            title=args.title,
        )
    except Exception as exc:
        print(f"create_annotation_workbench failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(workbench.to_dict(), indent=2))
    output = Path(args.output)
    print(f"\nWrote {output / 'annotation_workbench.html'}")
    print(f"Wrote {output / 'annotation_workbench_manifest.json'}")
    print(f"Wrote {output / 'annotation_seed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
