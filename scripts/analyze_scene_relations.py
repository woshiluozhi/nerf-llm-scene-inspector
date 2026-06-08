#!/usr/bin/env python
"""Build a scene-level relation graph from query_result.json artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.scene_relations import analyze_scene_relations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Directory containing query_result.json files.")
    parser.add_argument("--output", required=True, help="Directory for relation graph artifacts.")
    parser.add_argument("--scene-name", default="unknown")
    parser.add_argument("--top-k-per-query", type=int, default=1)
    parser.add_argument("--max-edges", type=int, default=100)
    parser.add_argument("--near-threshold-px", type=float, default=120.0)
    parser.add_argument("--near-threshold-3d", type=float, default=0.5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create synthetic relation artifacts when no query results are available.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results)
    if not results_dir.exists() and not args.dry_run:
        print(
            f"analyze_scene_relations failed: results directory does not exist: {results_dir}",
            file=sys.stderr,
        )
        print("Run query_scene.py or run_scene_pipeline.py first, or pass --dry-run.", file=sys.stderr)
        return 1
    report = analyze_scene_relations(
        results_dir=results_dir,
        output_dir=args.output,
        scene_name=args.scene_name,
        top_k_per_query=args.top_k_per_query,
        max_edges=args.max_edges,
        near_threshold_px=args.near_threshold_px,
        near_threshold_3d=args.near_threshold_3d,
        dry_run=args.dry_run,
    )
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {Path(args.output) / 'scene_relations_summary.json'}")
    print(f"Wrote {Path(args.output) / 'scene_relations_edges.csv'}")
    print(f"Wrote {Path(args.output) / 'scene_relations_report.md'}")
    return 0 if report.entities or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
