#!/usr/bin/env python
"""Inspect a processed Nerfstudio scene before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.scene_validation import inspect_processed_scene  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Processed scene directory with transforms.json.")
    parser.add_argument("--output", default="results/scene_checks", help="Directory for inspection reports.")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-missing-image-ratio", type=float, default=0.0)
    parser.add_argument(
        "--min-pose-extent",
        type=float,
        default=0.05,
        help="Minimum camera translation extent required across any axis.",
    )
    parser.add_argument("--allow-warnings", action="store_true", help="Exit 0 even if not training-ready.")
    parser.add_argument("--json", action="store_true", help="Print only the JSON report.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inspection = inspect_processed_scene(
        args.data,
        min_frames=args.min_frames,
        max_missing_image_ratio=args.max_missing_image_ratio,
        min_pose_extent=args.min_pose_extent,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = inspection.to_json(output_dir / "scene_data_inspection.json")
    md_path = inspection.to_markdown(output_dir / "scene_data_inspection.md")
    payload = inspection.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
        print(f"\nWrote {json_path}")
        print(f"Wrote {md_path}")
    if inspection.ready_for_training or args.allow_warnings:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
