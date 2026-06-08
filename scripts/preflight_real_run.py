#!/usr/bin/env python
"""Run CPU-safe preflight checks before an expensive real-scene NeRF/LERF run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.preflight import build_real_run_preflight  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Raw phone video file or image directory.")
    parser.add_argument("--type", choices=["video", "images"], default="video")
    parser.add_argument("--data", help="Optional processed Nerfstudio scene directory.")
    parser.add_argument("--config", help="Optional trained language-field config.yml.")
    parser.add_argument("--scene-name", default="desk_scene")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--variant", default="lerf-lite")
    parser.add_argument("--output", default="results/preflight", help="Directory for report files.")
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--max-missing-image-ratio", type=float, default=0.0)
    parser.add_argument("--min-pose-extent", type=float, default=0.05)
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument(
        "--check-upstream",
        dest="check_upstream",
        action="store_true",
        default=True,
        help="Require Nerfstudio/LERF shell commands and ns-train methods.",
    )
    parser.add_argument(
        "--no-check-upstream",
        dest="check_upstream",
        action="store_false",
        help="Skip treating upstream Nerfstudio/LERF commands as required.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Relax raw-input checks for smoke tests; no heavy external commands are run.",
    )
    parser.add_argument("--allow-warnings", action="store_true", help="Exit 0 for warn-only reports.")
    parser.add_argument("--json", action="store_true", help="Print only the structured JSON report.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_real_run_preflight(
        input_path=args.input,
        input_type=args.type,
        data_path=args.data,
        config_path=args.config,
        scene_name=args.scene_name,
        backend=args.backend,
        variant=args.variant,
        min_frames=args.min_frames,
        max_missing_image_ratio=args.max_missing_image_ratio,
        min_pose_extent=args.min_pose_extent,
        require_gpu=args.require_gpu,
        check_upstream=args.check_upstream,
        dry_run=args.dry_run,
    )

    output_dir = Path(args.output)
    json_path = report.to_json(output_dir / "preflight_report.json")
    md_path = report.to_markdown(output_dir / "preflight_report.md")
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
        print(f"\nWrote {json_path}")
        print(f"Wrote {md_path}")

    if report.status == "ready":
        return 0
    if report.status == "needs_attention" and args.allow_warnings:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
