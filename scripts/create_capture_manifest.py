#!/usr/bin/env python
"""Create and validate a reproducible scene-capture manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.capture_manifest import (  # noqa: E402
    build_capture_manifest,
    write_capture_manifest_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw video file or image directory.")
    parser.add_argument("--type", choices=["video", "images"], default="video")
    parser.add_argument("--scene-name", default="desk_scene")
    parser.add_argument("--output", default="results/capture_manifest", help="Output directory.")
    parser.add_argument("--capture-device", default="unknown")
    parser.add_argument("--scene-type", default="unknown")
    parser.add_argument("--lighting", default="unknown")
    parser.add_argument("--camera-motion", default="unknown")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--approximate-frame-count", type=int)
    parser.add_argument("--static-scene", action="store_true")
    parser.add_argument("--not-static-scene", dest="static_scene", action="store_false")
    parser.set_defaults(static_scene=None)
    parser.add_argument("--high-overlap", action="store_true")
    parser.add_argument("--low-overlap", dest="high_overlap", action="store_false")
    parser.set_defaults(high_overlap=None)
    parser.add_argument("--privacy-reviewed", action="store_true")
    parser.add_argument("--contains-people", action="store_true")
    parser.add_argument("--contains-private-text", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--min-images", type=int, default=50)
    parser.add_argument("--require-privacy-review", action="store_true")
    parser.add_argument("--allow-warnings", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_capture_manifest(
        input_path=args.input,
        input_type=args.type,
        scene_name=args.scene_name,
        capture_device=args.capture_device,
        scene_type=args.scene_type,
        lighting=args.lighting,
        camera_motion=args.camera_motion,
        duration_seconds=args.duration_seconds,
        approximate_frame_count=args.approximate_frame_count,
        static_scene=args.static_scene,
        high_overlap=args.high_overlap,
        privacy_reviewed=args.privacy_reviewed,
        contains_people=args.contains_people if args.contains_people else None,
        contains_private_text=args.contains_private_text if args.contains_private_text else None,
        notes=args.notes,
    )
    manifest_json, manifest_md, validation_json, validation_md, validation = write_capture_manifest_bundle(
        manifest,
        args.output,
        min_images=args.min_images,
        require_privacy_review=args.require_privacy_review,
    )
    payload = {
        "manifest": manifest.to_dict(),
        "validation": validation.to_dict(),
        "outputs": {
            "manifest_json": str(manifest_json),
            "manifest_markdown": str(manifest_md),
            "validation_json": str(validation_json),
            "validation_markdown": str(validation_md),
        },
    }
    print(json.dumps(payload, indent=2))
    if validation.ok or args.allow_warnings:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
