#!/usr/bin/env python
"""Prepare Nerfstudio data from video or images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.data_processing import prepare_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input video or image directory.")
    parser.add_argument("--output", required=True, help="Output processed scene directory.")
    parser.add_argument("--type", required=True, choices=["video", "images"], help="Input type.")
    parser.add_argument("--log-path", help="Optional command log JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and create mock metadata.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        metadata = prepare_data(
            args.input,
            args.output,
            args.type,
            dry_run=args.dry_run,
            command_log_path=args.log_path,
        )
    except Exception as exc:
        print(f"prepare_data failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
