#!/usr/bin/env python
"""Verify run-local reproduction manifest artifact hashes and sizes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.reproducibility import verify_reproduction_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument(
        "--manifest",
        help="Manifest path. Defaults to RUN_DIR/reproduction_manifest.json.",
    )
    parser.add_argument(
        "--output",
        help="JSON output path. Defaults to RUN_DIR/reproduction_manifest_validation.json.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/reproduction_manifest_validation.md.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if the manifest records any artifact as missing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "reproduction_manifest_validation.json"
    markdown_output = (
        Path(args.markdown_output) if args.markdown_output else run_dir / "reproduction_manifest_validation.md"
    )
    report = verify_reproduction_manifest(
        run_dir,
        manifest_path=args.manifest,
        require_complete=args.require_complete,
    )
    report.to_json(output)
    report.to_markdown(markdown_output)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
