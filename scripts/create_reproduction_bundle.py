#!/usr/bin/env python
"""Create reproduction manifest, report, and replay script for a pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.reproducibility import build_reproduction_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/reproduction_manifest.json.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/reproduction_report.md.",
    )
    parser.add_argument("--script-output", help="Shell script path. Defaults to RUN_DIR/reproduce_run.sh.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "reproduction_manifest.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else run_dir / "reproduction_report.md"
    script_output = Path(args.script_output) if args.script_output else run_dir / "reproduce_run.sh"
    try:
        bundle = build_reproduction_bundle(run_dir)
        bundle.to_json(output)
        bundle.to_markdown(markdown_output)
        bundle.to_shell_script(script_output)
    except Exception as exc:
        print(f"create_reproduction_bundle failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(bundle.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    print(f"Wrote {script_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
