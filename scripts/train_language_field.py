#!/usr/bin/env python
"""Train a LERF or OpenNeRF language field."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.training import train_language_field  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Processed Nerfstudio data directory.")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--variant", choices=["lerf", "lerf-lite", "lerf-big", "opennerf"], default="lerf-lite")
    parser.add_argument("--max-num-iterations", type=int, default=None)
    parser.add_argument("--output", required=True, help="Training output directory.")
    parser.add_argument("--log-path", help="Optional training command log JSON path.")
    parser.add_argument("--method-check-log-path", help="Optional ns-train method check log JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and create mock summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = train_language_field(
            args.data,
            args.backend,
            args.variant,
            args.output,
            max_num_iterations=args.max_num_iterations,
            dry_run=args.dry_run,
            command_log_path=args.log_path,
            method_check_log_path=args.method_check_log_path,
        )
    except Exception as exc:
        print(f"train_language_field failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    if summary.get("config_path"):
        print("\nLaunch viewer:")
        print(f"ns-viewer --load-config {summary['config_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
