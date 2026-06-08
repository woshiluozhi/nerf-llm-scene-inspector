#!/usr/bin/env python
"""Import manually saved LERF viewer outputs as a structured QueryResult."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.querying.viewer_import import import_viewer_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Text query represented by the saved viewer outputs.")
    parser.add_argument("--config", required=True, help="Trained LERF/Nerfstudio config.yml.")
    parser.add_argument("--input", required=True, help="Directory with viewer screenshots/renders.")
    parser.add_argument("--output", required=True, help="Output directory for query_result.json.")
    parser.add_argument("--backend", default="lerf", help="Backend name to store in QueryResult.")
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument(
        "--no-create-overlays",
        action="store_true",
        help="Do not generate overlay images when RGB and relevancy files are present.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result, summary = import_viewer_outputs(
            query=args.query,
            config_path=args.config,
            input_dir=args.input,
            output_dir=args.output,
            backend_name=args.backend,
            threshold_quantile=args.threshold_quantile,
            create_missing_overlays=not args.no_create_overlays,
        )
    except Exception as exc:
        print(f"import_viewer_outputs failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"query_result": result.to_dict(), "summary": summary.to_dict()}, indent=2))
    print(f"\nWrote {summary.query_result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
