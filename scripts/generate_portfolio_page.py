#!/usr/bin/env python
"""Generate a static HTML portfolio page for a pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.visualization.portfolio_page import build_portfolio_page  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument(
        "--output",
        help="HTML output path. Defaults to RUN_DIR/portfolio_page.html.",
    )
    parser.add_argument("--json", action="store_true", help="Print a structured summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "portfolio_page.html"
    page = build_portfolio_page(run_dir)
    page.write_html(output)
    payload = {
        "output": str(output),
        "scene_name": page.scene_name,
        "evidence_level": page.evidence_level,
        "evidence_score": page.evidence_score,
        "image_count": len(page.images),
        "artifact_count": len(page.artifacts),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
