#!/usr/bin/env python
"""Collect portfolio-facing project artifacts into results/portfolio_pack."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/portfolio_pack")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    files = [
        ROOT / "README.md",
        ROOT / "docs" / "portfolio_result_card.md",
        ROOT / "docs" / "cv_bullets.md",
        ROOT / "docs" / "cold_email_paragraph.md",
        ROOT / "results" / "dry_run_demo_summary.json",
        ROOT / "results" / "evaluation" / "eval_summary.json",
    ]
    copied: list[str] = []
    missing: list[str] = []
    for path in files:
        if path.exists():
            destination = output / path.name
            shutil.copy2(path, destination)
            copied.append(str(destination))
        else:
            missing.append(str(path))
    index = {
        "copied": copied,
        "missing": missing,
        "github": "https://github.com/woshiluozhi/nerf-llm-scene-inspector",
        "recommended_demo_command": "python scripts/run_dry_run_demo.py",
    }
    (output / "portfolio_pack_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
