#!/usr/bin/env python
"""Run the complete dry-run portfolio demo.

This script intentionally avoids GPU-only dependencies. It exercises the same
public pipeline entry points that a real Nerfstudio/LERF run would use, but it
creates mock metadata and synthetic relevancy visualizations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.data_processing import prepare_data  # noqa: E402
from nerf_llm_scene_inspector.training import train_baseline_nerf, train_language_field  # noqa: E402
from nerf_llm_scene_inspector.utils.shell import format_command, run_command  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", default="desk_scene", help="Name for the mock scene.")
    parser.add_argument(
        "--query",
        default="Find objects related to making coffee.",
        help="Natural-language query to run after mock training.",
    )
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--variant", choices=["lerf", "lerf-lite", "lerf-big"], default="lerf-lite")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    processed_dir = ROOT / "data" / "processed" / args.scene_name
    baseline_dir = ROOT / "runs" / f"baseline_{args.scene_name}"
    language_dir = ROOT / "runs" / f"language_{args.scene_name}"
    query_dir = ROOT / "results" / "query_outputs"

    steps: list[dict[str, object]] = []
    try:
        steps.append(
            {
                "name": "prepare_data",
                "result": prepare_data(ROOT / "examples", processed_dir, "images", dry_run=True),
            }
        )
        steps.append(
            {
                "name": "train_baseline_nerf",
                "result": train_baseline_nerf(
                    processed_dir,
                    "nerfacto",
                    baseline_dir,
                    dry_run=True,
                ),
            }
        )
        language_summary = train_language_field(
            processed_dir,
            args.backend,
            args.variant,
            language_dir,
            dry_run=True,
        )
        steps.append({"name": "train_language_field", "result": language_summary})

        query_command = [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(language_dir / "config.yml"),
            "--backend",
            args.backend,
            "--query",
            args.query,
            "--output",
            str(query_dir),
            "--num-views",
            "2",
            "--dry-run",
        ]
        demo_command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_demo_assets.py"),
            "--config",
            str(language_dir / "config.yml"),
            "--backend",
            args.backend,
            "--num-views",
            "1",
            "--dry-run",
        ]
        eval_command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_queries.py"),
            "--queries",
            str(ROOT / "examples" / "queries_demo.yaml"),
            "--annotations",
            str(ROOT / "examples" / "annotations_example.json"),
            "--results",
            str(ROOT / "results" / "demo_assets"),
            "--dry-run",
        ]
        for name, command in [
            ("query_scene", query_command),
            ("generate_demo_assets", demo_command),
            ("evaluate_queries", eval_command),
        ]:
            result = run_command(command, cwd=ROOT, check=True)
            steps.append({"name": name, "command": format_command(command), "result": result.to_dict()})
    except Exception as exc:
        print(f"dry-run demo failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "scene_name": args.scene_name,
        "backend": args.backend,
        "query": args.query,
        "outputs": {
            "query_report": str(query_dir / "scene_query_report.json"),
            "demo_gif": str(ROOT / "results" / "demo_assets" / "demo_montage.gif"),
            "query_grid": str(ROOT / "results" / "demo_assets" / "query_grid.png"),
            "project_report": str(ROOT / "docs" / "project_report.md"),
            "portfolio_card": str(ROOT / "docs" / "portfolio_result_card.md"),
            "eval_summary": str(ROOT / "results" / "evaluation" / "eval_summary.json"),
            "qualitative_report": str(ROOT / "results" / "evaluation" / "qualitative_report.md"),
        },
        "steps": steps,
    }
    summary_path = ROOT / "results" / "dry_run_demo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
