#!/usr/bin/env python
"""Generate portfolio demo assets from example queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend  # noqa: E402
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend  # noqa: E402
from nerf_llm_scene_inspector.config import load_mapping  # noqa: E402
from nerf_llm_scene_inspector.evaluation.report import write_project_report  # noqa: E402
from nerf_llm_scene_inspector.utils.paths import slugify  # noqa: E402
from nerf_llm_scene_inspector.visualization.make_video import make_mp4_or_gif  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/language_desk_scene/config.yml")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--queries", default="examples/queries_demo.yaml")
    parser.add_argument("--output", default="results/demo_assets")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_queries = load_mapping(args.queries)
    scene_name = str(raw_queries.get("scene_name", "desk_scene"))
    queries = [str(item) for item in raw_queries.get("queries", [])]
    if not queries:
        print("No queries found in query file.", file=sys.stderr)
        return 1
    backend = LERFBackend(dry_run=args.dry_run) if args.backend == "lerf" else OpenNeRFBackend(dry_run=args.dry_run)
    try:
        backend.load(args.config)
        results = []
        overlay_paths: list[Path] = []
        for query in queries:
            result = backend.query_text(query, str(output / slugify(query)), top_k=5)
            results.append(result)
            overlay_paths.extend(Path(view.path) for view in result.rendered_images if view.kind == "overlay")
        video_path = None
        if overlay_paths:
            video_path = make_mp4_or_gif(overlay_paths, output / "demo_montage.gif")
        query_rows = [
            {
                "query": result.query,
                "target_description": "demo query",
                "topk_hit": "qualitative",
                "best_iou_2d": 0.0,
                "confidence": result.confidence if result.confidence is not None else "",
                "warnings": "; ".join(result.warnings),
            }
            for result in results
        ]
        write_project_report(
            ROOT / "docs" / "project_report.md",
            title="NeRF-LLM Scene Inspector Report",
            scene_name=scene_name,
            backend=args.backend,
            query_rows=query_rows,
            metrics={"num_queries": len(results), "demo_video": str(video_path) if video_path else "not generated"},
            notes=[
                "Demo assets may be dry-run synthetic artifacts unless generated from a trained LERF config.",
                "This report is portfolio-ready but does not claim state-of-the-art performance.",
            ],
        )
    except Exception as exc:
        print(f"generate_demo_assets failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "scene_name": scene_name,
        "backend": args.backend,
        "num_queries": len(results),
        "output": str(output),
        "video": str(video_path) if video_path else None,
        "results": [result.to_dict() for result in results],
    }
    summary_path = output / "demo_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
