#!/usr/bin/env python
"""Run natural-language semantic scene queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.agent.planner import get_default_planner  # noqa: E402
from nerf_llm_scene_inspector.backends.base import SceneQueryReport  # noqa: E402
from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend  # noqa: E402
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend  # noqa: E402
from nerf_llm_scene_inspector.querying.answer_synthesis import synthesize_scene_answer  # noqa: E402
from nerf_llm_scene_inspector.querying.semantic_query import planned_backend_calls  # noqa: E402
from nerf_llm_scene_inspector.querying.spatial_reasoning import aggregate_multi_query_results  # noqa: E402
from nerf_llm_scene_inspector.utils.paths import slugify  # noqa: E402
from nerf_llm_scene_inspector.visualization.make_video import make_mp4_or_gif, make_query_grid  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Trained Nerfstudio/LERF config.yml.")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--query", required=True, help="Text query or high-level scene task.")
    parser.add_argument("--output", required=True, help="Output directory for query artifacts.")
    parser.add_argument("--scene-name", default="unknown", help="Scene name stored in the query report.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help="Maximum expanded backend text queries to run for one high-level task.",
    )
    parser.add_argument(
        "--include-negative-queries",
        action="store_true",
        help="Also run planner negative/disambiguation prompts. They are excluded by default.",
    )
    parser.add_argument("--prefer-llm", action="store_true", help="Use optional LLM planner when configured.")
    parser.add_argument("--exact-query", action="store_true", help="Run the query text directly without expansion.")
    parser.add_argument("--dry-run", action="store_true", help="Create mock query artifacts.")
    parser.add_argument("--num-views", type=int, default=1, help="Number of camera views to render per query.")
    parser.add_argument(
        "--render-output-names",
        default="rgb,relevancy_0,composited_0",
        help="Comma-separated backend render outputs to request/save.",
    )
    parser.add_argument(
        "--save-manual-template",
        action="store_true",
        help="Write a manual QueryResult template when backend rendering falls back.",
    )
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help="Fail instead of writing viewer fallback artifacts when automated rendering fails.",
    )
    parser.add_argument(
        "--no-query-grid",
        action="store_true",
        help="Skip writing output/query_grid.png from rendered overlay images.",
    )
    parser.add_argument(
        "--make-montage",
        action="store_true",
        help="Also write output/query_montage.gif from rendered overlay images.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    render_output_names = [
        name.strip() for name in args.render_output_names.split(",") if name.strip()
    ]
    backend = (
        LERFBackend(
            dry_run=args.dry_run,
            num_views=args.num_views,
            render_output_names=render_output_names,
            save_manual_template=args.save_manual_template,
            strict_backend=args.strict_backend,
        )
        if args.backend == "lerf"
        else OpenNeRFBackend(
            dry_run=args.dry_run,
            num_views=args.num_views,
            save_manual_template=args.save_manual_template,
            strict_backend=args.strict_backend,
        )
    )
    try:
        backend.load(args.config)
        planner = get_default_planner(prefer_llm=args.prefer_llm)
        plan = planner.plan(args.query)
        calls = planned_backend_calls(
            plan,
            task=args.query,
            exact_query=args.exact_query,
            include_negative=args.include_negative_queries,
            max_queries=args.max_queries,
        )
        results = []
        for call in calls:
            query = call.query
            query_dir = output / slugify(query)
            result = backend.query_text(query, str(query_dir), top_k=args.top_k)
            result.provenance["planner_backend_call"] = call.to_dict()
            result.to_json(query_dir / "query_result.json")
            results.append(result)
        aggregate = aggregate_multi_query_results(results)
        answer = synthesize_scene_answer(
            task=args.query,
            plan=plan.to_dict(),
            results=results,
            top_k=args.top_k,
        )
        report = SceneQueryReport(
            scene_name=args.scene_name,
            task=args.query,
            plan=plan.to_dict(),
            query_results=results,
            answer=answer.answer,
            answer_summary=answer.to_dict(),
            warnings=plan.warnings + aggregate.warnings,
        )
        report_path = report.to_json(output / "scene_query_report.json")
        report_md_path = report.to_markdown(output / "scene_query_report.md")
        overlay_paths = [
            Path(view.path)
            for result in results
            for view in result.rendered_images
            if view.kind == "overlay"
        ]
        grid_path = None
        if not args.no_query_grid:
            grid_path = make_query_grid(overlay_paths, output / "query_grid.png")
        montage_path = None
        if args.make_montage and overlay_paths:
            montage_path = make_mp4_or_gif(overlay_paths, output / "query_montage.gif")
        visual_summary_path = output / "query_visual_summary.json"
        visual_summary_path.write_text(
            json.dumps(
                {
                    "scene_name": args.scene_name,
                    "task": args.query,
                    "backend": args.backend,
                    "num_overlay_images": len([path for path in overlay_paths if path.exists()]),
                    "query_grid": _relative_to_output(grid_path, output) if grid_path else None,
                    "query_montage": _relative_to_output(montage_path, output) if montage_path else None,
                    "expanded_queries": [result.query for result in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"query_scene failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote report: {report_path}")
    print(f"Wrote markdown report: {report_md_path}")
    print(f"Wrote visual summary: {visual_summary_path}")
    if grid_path:
        print(f"Wrote query grid: {grid_path}")
    if montage_path:
        print(f"Wrote query montage: {montage_path}")
    return 0


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
