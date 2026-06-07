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
from nerf_llm_scene_inspector.querying.spatial_reasoning import aggregate_multi_query_results  # noqa: E402
from nerf_llm_scene_inspector.utils.paths import slugify  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Trained Nerfstudio/LERF config.yml.")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--query", required=True, help="Text query or high-level scene task.")
    parser.add_argument("--output", required=True, help="Output directory for query artifacts.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prefer-llm", action="store_true", help="Use optional LLM planner when configured.")
    parser.add_argument("--exact-query", action="store_true", help="Run the query text directly without expansion.")
    parser.add_argument("--dry-run", action="store_true", help="Create mock query artifacts.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    backend = LERFBackend(dry_run=args.dry_run) if args.backend == "lerf" else OpenNeRFBackend(dry_run=args.dry_run)
    try:
        backend.load(args.config)
        planner = get_default_planner(prefer_llm=args.prefer_llm)
        plan = planner.plan(args.query)
        queries = [args.query] if args.exact_query else plan.primary_visual_queries or [args.query]
        results = []
        for query in queries[: args.top_k]:
            query_dir = output / slugify(query)
            results.append(backend.query_text(query, str(query_dir), top_k=args.top_k))
        aggregate = aggregate_multi_query_results(results)
        items = [region.label for region in aggregate.bounding_regions[: args.top_k]]
        if not items:
            items = [result.query for result in results]
        answer = plan.final_answer_template.format(items=", ".join(items))
        report = SceneQueryReport(
            scene_name="unknown",
            task=args.query,
            plan=plan.to_dict(),
            query_results=results,
            answer=answer,
            warnings=plan.warnings + aggregate.warnings,
        )
        report_path = report.to_json(output / "scene_query_report.json")
    except Exception as exc:
        print(f"query_scene failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
