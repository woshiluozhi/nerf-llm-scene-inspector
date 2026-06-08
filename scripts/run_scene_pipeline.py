#!/usr/bin/env python
"""Run the practical end-to-end scene inspection pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.config import load_mapping  # noqa: E402
from nerf_llm_scene_inspector.pipeline import (  # noqa: E402
    DEFAULT_PIPELINE_QUERIES,
    PipelineConfig,
    run_scene_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="examples", help="Raw video file or image directory.")
    parser.add_argument("--scene-name", default="desk_scene")
    parser.add_argument("--type", choices=["video", "images"], default="images")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--variant", choices=["lerf", "lerf-lite", "lerf-big"], default="lerf-lite")
    parser.add_argument("--baseline-method", default="nerfacto")
    parser.add_argument("--query", action="append", help="Query to run. Can be repeated.")
    parser.add_argument("--queries-file", help="YAML file with queries and/or tasks.")
    parser.add_argument("--config", help="Existing trained config.yml, useful with --skip-language.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-root", default="results/pipeline_runs")
    parser.add_argument("--annotations", default="examples/annotations_example.json")
    parser.add_argument(
        "--prompt-suite",
        help="Optional YAML prompt suite for robustness analysis; suite prompts are added to queries.",
    )
    parser.add_argument("--capture-manifest", help="Optional capture_manifest.json to copy into the run.")
    parser.add_argument("--max-num-iterations", type=int)
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--min-pose-extent", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail on environment/data readiness issues.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-language", action="store_true")
    parser.add_argument("--skip-queries", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--no-clean-run",
        action="store_true",
        help="Keep existing query/demo/evaluation files under the pipeline run directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    queries = _collect_queries(args.query, args.queries_file)
    config = PipelineConfig(
        input_path=args.input,
        scene_name=args.scene_name,
        data_type=args.type,
        backend=args.backend,
        variant=args.variant,
        baseline_method=args.baseline_method,
        queries=queries,
        data_root=args.data_root,
        runs_root=args.runs_root,
        output_root=args.output_root,
        annotations_path=args.annotations,
        prompt_suite_path=args.prompt_suite,
        capture_manifest_path=args.capture_manifest,
        config_path=args.config,
        max_num_iterations=args.max_num_iterations,
        num_views=args.num_views,
        top_k=args.top_k,
        min_frames=args.min_frames,
        min_pose_extent=args.min_pose_extent,
        dry_run=args.dry_run,
        strict=args.strict,
        skip_prepare=args.skip_prepare,
        skip_baseline=args.skip_baseline,
        skip_language=args.skip_language,
        skip_queries=args.skip_queries,
        skip_demo=args.skip_demo,
        skip_eval=args.skip_eval,
        clean_run_outputs=not args.no_clean_run,
        command=list(sys.argv),
    )
    summary = run_scene_pipeline(config)
    print(json.dumps(summary.to_dict(), indent=2))
    summary_path = Path(args.output_root) / args.scene_name / "pipeline_summary.json"
    print(f"\nWrote {summary_path}")
    return 0 if summary.success else 1


def _collect_queries(cli_queries: list[str] | None, queries_file: str | None) -> list[str]:
    queries: list[str] = []
    if queries_file:
        raw = load_mapping(queries_file)
        for key in ("queries", "tasks"):
            values = raw.get(key) or []
            if isinstance(values, list):
                queries.extend(str(item) for item in values if str(item).strip())
    if cli_queries:
        queries.extend(item for item in cli_queries if item.strip())
    seen: set[str] = set()
    deduped = []
    for query in queries or DEFAULT_PIPELINE_QUERIES:
        normalized = query.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            deduped.append(normalized)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
