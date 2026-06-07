#!/usr/bin/env python
"""Evaluate semantic query outputs against lightweight manual annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult  # noqa: E402
from nerf_llm_scene_inspector.config import load_mapping  # noqa: E402
from nerf_llm_scene_inspector.evaluation.annotation_schema import load_annotations  # noqa: E402
from nerf_llm_scene_inspector.evaluation.metrics import (  # noqa: E402
    average_relevancy_score,
    qualitative_success_table,
    semantic_query_success_rate,
)
from nerf_llm_scene_inspector.evaluation.report import write_project_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="YAML file containing demo queries.")
    parser.add_argument("--annotations", required=True, help="Annotation JSON file.")
    parser.add_argument("--results", required=True, help="Directory containing query_result.json files.")
    parser.add_argument("--output", default="results/evaluation", help="Evaluation output directory.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Create synthetic predictions from annotations.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        query_file = load_mapping(args.queries)
        annotations = load_annotations(args.annotations)
        results = _load_results(Path(args.results))
        if args.dry_run and not results:
            results = _synthetic_results_from_annotations(annotations)
        annotation_dict = {
            item.query: item.to_dict()
            for item in annotations.queries
            if not query_file.get("queries") or item.query in query_file.get("queries", [])
        }
        rows = qualitative_success_table(results, annotation_dict, k=args.top_k)
        successes = [bool(row["topk_hit"]) for row in rows]
        metrics = {
            "top_k_hit_rate": semantic_query_success_rate(successes),
            "mean_iou_2d": _mean([float(row["best_iou_2d"]) for row in rows]),
            "semantic_success_rate": semantic_query_success_rate(successes),
            "average_relevancy_score": average_relevancy_score(results),
            "num_evaluated_queries": len(rows),
        }
        summary_path = output / "eval_summary.json"
        table_path = output / "eval_table.csv"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_csv(table_path, rows)
        write_project_report(
            ROOT / "docs" / "project_report.md",
            title="NeRF-LLM Scene Inspector Report",
            scene_name=annotations.scene_name,
            backend="lerf/opennerf",
            query_rows=rows,
            metrics=metrics,
            notes=[
                "Metrics are lightweight portfolio metrics and depend on manual annotations.",
                "Dry-run results are synthetic and only validate the evaluation pipeline.",
            ],
        )
    except Exception as exc:
        print(f"evaluate_queries failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")
    return 0


def _load_results(results_dir: Path) -> list[QueryResult]:
    if not results_dir.exists():
        return []
    return [QueryResult.from_json(path) for path in results_dir.rglob("query_result.json")]


def _synthetic_results_from_annotations(annotations) -> list[QueryResult]:
    results: list[QueryResult] = []
    for annotation in annotations.queries:
        regions = []
        if annotation.bbox_2d:
            regions.append(
                BoundingRegion(
                    label=annotation.query,
                    score=0.9,
                    coordinate_frame="image",
                    bbox_2d=annotation.bbox_2d,
                    source_view=annotation.acceptable_views[0] if annotation.acceptable_views else "view_0000.png",
                    notes="Synthetic dry-run prediction copied from annotation.",
                )
            )
        results.append(
            QueryResult(
                query=annotation.query,
                backend_name="dry-run",
                config_path="dry-run",
                bounding_regions=regions,
                confidence=0.9 if regions else 0.0,
                warnings=["Synthetic dry-run result."],
            )
        )
    return results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query", "target_description", "topk_hit", "best_iou_2d", "confidence", "num_regions", "warnings"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
