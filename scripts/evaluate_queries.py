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
from nerf_llm_scene_inspector.evaluation.annotation_validation import validate_annotations  # noqa: E402
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
    parser.add_argument("--report-output", default="docs/project_report.md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Create synthetic predictions from annotations.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        validation = validate_annotations(
            args.annotations,
            queries_path=args.queries,
            results_dir=args.results,
        )
        validation_path = output / "annotation_validation.json"
        validation.to_json(validation_path)
        if not validation.ok:
            raise RuntimeError(
                "Annotation validation failed. See "
                f"{validation_path}: {'; '.join(validation.errors)}"
            )
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
        evaluated_rows = [row for row in rows if row.get("evaluation_status") == "evaluated"]
        metric_rows = _best_rows_by_query(evaluated_rows)
        successes = [bool(row["topk_hit"]) for row in metric_rows]
        metrics = {
            "top_k_hit_rate": semantic_query_success_rate(successes) if metric_rows else None,
            "mean_iou_2d": _mean_or_none([float(row["best_iou_2d"]) for row in metric_rows]),
            "semantic_success_rate": semantic_query_success_rate(successes) if metric_rows else None,
            "average_relevancy_score": average_relevancy_score(results),
            "num_evaluated_queries": len(metric_rows),
            "num_result_queries": len(rows),
            "num_unique_result_queries": len({str(row.get("query", "")) for row in rows}),
            "num_annotated_queries": len(
                {str(row.get("query", "")) for row in rows if row.get("annotation_available")}
            ),
            "num_bbox_annotated_queries": len(metric_rows),
            "num_qualitative_only_queries": len(
                {str(row.get("query", "")) for row in rows if row.get("evaluation_status") != "evaluated"}
            ),
        }
        summary_path = output / "eval_summary.json"
        table_path = output / "eval_table.csv"
        qualitative_path = output / "qualitative_report.md"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_csv(table_path, rows)
        _write_qualitative_report(qualitative_path, annotations.scene_name, rows, metrics)
        write_project_report(
            args.report_output,
            title="NeRF-LLM Scene Inspector Report",
            scene_name=annotations.scene_name,
            backend="lerf/opennerf",
            query_rows=rows,
            metrics=metrics,
            notes=[
                "Metrics are lightweight portfolio metrics and depend on manual annotations.",
                "Dry-run results are synthetic and only validate the evaluation pipeline.",
                *validation.warnings,
            ],
        )
    except Exception as exc:
        print(f"evaluate_queries failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {validation_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {qualitative_path}")
    return 0


def _load_results(results_dir: Path) -> list[QueryResult]:
    if not results_dir.exists():
        return []
    return [QueryResult.from_json(path) for path in sorted(results_dir.rglob("query_result.json"))]


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
    fieldnames = [
        "query",
        "target_description",
        "evaluation_status",
        "annotation_available",
        "has_bbox_annotation",
        "topk_hit",
        "best_iou_2d",
        "confidence",
        "num_regions",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_qualitative_report(
    path: Path,
    scene_name: str,
    rows: list[dict[str, object]],
    metrics: dict[str, object],
) -> None:
    lines = [
        f"# Qualitative Evaluation Report: {scene_name}",
        "",
        "This report summarizes lightweight query evaluation against manual annotations.",
        "Dry-run outputs validate pipeline wiring only; real scores require trained semantic fields.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in metrics.items())
    lines.extend(
        [
            "",
            "## Query Table",
            "",
            "| Query | Target | Status | Top-k Hit | Best IoU | Confidence | Warnings |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {query} | {target} | {status} | {hit} | {iou} | {confidence} | {warnings} |".format(
                query=row.get("query", ""),
                target=row.get("target_description", ""),
                status=row.get("evaluation_status", ""),
                hit=_display_value(row.get("topk_hit", "")),
                iou=_display_iou(row.get("best_iou_2d")),
                confidence=row.get("confidence", ""),
                warnings=str(row.get("warnings", "")).replace("|", "/"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _best_rows_by_query(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        query = str(row.get("query", ""))
        if query not in best or _metric_row_score(row) > _metric_row_score(best[query]):
            best[query] = row
    return list(best.values())


def _metric_row_score(row: dict[str, object]) -> tuple[int, float, float]:
    hit_score = 1 if row.get("topk_hit") is True else 0
    iou_score = float(row.get("best_iou_2d") or 0.0)
    confidence = float(row.get("confidence") or 0.0)
    return hit_score, iou_score, confidence


def _display_value(value: object) -> str:
    return "n/a" if value in {"", None} else str(value)


def _display_iou(value: object) -> str:
    if value in {"", None}:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
