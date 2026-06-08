#!/usr/bin/env python
"""Create a manual annotation template for semantic query evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.backends.base import QueryResult  # noqa: E402
from nerf_llm_scene_inspector.config import load_mapping  # noqa: E402
from nerf_llm_scene_inspector.utils.paths import utc_timestamp  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="YAML file with queries and optional tasks.")
    parser.add_argument("--scene-name", help="Scene name. Defaults to scene_name in the query file.")
    parser.add_argument(
        "--results",
        help="Optional query output directory. Used to collect candidate views and suggested boxes.",
    )
    parser.add_argument("--output", default="results/annotations_template.json")
    parser.add_argument("--include-tasks", action="store_true", help="Also include high-level tasks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing annotation template: {output}", file=sys.stderr)
        return 1
    payload = build_annotation_template(
        queries_path=args.queries,
        scene_name=args.scene_name,
        results_dir=args.results,
        include_tasks=args.include_tasks,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {output}")
    return 0


def build_annotation_template(
    *,
    queries_path: str | Path,
    scene_name: str | None = None,
    results_dir: str | Path | None = None,
    include_tasks: bool = False,
) -> dict[str, Any]:
    query_file = load_mapping(queries_path)
    queries = _collect_queries(query_file, include_tasks=include_tasks)
    result_by_query = _load_query_results(Path(results_dir)) if results_dir else {}
    return {
        "scene_name": scene_name or str(query_file.get("scene_name") or "scene"),
        "created_at": utc_timestamp(),
        "instructions": [
            "Fill target_description with the intended object or region.",
            "Set acceptable_views to camera ids such as view_0000 after inspecting overlays.",
            "Fill bbox_2d as [x1, y1, x2, y2] in the selected source view.",
            "Leave bbox_2d as null for qualitative-only queries.",
            "candidate_* fields are suggestions from the model output, not ground truth.",
        ],
        "queries": [_annotation_entry(query, result_by_query.get(query)) for query in queries],
    }


def _collect_queries(query_file: dict[str, Any], *, include_tasks: bool) -> list[str]:
    values: list[str] = []
    for query in query_file.get("queries") or []:
        if str(query).strip():
            values.append(str(query).strip())
    if include_tasks:
        for task in query_file.get("tasks") or []:
            if str(task).strip():
                values.append(str(task).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _load_query_results(results_dir: Path) -> dict[str, QueryResult]:
    if not results_dir.exists():
        return {}
    results: dict[str, QueryResult] = {}
    for path in sorted(results_dir.rglob("query_result.json")):
        try:
            result = QueryResult.from_json(path)
        except Exception:
            continue
        results.setdefault(result.query, result)
    return results


def _annotation_entry(query: str, result: QueryResult | None) -> dict[str, Any]:
    candidate_views = _candidate_views(result)
    suggested_boxes = _suggested_boxes(result)
    return {
        "query": query,
        "target_description": "",
        "acceptable_views": candidate_views[:3],
        "bbox_2d": None,
        "notes": "TODO: replace candidate suggestions with manual annotation.",
        "candidate_views": candidate_views,
        "candidate_bbox_2d_suggestions": suggested_boxes,
    }


def _candidate_views(result: QueryResult | None) -> list[str]:
    if result is None:
        return []
    views: list[str] = []
    for region in result.bounding_regions:
        if region.source_view:
            views.append(region.source_view)
    for view in result.rendered_images:
        if view.camera_id:
            views.append(view.camera_id)
    return _dedupe(views)


def _suggested_boxes(result: QueryResult | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    boxes: list[dict[str, Any]] = []
    for region in result.bounding_regions:
        if region.bbox_2d is None:
            continue
        boxes.append(
            {
                "source_view": region.source_view,
                "bbox_2d": list(region.bbox_2d),
                "score": region.score,
                "notes": region.notes,
            }
        )
    return boxes[:5]


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
