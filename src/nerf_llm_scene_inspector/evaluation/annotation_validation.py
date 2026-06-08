"""Validation helpers for manual query annotations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.backends.base import QueryResult
from nerf_llm_scene_inspector.config import load_mapping


@dataclass
class AnnotationValidationReport:
    """Structured annotation quality report."""

    ok: bool
    scene_name: str
    annotation_count: int
    expected_query_count: int = 0
    bbox_annotation_count: int = 0
    missing_annotations: list[str] = field(default_factory=list)
    extra_annotations: list[str] = field(default_factory=list)
    duplicate_annotations: list[str] = field(default_factory=list)
    missing_result_queries: list[str] = field(default_factory=list)
    invalid_bboxes: list[dict[str, object]] = field(default_factory=list)
    unknown_views: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path


def validate_annotations(
    annotations_path: str | Path,
    *,
    queries_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> AnnotationValidationReport:
    """Validate annotation structure and optional query/result consistency."""

    raw = _load_json(annotations_path)
    if not isinstance(raw, dict):
        return AnnotationValidationReport(
            ok=False,
            scene_name="",
            annotation_count=0,
            errors=["Annotation JSON must be an object with scene_name and queries fields."],
        )
    scene_name = str(raw.get("scene_name") or "")
    raw_queries = raw.get("queries")
    errors: list[str] = []
    warnings: list[str] = []
    if not scene_name:
        errors.append("Missing required top-level scene_name.")
    if not isinstance(raw_queries, list):
        errors.append("Top-level queries must be a list.")
        raw_queries = []

    expected_scene_name, expected_queries = _load_expected_query_file(queries_path)
    if scene_name and expected_scene_name and scene_name != expected_scene_name:
        warnings.append(f"Annotation scene_name '{scene_name}' does not match query scene_name '{expected_scene_name}'.")
    result_views = _load_result_views(Path(results_dir)) if results_dir else {}
    result_views_by_normalized_query = {query.lower(): views for query, views in result_views.items()}
    seen: set[str] = set()
    duplicate_annotations: list[str] = []
    annotation_names: list[str] = []
    missing_result_queries: list[str] = []
    invalid_bboxes: list[dict[str, object]] = []
    unknown_views: list[dict[str, object]] = []
    bbox_annotation_count = 0

    for index, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            errors.append(f"Annotation at index {index} must be an object.")
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            errors.append(f"Annotation at index {index} is missing query.")
            continue
        annotation_names.append(query)
        normalized = query.lower()
        if normalized in seen:
            duplicate_annotations.append(query)
        seen.add(normalized)

        bbox = item.get("bbox_2d")
        if bbox is not None:
            bbox_annotation_count += 1
            bbox_error = _bbox_error(bbox)
            if bbox_error:
                invalid_bboxes.append({"query": query, "bbox_2d": bbox, "error": bbox_error})

        acceptable_views = item.get("acceptable_views") or []
        if not isinstance(acceptable_views, list):
            errors.append(f"Annotation '{query}' acceptable_views must be a list.")
            acceptable_views = []
        known_views = result_views_by_normalized_query.get(query.lower())
        if results_dir and known_views is None:
            missing_result_queries.append(query)
        elif known_views:
            unknown = [
                str(view)
                for view in acceptable_views
                if str(view) not in known_views and _strip_image_suffix(str(view)) not in known_views
            ]
            if unknown:
                unknown_views.append(
                    {
                        "query": query,
                        "unknown_views": unknown,
                        "known_views": sorted(known_views),
                    }
                )

    missing_annotations = [
        query for query in expected_queries if query.lower() not in {name.lower() for name in annotation_names}
    ]
    extra_annotations = [
        name for name in annotation_names if expected_queries and name.lower() not in {query.lower() for query in expected_queries}
    ]

    if duplicate_annotations:
        errors.append("Duplicate query annotations found: " + ", ".join(sorted(set(duplicate_annotations))))
    if invalid_bboxes:
        errors.append(f"{len(invalid_bboxes)} bbox_2d entries are invalid.")
    if missing_annotations:
        warnings.append("Missing annotations for queries: " + ", ".join(missing_annotations))
    if extra_annotations:
        warnings.append("Annotations not present in query file: " + ", ".join(extra_annotations))
    if missing_result_queries:
        warnings.append("Annotations without query results: " + ", ".join(missing_result_queries))
    if unknown_views:
        warnings.append(f"{len(unknown_views)} annotations reference views not found in query results.")

    return AnnotationValidationReport(
        ok=not errors,
        scene_name=scene_name,
        annotation_count=len(annotation_names),
        expected_query_count=len(expected_queries),
        bbox_annotation_count=bbox_annotation_count,
        missing_annotations=missing_annotations,
        extra_annotations=extra_annotations,
        duplicate_annotations=sorted(set(duplicate_annotations)),
        missing_result_queries=sorted(set(missing_result_queries)),
        invalid_bboxes=invalid_bboxes,
        unknown_views=unknown_views,
        warnings=warnings,
        errors=errors,
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_expected_query_file(queries_path: str | Path | None) -> tuple[str, list[str]]:
    if not queries_path:
        return "", []
    raw = load_mapping(queries_path)
    scene_name = str(raw.get("scene_name") or "")
    queries = [str(query).strip() for query in raw.get("queries") or [] if str(query).strip()]
    return scene_name, queries


def _load_result_views(results_dir: Path) -> dict[str, set[str]]:
    if not results_dir.exists():
        return {}
    views: dict[str, set[str]] = {}
    for path in sorted(results_dir.rglob("query_result.json")):
        try:
            result = QueryResult.from_json(path)
        except Exception:
            continue
        query_views = views.setdefault(result.query, set())
        for view in result.rendered_images:
            if view.camera_id:
                query_views.add(view.camera_id)
            if view.path:
                query_views.add(Path(view.path).name)
                query_views.add(Path(view.path).stem)
        for region in result.bounding_regions:
            if region.source_view:
                query_views.add(region.source_view)
                query_views.add(_strip_image_suffix(region.source_view))
    return views


def _bbox_error(raw_bbox: object) -> str:
    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        return "bbox_2d must be a list of four numbers [x1, y1, x2, y2]."
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return "bbox_2d values must be numeric."
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return "bbox_2d values must be finite."
    if min(x1, y1, x2, y2) < 0:
        return "bbox_2d values must be non-negative pixel coordinates."
    if x2 <= x1 or y2 <= y1:
        return "bbox_2d must satisfy x2 > x1 and y2 > y1."
    return ""


def _strip_image_suffix(view: str) -> str:
    path = Path(view)
    return path.stem if path.suffix else view
