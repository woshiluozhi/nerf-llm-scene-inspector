"""Merge offline workbench annotation exports back into evaluation annotations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.evaluation.annotation_validation import validate_annotations
from nerf_llm_scene_inspector.utils.paths import utc_timestamp


@dataclass
class AnnotationMergeChange:
    """One query changed by an annotation merge."""

    query: str
    fields_changed: list[str] = field(default_factory=list)
    bbox_2d: list[float] | None = None
    acceptable_views: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AnnotationMergeReport:
    """Structured report for a workbench annotation merge."""

    ok: bool
    scene_name: str
    template_path: str
    filled_path: str
    output_path: str
    generated_at: str
    query_count: int
    updated_count: int
    bbox_annotation_count: int
    missing_filled_queries: list[str] = field(default_factory=list)
    extra_filled_queries: list[str] = field(default_factory=list)
    duplicate_filled_queries: list[str] = field(default_factory=list)
    invalid_bboxes: list[dict[str, object]] = field(default_factory=list)
    changes: list[AnnotationMergeChange] = field(default_factory=list)
    validation: dict[str, object] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["changes"] = [change.to_dict() for change in self.changes]
        return payload

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path


def merge_workbench_annotations(
    *,
    template_path: str | Path,
    filled_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    queries_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    overwrite: bool = False,
) -> AnnotationMergeReport:
    """Merge a downloaded workbench JSON file into a clean annotation schema."""

    template_file = Path(template_path)
    filled_file = Path(filled_path)
    output_file = Path(output_path)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing merged annotations: {output_file}")

    template = _load_annotation_object(template_file, label="template")
    filled = _load_annotation_object(filled_file, label="filled annotations")
    scene_name = str(filled.get("scene_name") or template.get("scene_name") or "scene")
    filled_by_query, duplicate_filled = _indexed_queries(filled.get("queries") or [])
    template_queries = [item for item in template.get("queries") or [] if isinstance(item, dict)]

    merged_queries: list[dict[str, object]] = []
    changes: list[AnnotationMergeChange] = []
    invalid_bboxes: list[dict[str, object]] = []
    missing_filled: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    updated_count = 0
    bbox_count = 0

    template_keys: set[str] = set()
    for raw in template_queries:
        query = str(raw.get("query") or "").strip()
        if not query:
            continue
        key = _normalize_query(query)
        template_keys.add(key)
        filled_item = filled_by_query.get(key)
        if filled_item is None:
            missing_filled.append(query)
        merged, change, invalid_bbox = _merge_query(raw, filled_item)
        if invalid_bbox:
            invalid_bboxes.append(invalid_bbox)
        if change.fields_changed:
            updated_count += 1
            changes.append(change)
        if merged.get("bbox_2d") is not None:
            bbox_count += 1
        merged_queries.append(merged)

    extra_filled = [
        str(item.get("query") or "").strip()
        for key, item in filled_by_query.items()
        if key not in template_keys and str(item.get("query") or "").strip()
    ]
    if missing_filled:
        warnings.append("Filled annotations are missing template queries: " + ", ".join(missing_filled))
    if extra_filled:
        warnings.append("Filled annotations contain queries not present in the template: " + ", ".join(extra_filled))
    if duplicate_filled:
        errors.append("Filled annotations contain duplicate queries: " + ", ".join(duplicate_filled))
    if invalid_bboxes:
        errors.append(f"{len(invalid_bboxes)} filled bbox_2d entries are invalid.")

    payload = {
        "scene_name": scene_name,
        "created_at": utc_timestamp(),
        "source_template": _display_path(template_file),
        "source_filled_annotations": _display_path(filled_file),
        "instructions": [
            "Merged from an offline annotation workbench export.",
            "Use validate_annotations.py before reporting localization metrics.",
        ],
        "queries": merged_queries,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    validation: dict[str, object] | None = None
    if queries_path or results_dir:
        validation_report = validate_annotations(output_file, queries_path=queries_path, results_dir=results_dir)
        validation = validation_report.to_dict()
        if not validation_report.ok:
            errors.extend(str(item) for item in validation_report.errors)
        warnings.extend(str(item) for item in validation_report.warnings)

    report = AnnotationMergeReport(
        ok=not errors,
        scene_name=scene_name,
        template_path=_display_path(template_file),
        filled_path=_display_path(filled_file),
        output_path=_display_path(output_file),
        generated_at=utc_timestamp(),
        query_count=len(merged_queries),
        updated_count=updated_count,
        bbox_annotation_count=bbox_count,
        missing_filled_queries=missing_filled,
        extra_filled_queries=extra_filled,
        duplicate_filled_queries=duplicate_filled,
        invalid_bboxes=invalid_bboxes,
        changes=changes,
        validation=validation,
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
    )
    if report_path:
        report.to_json(report_path)
    return report


def _merge_query(
    template_item: dict[str, Any],
    filled_item: dict[str, Any] | None,
) -> tuple[dict[str, object], AnnotationMergeChange, dict[str, object] | None]:
    query = str(template_item.get("query") or "").strip()
    source = filled_item or {}
    invalid_bbox: dict[str, object] | None = None
    fields_changed: list[str] = []

    target_description = _first_text(source.get("target_description"), template_item.get("target_description"))
    acceptable_views = _first_string_list(source.get("acceptable_views"), template_item.get("acceptable_views"))
    notes = _first_text(source.get("notes"), template_item.get("notes"))
    bbox, bbox_error = _bbox_list(source.get("bbox_2d") if filled_item is not None else template_item.get("bbox_2d"))
    if bbox_error and filled_item is not None and source.get("bbox_2d") is not None:
        invalid_bbox = {"query": query, "bbox_2d": source.get("bbox_2d"), "error": bbox_error}
        bbox = _bbox_list(template_item.get("bbox_2d"))[0]

    comparisons = {
        "target_description": target_description,
        "acceptable_views": acceptable_views,
        "bbox_2d": bbox,
        "notes": notes,
    }
    for field_name, value in comparisons.items():
        if _normalized_value(template_item.get(field_name)) != _normalized_value(value):
            fields_changed.append(field_name)

    merged = {
        "query": query,
        "target_description": target_description,
        "acceptable_views": acceptable_views,
        "bbox_2d": bbox,
        "notes": notes,
    }
    return (
        merged,
        AnnotationMergeChange(
            query=query,
            fields_changed=fields_changed,
            bbox_2d=bbox,
            acceptable_views=acceptable_views,
        ),
        invalid_bbox,
    )


def _load_annotation_object(path: Path, *, label: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} JSON must be an object.")
    if not isinstance(raw.get("queries"), list):
        raise ValueError(f"{label} JSON must contain a queries list.")
    return raw


def _indexed_queries(raw_queries: list[object]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        key = _normalize_query(query)
        if key in indexed:
            duplicates.append(query)
            continue
        indexed[key] = item
    return indexed, sorted(set(duplicates))


def _bbox_list(value: object) -> tuple[list[float] | None, str]:
    if value is None:
        return None, ""
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None, "bbox_2d must be a list of four numbers [x1, y1, x2, y2]."
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None, "bbox_2d values must be numeric."
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None, "bbox_2d values must be finite."
    if min(x1, y1, x2, y2) < 0:
        return None, "bbox_2d values must be non-negative pixel coordinates."
    if x2 <= x1 or y2 <= y1:
        return None, "bbox_2d must satisfy x2 > x1 and y2 > y1."
    return [x1, y1, x2, y2], ""


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_string_list(*values: object) -> list[str]:
    for value in values:
        if not isinstance(value, list):
            continue
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return _dedupe(items)
    return []


def _normalize_query(value: str) -> str:
    return " ".join(value.lower().split())


def _normalized_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_normalized_value(item) for item in value]
    if isinstance(value, list):
        return [_normalized_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, int):
        return float(value)
    if value is None:
        return None
    return str(value).strip()


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")
