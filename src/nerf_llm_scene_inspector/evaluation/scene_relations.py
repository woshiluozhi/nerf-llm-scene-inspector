"""Scene-level relation graph extraction from semantic query outputs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.backends.base import BoundingRegion, Candidate3DPoint, QueryResult
from nerf_llm_scene_inspector.evaluation.metrics import (
    bbox_center,
    bbox_center_distance,
    bbox_intersection_area,
    bbox_iou,
    containment_ratio,
)
from nerf_llm_scene_inspector.querying.spatial_reasoning import rank_candidate_regions
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp


@dataclass
class SceneEntity:
    """One object-like entity inferred from a semantic query result."""

    entity_id: str
    label: str
    query: str
    score: float | None
    evidence_type: str
    result_path: str
    source_view: str | None = None
    bbox_2d: tuple[float, float, float, float] | None = None
    center_2d: tuple[float, float] | None = None
    point_3d: tuple[float, float, float] | None = None
    coordinate_frame: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneRelation:
    """A directed relation edge between two inferred scene entities."""

    subject_id: str
    object_id: str
    subject_label: str
    object_label: str
    relation: str
    confidence: float
    evidence_type: str
    source_view: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneRelationReport:
    """Portable relation graph report for a pipeline query directory."""

    scene_name: str
    results_dir: str
    generated_at: str
    top_k_per_query: int
    entities: list[SceneEntity] = field(default_factory=list)
    relations: list[SceneRelation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "results_dir": self.results_dir,
            "generated_at": self.generated_at,
            "top_k_per_query": self.top_k_per_query,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "subject_id",
            "subject_label",
            "relation",
            "object_id",
            "object_label",
            "confidence",
            "evidence_type",
            "source_view",
            "notes",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for relation in self.relations:
                writer.writerow({key: relation.to_dict().get(key, "") for key in fieldnames})
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Scene Relation Report",
            "",
            f"- Scene: {self.scene_name}",
            f"- Results: `{self.results_dir}`",
            f"- Entities: {self.num_entities}",
            f"- Relations: {self.num_relations}",
            f"- Generated: {self.generated_at}",
            "",
            "## Relation Edges",
            "",
            *_relation_lines(self.relations),
            "",
            "## Entities",
            "",
            *_entity_lines(self.entities),
            "",
            "## Warnings",
            "",
            *_list_lines(self.warnings),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def analyze_scene_relations(
    *,
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    scene_name: str = "unknown",
    top_k_per_query: int = 1,
    max_edges: int = 100,
    near_threshold_px: float = 120.0,
    near_threshold_3d: float = 0.5,
    dry_run: bool = False,
) -> SceneRelationReport:
    """Create a scene relation graph from saved query_result.json files."""

    root = Path(results_dir)
    warnings: list[str] = []
    loaded = load_scene_entities(root, top_k_per_query=top_k_per_query)
    entities = loaded[0]
    warnings.extend(loaded[1])
    if dry_run and not entities:
        entities = _mock_entities()
        warnings.append("Dry-run relation graph uses synthetic entities because no query results were found.")
    if not entities:
        warnings.append("No entities were extracted. Run query_scene.py or run_scene_pipeline.py first.")

    relations = build_scene_relations(
        entities,
        max_edges=max_edges,
        near_threshold_px=near_threshold_px,
        near_threshold_3d=near_threshold_3d,
    )
    report = SceneRelationReport(
        scene_name=scene_name,
        results_dir=str(root),
        generated_at=utc_timestamp(),
        top_k_per_query=top_k_per_query,
        entities=entities,
        relations=relations,
        warnings=warnings,
    )
    if output_dir is not None:
        output = Path(output_dir)
        report.to_json(output / "scene_relations_summary.json")
        report.to_csv(output / "scene_relations_edges.csv")
        report.to_markdown(output / "scene_relations_report.md")
    return report


def load_scene_entities(
    results_dir: str | Path,
    *,
    top_k_per_query: int = 1,
) -> tuple[list[SceneEntity], list[str]]:
    """Load query artifacts and convert top regions/points into graph entities."""

    root = Path(results_dir)
    warnings: list[str] = []
    records = _load_query_results(root, warnings)
    entities: list[SceneEntity] = []
    used_ids: set[str] = set()
    for result, result_path in records:
        for index, region in enumerate(rank_candidate_regions(result.bounding_regions)[:top_k_per_query]):
            entity = _entity_from_region(result, result_path, region, index, used_ids)
            entities.append(entity)
        points = sorted(
            result.candidate_points,
            key=lambda point: point.score if point.score is not None else -1.0,
            reverse=True,
        )[:top_k_per_query]
        for index, point in enumerate(points):
            entity = _entity_from_point(result, result_path, point, index, used_ids)
            entities.append(entity)
    return _dedupe_entities(entities), warnings


def build_scene_relations(
    entities: list[SceneEntity],
    *,
    max_edges: int = 100,
    near_threshold_px: float = 120.0,
    near_threshold_3d: float = 0.5,
) -> list[SceneRelation]:
    """Compute deterministic pairwise relation edges for extracted scene entities."""

    relations: list[SceneRelation] = []
    for index, source in enumerate(entities):
        for target in entities[index + 1 :]:
            if _same_semantic_entity(source, target):
                continue
            if source.point_3d is not None and target.point_3d is not None:
                relations.extend(_point_relations(source, target, near_threshold_3d))
            elif source.bbox_2d is not None and target.bbox_2d is not None:
                relations.extend(_bbox_relations(source, target, near_threshold_px))
    relations = sorted(relations, key=lambda item: (_relation_priority(item), item.confidence), reverse=True)
    return relations[: max(0, max_edges)]


def _load_query_results(root: Path, warnings: list[str]) -> list[tuple[QueryResult, str]]:
    records: list[tuple[QueryResult, str]] = []
    for path in sorted(root.rglob("query_result.json")):
        try:
            records.append((QueryResult.from_json(path), str(path)))
        except (json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Could not parse {path}: {exc}")
    if records:
        return records
    for path in sorted(root.rglob("scene_query_report.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Could not parse {path}: {exc}")
            continue
        for index, item in enumerate(raw.get("query_results") or []):
            if isinstance(item, dict):
                try:
                    records.append((QueryResult.from_dict(item), f"{path}#query_results[{index}]"))
                except ValueError as exc:
                    warnings.append(f"Could not parse embedded query result in {path}: {exc}")
    return records


def _entity_from_region(
    result: QueryResult,
    result_path: str,
    region: BoundingRegion,
    index: int,
    used_ids: set[str],
) -> SceneEntity:
    point_3d = _region_center_3d(region)
    center_2d = bbox_center(region.bbox_2d) if region.bbox_2d is not None else None
    evidence_type = "3d_region" if point_3d is not None else "2d_fallback"
    return SceneEntity(
        entity_id=_unique_entity_id(result.query, region.label, region.source_view, index, used_ids),
        label=region.label or result.query,
        query=result.query,
        score=region.score if region.score is not None else result.confidence,
        evidence_type=evidence_type,
        result_path=result_path,
        source_view=region.source_view,
        bbox_2d=region.bbox_2d,
        center_2d=center_2d,
        point_3d=point_3d,
        coordinate_frame=region.coordinate_frame,
        notes=region.notes or "",
    )


def _entity_from_point(
    result: QueryResult,
    result_path: str,
    point: Candidate3DPoint,
    index: int,
    used_ids: set[str],
) -> SceneEntity:
    return SceneEntity(
        entity_id=_unique_entity_id(result.query, point.label, point.source_view, index, used_ids),
        label=point.label or result.query,
        query=result.query,
        score=point.score if point.score is not None else result.confidence,
        evidence_type="3d_point",
        result_path=result_path,
        source_view=point.source_view,
        point_3d=(point.x, point.y, point.z),
        coordinate_frame="world",
        notes=str(point.metadata.get("notes", "")) if point.metadata else "",
    )


def _unique_entity_id(
    query: str,
    label: str,
    source_view: str | None,
    index: int,
    used_ids: set[str],
) -> str:
    base = slugify("_".join(part for part in [query, label, source_view or "scene", str(index)] if part))
    candidate = base
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used_ids.add(candidate)
    return candidate


def _region_center_3d(region: BoundingRegion) -> tuple[float, float, float] | None:
    if region.min_point_3d is None or region.max_point_3d is None:
        return None
    return tuple((lo + hi) / 2.0 for lo, hi in zip(region.min_point_3d, region.max_point_3d))


def _dedupe_entities(entities: list[SceneEntity]) -> list[SceneEntity]:
    seen: set[tuple[str, str | None, tuple[float, ...] | None, tuple[float, ...] | None]] = set()
    deduped: list[SceneEntity] = []
    for entity in entities:
        key = (
            _normalize_label(entity.label),
            entity.source_view,
            entity.bbox_2d,
            entity.point_3d,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _point_relations(
    source: SceneEntity,
    target: SceneEntity,
    near_threshold: float,
) -> list[SceneRelation]:
    assert source.point_3d is not None
    assert target.point_3d is not None
    sx, sy, sz = source.point_3d
    tx, ty, tz = target.point_3d
    dx = sx - tx
    dy = sy - ty
    dz = sz - tz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
        relation = "right_of" if dx > 0 else "left_of"
    elif abs(dz) >= abs(dx) and abs(dz) >= abs(dy):
        relation = "above" if dz > 0 else "below"
    elif distance <= near_threshold:
        relation = "near"
    else:
        relation = "far"
    confidence = _score_confidence(source, target) * (1.0 / (1.0 + distance))
    relations = [
        SceneRelation(
            subject_id=source.entity_id,
            object_id=target.entity_id,
            subject_label=source.label,
            object_label=target.label,
            relation=relation,
            confidence=_clamp(confidence),
            evidence_type="3d",
            source_view=source.source_view or target.source_view,
            notes="Approximate relation from available 3D points; coordinate convention depends on backend.",
        )
    ]
    horizontal = math.sqrt(dx * dx + dy * dy)
    if horizontal <= near_threshold and 0.02 < abs(dz) <= near_threshold:
        upper, lower = (source, target) if dz > 0 else (target, source)
        relations.append(
            SceneRelation(
                subject_id=upper.entity_id,
                object_id=lower.entity_id,
                subject_label=upper.label,
                object_label=lower.label,
                relation="on_top_of_or_supported_by",
                confidence=_clamp(_score_confidence(upper, lower) / (1.0 + horizontal + abs(dz))),
                evidence_type="3d",
                source_view=upper.source_view or lower.source_view,
                notes="Heuristic from nearby horizontal location and vertical separation.",
            )
        )
    return relations


def _bbox_relations(
    source: SceneEntity,
    target: SceneEntity,
    near_threshold_px: float,
) -> list[SceneRelation]:
    assert source.bbox_2d is not None
    assert target.bbox_2d is not None
    if source.source_view and target.source_view and source.source_view != target.source_view:
        return []
    sx, sy = bbox_center(source.bbox_2d)
    tx, ty = bbox_center(target.bbox_2d)
    dx = sx - tx
    dy = sy - ty
    distance = bbox_center_distance(source.bbox_2d, target.bbox_2d)
    if abs(dx) >= abs(dy):
        relation = "right_of" if dx > 0 else "left_of"
    elif abs(dy) > near_threshold_px:
        relation = "below" if dy > 0 else "above"
    elif distance <= near_threshold_px:
        relation = "near"
    else:
        relation = "far"
    relations = [
        SceneRelation(
            subject_id=source.entity_id,
            object_id=target.entity_id,
            subject_label=source.label,
            object_label=target.label,
            relation=relation,
            confidence=_clamp(_score_confidence(source, target) / (1.0 + distance / max(near_threshold_px, 1.0))),
            evidence_type="2d_fallback",
            source_view=source.source_view or target.source_view,
            notes="Image-space fallback from rendered 2D boxes; not a metric 3D claim.",
        )
    ]
    relations.extend(_bbox_overlap_relations(source, target))
    relations.extend(_bbox_support_relations(source, target))
    relations.extend(_bbox_containment_relations(source, target))
    return relations


def _bbox_overlap_relations(source: SceneEntity, target: SceneEntity) -> list[SceneRelation]:
    assert source.bbox_2d is not None
    assert target.bbox_2d is not None
    iou = bbox_iou(source.bbox_2d, target.bbox_2d)
    if iou < 0.10:
        return []
    return [
        SceneRelation(
            subject_id=source.entity_id,
            object_id=target.entity_id,
            subject_label=source.label,
            object_label=target.label,
            relation="overlaps",
            confidence=_clamp(iou * _score_confidence(source, target)),
            evidence_type="2d_fallback",
            source_view=source.source_view or target.source_view,
            notes="Image-space overlap; may indicate occlusion, containment, or duplicate query hits.",
        )
    ]


def _bbox_support_relations(source: SceneEntity, target: SceneEntity) -> list[SceneRelation]:
    relations: list[SceneRelation] = []
    for lower, upper in ((source, target), (target, source)):
        support = _support_score(lower, upper)
        if support <= 0:
            continue
        confidence = _clamp(support * _score_confidence(lower, upper))
        relations.append(
            SceneRelation(
                subject_id=lower.entity_id,
                object_id=upper.entity_id,
                subject_label=lower.label,
                object_label=upper.label,
                relation="likely_supports",
                confidence=confidence,
                evidence_type="2d_fallback",
                source_view=lower.source_view or upper.source_view,
                notes="Image-space heuristic: upper box bottom is close to lower box top with horizontal overlap.",
            )
        )
        relations.append(
            SceneRelation(
                subject_id=upper.entity_id,
                object_id=lower.entity_id,
                subject_label=upper.label,
                object_label=lower.label,
                relation="likely_on_top_of",
                confidence=confidence,
                evidence_type="2d_fallback",
                source_view=lower.source_view or upper.source_view,
                notes="Converse of likely_supports; use as qualitative evidence only.",
            )
        )
    return relations


def _bbox_containment_relations(source: SceneEntity, target: SceneEntity) -> list[SceneRelation]:
    relations: list[SceneRelation] = []
    for inner, outer in ((source, target), (target, source)):
        assert inner.bbox_2d is not None
        assert outer.bbox_2d is not None
        ratio = containment_ratio(inner.bbox_2d, outer.bbox_2d)
        if ratio < 0.65:
            continue
        relations.append(
            SceneRelation(
                subject_id=inner.entity_id,
                object_id=outer.entity_id,
                subject_label=inner.label,
                object_label=outer.label,
                relation="likely_contained_in",
                confidence=_clamp(ratio * _score_confidence(inner, outer)),
                evidence_type="2d_fallback",
                source_view=inner.source_view or outer.source_view,
                notes="Image-space containment fallback; verify with real 3D geometry before strong claims.",
            )
        )
    return relations


def _support_score(lower: SceneEntity, upper: SceneEntity) -> float:
    assert lower.bbox_2d is not None
    assert upper.bbox_2d is not None
    lower_x1, lower_y1, lower_x2, _lower_y2 = lower.bbox_2d
    upper_x1, _upper_y1, upper_x2, upper_y2 = upper.bbox_2d
    overlap = bbox_intersection_area(
        (lower_x1, 0.0, lower_x2, 1.0),
        (upper_x1, 0.0, upper_x2, 1.0),
    )
    min_width = max(1.0, min(lower_x2 - lower_x1, upper_x2 - upper_x1))
    overlap_ratio = overlap / min_width
    vertical_gap = lower_y1 - upper_y2
    if overlap_ratio < 0.25 or not (-25.0 <= vertical_gap <= 80.0):
        return 0.0
    gap_penalty = 1.0 / (1.0 + max(0.0, abs(vertical_gap)) / 40.0)
    return min(1.0, overlap_ratio * gap_penalty)


def _score_confidence(source: SceneEntity, target: SceneEntity) -> float:
    values = [value for value in (source.score, target.score) if value is not None]
    if not values:
        return 0.5
    return sum(values) / len(values)


def _same_semantic_entity(source: SceneEntity, target: SceneEntity) -> bool:
    same_label = _normalize_label(source.label) == _normalize_label(target.label)
    same_query = _normalize_label(source.query) == _normalize_label(target.query)
    same_view = source.source_view == target.source_view
    same_box = source.bbox_2d is not None and source.bbox_2d == target.bbox_2d
    same_point = source.point_3d is not None and source.point_3d == target.point_3d
    return (same_label and same_query and same_view) or (same_label and (same_box or same_point))


def _normalize_label(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _relation_priority(relation: SceneRelation) -> int:
    priorities = {
        "likely_supports": 6,
        "likely_on_top_of": 6,
        "on_top_of_or_supported_by": 6,
        "likely_contained_in": 5,
        "overlaps": 4,
        "near": 3,
        "left_of": 2,
        "right_of": 2,
        "above": 2,
        "below": 2,
        "far": 1,
    }
    return priorities.get(relation.relation, 0)


def _mock_entities() -> list[SceneEntity]:
    return [
        SceneEntity(
            entity_id="desk_view_0000",
            label="desk",
            query="desk",
            score=0.82,
            evidence_type="2d_fallback",
            result_path="<dry-run>",
            source_view="view_0000",
            bbox_2d=(60.0, 220.0, 460.0, 355.0),
            center_2d=(260.0, 287.5),
            coordinate_frame="image",
            notes="Synthetic dry-run entity.",
        ),
        SceneEntity(
            entity_id="mug_view_0000",
            label="mug",
            query="mug",
            score=0.76,
            evidence_type="2d_fallback",
            result_path="<dry-run>",
            source_view="view_0000",
            bbox_2d=(95.0, 150.0, 160.0, 225.0),
            center_2d=(127.5, 187.5),
            coordinate_frame="image",
            notes="Synthetic dry-run entity.",
        ),
        SceneEntity(
            entity_id="laptop_view_0000",
            label="laptop",
            query="laptop",
            score=0.71,
            evidence_type="2d_fallback",
            result_path="<dry-run>",
            source_view="view_0000",
            bbox_2d=(210.0, 135.0, 390.0, 230.0),
            center_2d=(300.0, 182.5),
            coordinate_frame="image",
            notes="Synthetic dry-run entity.",
        ),
    ]


def _relation_lines(relations: list[SceneRelation], limit: int = 40) -> list[str]:
    if not relations:
        return ["- No relation edges were inferred."]
    lines = ["| Subject | Relation | Object | Confidence | Evidence |", "| --- | --- | --- | ---: | --- |"]
    for relation in relations[:limit]:
        lines.append(
            "| "
            f"`{relation.subject_label}` | `{relation.relation}` | `{relation.object_label}` | "
            f"{relation.confidence:.3f} | `{relation.evidence_type}` |"
        )
    if len(relations) > limit:
        lines.append(f"| ... | ... | ... | ... | {len(relations) - limit} more edges omitted |")
    return lines


def _entity_lines(entities: list[SceneEntity], limit: int = 40) -> list[str]:
    if not entities:
        return ["- No entities were extracted."]
    lines = ["| Entity | Query | Score | Evidence | Source |", "| --- | --- | ---: | --- | --- |"]
    for entity in entities[:limit]:
        score = "" if entity.score is None else f"{entity.score:.3f}"
        source = entity.source_view or "scene"
        lines.append(
            f"| `{entity.label}` | `{entity.query}` | {score} | `{entity.evidence_type}` | `{source}` |"
        )
    if len(entities) > limit:
        lines.append(f"| ... | ... | ... | ... | {len(entities) - limit} more entities omitted |")
    return lines


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
