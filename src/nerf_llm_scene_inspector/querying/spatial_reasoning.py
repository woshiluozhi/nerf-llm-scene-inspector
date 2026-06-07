"""Spatial reasoning utilities for semantic query results."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from nerf_llm_scene_inspector.backends.base import BoundingRegion, Candidate3DPoint, QueryResult


@dataclass
class SpatialRelation:
    """A pairwise spatial relation between two candidates."""

    source_label: str
    target_label: str
    relation: str
    confidence: float
    evidence_type: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def rank_candidate_regions(regions: Iterable[BoundingRegion]) -> list[BoundingRegion]:
    """Sort regions by descending score, leaving missing scores last."""

    return sorted(regions, key=lambda region: region.score if region.score is not None else -1.0, reverse=True)


def rank_query_results(results: Iterable[QueryResult]) -> list[QueryResult]:
    """Sort query results by confidence and best region score."""

    def score(result: QueryResult) -> float:
        region_score = max((r.score or 0.0 for r in result.bounding_regions), default=0.0)
        return max(result.confidence or 0.0, region_score)

    return sorted(results, key=score, reverse=True)


def aggregate_multi_query_results(results: Iterable[QueryResult]) -> QueryResult:
    """Aggregate multiple query results into one report-like QueryResult."""

    result_list = list(results)
    if not result_list:
        return QueryResult(query="", backend_name="aggregate", config_path="", warnings=["No results to aggregate."])

    rendered = []
    points = []
    regions = []
    warnings = []
    commands = []
    for result in result_list:
        rendered.extend(result.rendered_images)
        points.extend(result.candidate_points)
        regions.extend(result.bounding_regions)
        warnings.extend(result.warnings)
        commands.extend(result.provenance.get("commands", []))

    ranked_regions = rank_candidate_regions(regions)
    confidence_values = [r.confidence for r in result_list if r.confidence is not None]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    return QueryResult(
        query="; ".join(result.query for result in result_list),
        backend_name="aggregate",
        config_path=result_list[0].config_path,
        rendered_images=rendered,
        candidate_points=points,
        bounding_regions=ranked_regions,
        confidence=confidence,
        warnings=warnings,
        provenance={"commands": commands, "source_queries": [result.query for result in result_list]},
    )


def pairwise_spatial_relations(
    points: Iterable[Candidate3DPoint],
    near_threshold: float = 0.5,
) -> list[SpatialRelation]:
    """Compute approximate world-space pairwise relations."""

    point_list = list(points)
    relations: list[SpatialRelation] = []
    for index, source in enumerate(point_list):
        for target in point_list[index + 1 :]:
            dx = source.x - target.x
            dy = source.y - target.y
            dz = source.z - target.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            confidence = 1.0 / (1.0 + distance)
            if abs(dx) > abs(dy) and abs(dx) > abs(dz):
                relation = "right_of" if dx > 0 else "left_of"
            elif abs(dz) > abs(dx) and abs(dz) > abs(dy):
                relation = "above" if dz > 0 else "below"
            elif distance <= near_threshold:
                relation = "near"
            else:
                relation = "far"
            relations.append(
                SpatialRelation(
                    source_label=source.label,
                    target_label=target.label,
                    relation=relation,
                    confidence=confidence,
                    evidence_type="3d",
                )
            )
            support_relation = support_heuristic(source, target, near_threshold=near_threshold)
            if support_relation is not None:
                relations.append(support_relation)
    return relations


def support_heuristic(
    source: Candidate3DPoint,
    target: Candidate3DPoint,
    near_threshold: float = 0.5,
) -> SpatialRelation | None:
    """Approximate support/on-top relation from 3D point positions."""

    horizontal = math.sqrt((source.x - target.x) ** 2 + (source.y - target.y) ** 2)
    vertical = source.z - target.z
    if horizontal <= near_threshold and 0.02 < vertical <= near_threshold:
        return SpatialRelation(
            source_label=source.label,
            target_label=target.label,
            relation="on_top_of_or_supported_by",
            confidence=1.0 / (1.0 + horizontal + abs(vertical)),
            evidence_type="3d",
            notes="Heuristic from nearby horizontal position and positive vertical offset.",
        )
    return None


def image_space_relation(
    source: BoundingRegion,
    target: BoundingRegion,
    near_threshold_px: float = 80.0,
) -> SpatialRelation:
    """Fallback relation from 2D bounding boxes."""

    if source.bbox_2d is None or target.bbox_2d is None:
        return SpatialRelation(
            source_label=source.label,
            target_label=target.label,
            relation="unknown",
            confidence=0.0,
            evidence_type="2d_fallback",
            notes="Missing 2D boxes.",
        )
    sx, sy = _bbox_center(source.bbox_2d)
    tx, ty = _bbox_center(target.bbox_2d)
    dx = sx - tx
    dy = sy - ty
    distance = math.sqrt(dx * dx + dy * dy)
    if abs(dx) > abs(dy):
        relation = "right_of" if dx > 0 else "left_of"
    elif abs(dy) > near_threshold_px:
        relation = "below" if dy > 0 else "above"
    elif distance <= near_threshold_px:
        relation = "near"
    else:
        relation = "far"
    return SpatialRelation(
        source_label=source.label,
        target_label=target.label,
        relation=relation,
        confidence=1.0 / (1.0 + distance / max(near_threshold_px, 1.0)),
        evidence_type="2d_fallback",
        notes="Image-space fallback; not a metric 3D relation.",
    )


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
