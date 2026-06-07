"""Lightweight evaluation metrics for semantic scene queries."""

from __future__ import annotations

from statistics import mean

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult


BBox = tuple[float, float, float, float]


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute 2D intersection over union for xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def topk_localization_hit(
    predicted_regions: list[BoundingRegion],
    target_bbox: BBox | None,
    *,
    k: int = 5,
    iou_threshold: float = 0.25,
    acceptable_views: list[str] | None = None,
) -> bool:
    """Return whether any top-k region overlaps a manual target region."""

    if target_bbox is None:
        return False
    acceptable = set(acceptable_views or [])
    for region in predicted_regions[:k]:
        if region.bbox_2d is None:
            continue
        if acceptable and region.source_view and region.source_view not in acceptable:
            continue
        if bbox_iou(region.bbox_2d, target_bbox) >= iou_threshold:
            return True
    return False


def semantic_query_success_rate(successes: list[bool]) -> float:
    """Average binary query success."""

    return mean([1.0 if item else 0.0 for item in successes]) if successes else 0.0


def average_relevancy_score(results: list[QueryResult]) -> float:
    """Average confidence over query results with scores."""

    scores = [result.confidence for result in results if result.confidence is not None]
    return float(mean(scores)) if scores else 0.0


def mean_best_iou(results: list[QueryResult], annotations: dict[str, BBox]) -> float:
    """Mean best 2D IoU by query string."""

    values: list[float] = []
    for result in results:
        target = annotations.get(result.query)
        if target is None:
            continue
        best = max(
            (bbox_iou(region.bbox_2d, target) for region in result.bounding_regions if region.bbox_2d),
            default=0.0,
        )
        values.append(best)
    return float(mean(values)) if values else 0.0


def qualitative_success_table(
    results: list[QueryResult],
    annotation_by_query: dict[str, dict[str, object]],
    *,
    k: int = 5,
) -> list[dict[str, object]]:
    """Create per-query qualitative rows."""

    rows: list[dict[str, object]] = []
    for result in results:
        annotation = annotation_by_query.get(result.query, {})
        bbox = annotation.get("bbox_2d")
        target_bbox = tuple(float(item) for item in bbox) if bbox else None
        hit = topk_localization_hit(
            result.bounding_regions,
            target_bbox,
            k=k,
            acceptable_views=list(annotation.get("acceptable_views") or []),
        )
        best_iou = 0.0
        if target_bbox:
            best_iou = max(
                (
                    bbox_iou(region.bbox_2d, target_bbox)
                    for region in result.bounding_regions
                    if region.bbox_2d
                ),
                default=0.0,
            )
        rows.append(
            {
                "query": result.query,
                "target_description": annotation.get("target_description", ""),
                "topk_hit": hit,
                "best_iou_2d": best_iou,
                "confidence": result.confidence if result.confidence is not None else "",
                "num_regions": len(result.bounding_regions),
                "warnings": "; ".join(result.warnings),
            }
        )
    return rows
