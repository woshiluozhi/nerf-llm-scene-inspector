"""Deterministic scene-answer synthesis from planned semantic queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from nerf_llm_scene_inspector.backends.base import (
    BoundingRegion,
    Candidate3DPoint,
    QueryResult,
    RenderedView,
)


@dataclass
class SceneAnswerEvidence:
    """One compact evidence item supporting a natural-language scene answer."""

    query: str
    label: str
    score: float | None = None
    evidence_type: str = "query_result"
    source_view: str | None = None
    coordinate_frame: str = "unknown"
    bbox_2d: tuple[float, float, float, float] | None = None
    point_3d: tuple[float, float, float] | None = None
    rendered_artifacts: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneAnswer:
    """Structured synthesized answer for a scene-level user task."""

    answer: str
    support_level: str
    confidence: float | None
    evidence: list[SceneAnswerEvidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommended_followups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "support_level": self.support_level,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": list(self.limitations),
            "recommended_followups": list(self.recommended_followups),
        }


def synthesize_scene_answer(
    *,
    task: str,
    plan: dict[str, Any],
    results: list[QueryResult],
    top_k: int = 5,
) -> SceneAnswer:
    """Create an evidence-grounded natural-language answer from query results."""

    evidence = _collect_evidence(results, top_k=top_k)
    labels = _unique_labels(evidence)
    answer = _format_answer(plan, labels)
    support_level = _support_level(results)
    confidence = _confidence(evidence, results, plan)
    limitations = _limitations(results, support_level)
    followups = _followups(task, plan, results, evidence)
    if evidence:
        answer = (
            f"{answer} Strongest evidence is based on {len(evidence)} ranked "
            f"{_support_phrase(support_level)} item(s)."
        )
    else:
        answer = f"{answer} No backend evidence was available yet."
    return SceneAnswer(
        answer=answer,
        support_level=support_level,
        confidence=confidence,
        evidence=evidence,
        limitations=limitations,
        recommended_followups=followups,
    )


def _collect_evidence(results: list[QueryResult], *, top_k: int) -> list[SceneAnswerEvidence]:
    evidence: list[SceneAnswerEvidence] = []
    for result in results:
        ranked_regions = sorted(
            result.bounding_regions,
            key=lambda region: region.score if region.score is not None else -1.0,
            reverse=True,
        )
        for region in ranked_regions[: max(top_k, 1)]:
            evidence.append(_region_evidence(result, region))
        ranked_points = sorted(
            result.candidate_points,
            key=lambda point: point.score if point.score is not None else -1.0,
            reverse=True,
        )
        for point in ranked_points[: max(top_k, 1)]:
            evidence.append(_point_evidence(result, point))
        if not ranked_regions:
            view = _best_rendered_view(result.rendered_images)
            if view is not None:
                evidence.append(_view_evidence(result, view))
            elif result.confidence is not None:
                evidence.append(
                    SceneAnswerEvidence(
                        query=result.query,
                        label=result.query,
                        score=result.confidence,
                        evidence_type="query_confidence",
                        notes="No region or rendered view was available; using query-level confidence.",
                    )
                )
    return sorted(
        evidence,
        key=lambda item: item.score if item.score is not None else -1.0,
        reverse=True,
    )[:top_k]


def _point_evidence(result: QueryResult, point: Candidate3DPoint) -> SceneAnswerEvidence:
    return SceneAnswerEvidence(
        query=result.query,
        label=point.label or result.query,
        score=point.score if point.score is not None else result.confidence,
        evidence_type="3d_point",
        source_view=point.source_view,
        coordinate_frame="world",
        point_3d=(point.x, point.y, point.z),
        rendered_artifacts=_artifact_paths(result.rendered_images, point.source_view),
        notes=str(point.metadata.get("notes") or "Approximate semantic 3D point from backend output."),
    )


def _region_evidence(result: QueryResult, region: BoundingRegion) -> SceneAnswerEvidence:
    return SceneAnswerEvidence(
        query=result.query,
        label=region.label or result.query,
        score=region.score if region.score is not None else result.confidence,
        evidence_type="3d_region" if region.coordinate_frame == "world" else "2d_region",
        source_view=region.source_view,
        coordinate_frame=region.coordinate_frame,
        bbox_2d=region.bbox_2d,
        rendered_artifacts=_artifact_paths(result.rendered_images, region.source_view),
        notes=region.notes or "",
    )


def _view_evidence(result: QueryResult, view: RenderedView) -> SceneAnswerEvidence:
    return SceneAnswerEvidence(
        query=result.query,
        label=result.query,
        score=view.score if view.score is not None else result.confidence,
        evidence_type=f"rendered_{view.kind}",
        source_view=view.camera_id,
        coordinate_frame="image",
        rendered_artifacts=[view.path],
        notes=view.caption or "Rendered view evidence without extracted region.",
    )


def _best_rendered_view(views: list[RenderedView]) -> RenderedView | None:
    if not views:
        return None
    priority = {"overlay": 3, "relevancy": 2, "rgb": 1}
    return sorted(
        views,
        key=lambda view: (priority.get(view.kind, 0), view.score if view.score is not None else -1.0),
        reverse=True,
    )[0]


def _artifact_paths(views: list[RenderedView], source_view: str | None) -> list[str]:
    matched = [
        view.path
        for view in views
        if source_view is None or view.camera_id is None or view.camera_id == source_view
    ]
    return matched[:3]


def _unique_labels(evidence: list[SceneAnswerEvidence]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        label = item.label.strip()
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            labels.append(label)
    return labels


def _format_answer(plan: dict[str, Any], labels: list[str]) -> str:
    template = str(plan.get("final_answer_template") or "Likely relevant scene regions are {items}.")
    items = ", ".join(labels) if labels else "pending"
    try:
        return template.format(items=items)
    except (IndexError, KeyError, ValueError):
        return f"Likely relevant scene regions are {items}."


def _support_level(results: list[QueryResult]) -> str:
    if not results:
        return "no_backend_evidence"
    if any(result.candidate_points for result in results):
        return "3d_candidate_points"
    if any(_has_world_region(result.bounding_regions) for result in results):
        return "3d_regions"
    if any(region.bbox_2d is not None for result in results for region in result.bounding_regions):
        return "2d_relevancy_fallback"
    if any(result.rendered_images for result in results):
        return "rendered_relevancy_only"
    return "query_only"


def _has_world_region(regions: list[BoundingRegion]) -> bool:
    return any(region.coordinate_frame == "world" for region in regions)


def _confidence(
    evidence: list[SceneAnswerEvidence],
    results: list[QueryResult],
    plan: dict[str, Any],
) -> float | None:
    values = [item.score for item in evidence if item.score is not None]
    if not values:
        values = [result.confidence for result in results if result.confidence is not None]
    if values:
        return round(sum(values) / len(values), 4)
    raw_plan_confidence = plan.get("confidence")
    try:
        return round(float(raw_plan_confidence), 4) if raw_plan_confidence is not None else None
    except (TypeError, ValueError):
        return None


def _limitations(results: list[QueryResult], support_level: str) -> list[str]:
    limitations: list[str] = []
    if support_level in {"2d_relevancy_fallback", "rendered_relevancy_only", "query_only"}:
        limitations.append(
            "Metric 3D localization was not available; spatial claims should be treated as image-space or qualitative evidence."
        )
    if _has_dry_run_caption(results):
        limitations.append(
            "This answer uses dry-run synthetic render artifacts and is not evidence of trained LERF model quality."
        )
    warnings = [warning for result in results for warning in result.warnings]
    if warnings:
        limitations.append("Backend warnings were present: " + "; ".join(warnings[:3]))
    if not results:
        limitations.append("No backend query results were produced.")
    return _dedupe(limitations)


def _has_dry_run_caption(results: list[QueryResult]) -> bool:
    for result in results:
        for view in result.rendered_images:
            if "dry-run" in (view.caption or "").lower():
                return True
    return False


def _followups(
    task: str,
    plan: dict[str, Any],
    results: list[QueryResult],
    evidence: list[SceneAnswerEvidence],
) -> list[str]:
    followups: list[str] = []
    if len(evidence) < min(3, max(1, len(results))):
        followups.append("Run additional views or prompts to improve evidence coverage.")
    if plan.get("negative_visual_queries"):
        followups.append("Run negative/disambiguation prompts before making a final decision.")
    if plan.get("relation_hypotheses"):
        followups.append("Use spatial_reasoning utilities on 3D points or image-space boxes to verify relation hypotheses.")
    if "?" in task or any(word in task.lower() for word in ("where", "which", "safe", "support")):
        followups.append("Inspect the overlay images manually before acting on the answer.")
    return _dedupe(followups)


def _support_phrase(support_level: str) -> str:
    return {
        "3d_candidate_points": "3D candidate point",
        "3d_regions": "3D region",
        "2d_relevancy_fallback": "2D relevancy fallback",
        "rendered_relevancy_only": "rendered relevancy",
        "query_only": "query-level",
        "no_backend_evidence": "unverified",
    }.get(support_level, support_level.replace("_", " "))


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
