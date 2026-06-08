"""Portfolio-facing evidence scorecards for pipeline runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


EvidenceLevel = Literal[
    "blocked",
    "needs_evidence",
    "needs_review",
    "dry_run_demo_ready",
    "portfolio_ready_real_run",
]


@dataclass
class EvidenceCriterion:
    """One scored evidence criterion."""

    name: str
    category: str
    score: int
    max_score: int
    status: str
    detail: str
    recommendation: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EvidenceScorecard:
    """Compact scorecard for deciding whether a run is strong enough to share."""

    run_dir: str
    scene_name: str
    evidence_level: EvidenceLevel
    score: int
    max_score: int
    dry_run: bool
    backend: str
    query_count: int
    query_report_count: int
    overlay_count: int
    bbox_annotated_query_count: int
    evaluated_query_count: int
    summary: str
    top_recommendations: list[str] = field(default_factory=list)
    criteria: list[EvidenceCriterion] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "evidence_level": self.evidence_level,
            "score": self.score,
            "max_score": self.max_score,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "query_count": self.query_count,
            "query_report_count": self.query_report_count,
            "overlay_count": self.overlay_count,
            "bbox_annotated_query_count": self.bbox_annotated_query_count,
            "evaluated_query_count": self.evaluated_query_count,
            "summary": self.summary,
            "top_recommendations": list(self.top_recommendations),
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "metrics": dict(self.metrics),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Evidence Scorecard",
            "",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Backend: {self.backend or 'unknown'}",
            f"- Evidence level: {self.evidence_level}",
            f"- Score: {self.score}/{self.max_score}",
            f"- Dry run: {self.dry_run}",
            f"- Query reports: {self.query_report_count}/{self.query_count}",
            f"- Overlays: {self.overlay_count}",
            f"- BBox-annotated queries: {self.bbox_annotated_query_count}",
            f"- Evaluated queries: {self.evaluated_query_count}",
            f"- Summary: {self.summary}",
            "",
            "## Top Recommendations",
            "",
            *_markdown_list(self.top_recommendations),
            "",
            "## Criteria",
            "",
            *_criterion_lines(self.criteria),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_evidence_scorecard(run_dir: str | Path) -> EvidenceScorecard:
    """Score how strong one run is as a reproducible portfolio artifact."""

    root = Path(run_dir)
    pipeline_summary = _read_json(root / "pipeline_summary.json")
    preflight = _read_json(root / "preflight_report.json")
    capture_validation = _read_json(root / "capture_manifest_validation.json")
    environment = _read_json(root / "environment_report.json")
    scene = _read_json(root / "scene_data_inspection.json")
    audit = _read_json(root / "run_audit.json")
    recommendations = _read_json(root / "run_recommendations.json")
    annotation_validation = _read_json(root / "evaluation" / "annotation_validation.json")
    evaluation = _read_json(root / "evaluation" / "eval_summary.json")

    scene_name = str(pipeline_summary.get("scene_name") or root.name)
    dry_run = bool(pipeline_summary.get("dry_run"))
    backend = str(pipeline_summary.get("backend") or "")
    query_count = len([query for query in pipeline_summary.get("queries") or [] if str(query).strip()])
    query_report_count = len(list((root / "queries").rglob("scene_query_report.json")))
    overlay_count = _count_overlays(root)
    bbox_count = _safe_int(evaluation.get("num_bbox_annotated_queries"))
    evaluated_count = _safe_int(evaluation.get("num_evaluated_queries"))

    criteria = [
        _pipeline_criterion(root, pipeline_summary, audit),
        _capture_manifest_criterion(capture_validation),
        _real_run_criterion(dry_run, preflight, environment),
        _scene_criterion(scene),
        _query_criterion(query_count, query_report_count, overlay_count, evaluation),
        _evaluation_criterion(query_count, annotation_validation, evaluation),
        _portfolio_criterion(root),
    ]
    score = sum(criterion.score for criterion in criteria)
    max_score = sum(criterion.max_score for criterion in criteria)
    level = _evidence_level(
        score=score,
        max_score=max_score,
        dry_run=dry_run,
        pipeline_summary=pipeline_summary,
        preflight=preflight,
        capture_validation=capture_validation,
        audit=audit,
    )
    top_recommendations = _top_recommendations(criteria, recommendations)
    return EvidenceScorecard(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        evidence_level=level,
        score=score,
        max_score=max_score,
        dry_run=dry_run,
        backend=backend,
        query_count=query_count,
        query_report_count=query_report_count,
        overlay_count=overlay_count,
        bbox_annotated_query_count=bbox_count,
        evaluated_query_count=evaluated_count,
        summary=_summary(level, score, max_score),
        top_recommendations=top_recommendations,
        criteria=criteria,
        metrics={
            "top_k_hit_rate": evaluation.get("top_k_hit_rate"),
            "mean_iou_2d": evaluation.get("mean_iou_2d"),
            "semantic_success_rate": evaluation.get("semantic_success_rate"),
            "average_relevancy_score": evaluation.get("average_relevancy_score"),
            "scene_quality_score": scene.get("quality_score"),
            "pose_coverage_score": scene.get("pose_coverage_score"),
            "audit_status": audit.get("status"),
            "preflight_status": preflight.get("status"),
            "capture_manifest_status": capture_validation.get("status"),
        },
    )


def _pipeline_criterion(root: Path, summary: dict[str, Any], audit: dict[str, Any]) -> EvidenceCriterion:
    score = 0
    notes: list[str] = []
    if summary.get("success") is True:
        score += 7
        notes.append("pipeline_summary.success=true")
    failed_steps = [
        str(step.get("name") or "unknown")
        for step in summary.get("steps") or []
        if isinstance(step, dict) and step.get("status") == "failed"
    ]
    if not failed_steps and summary:
        score += 4
    if (root / "logs").exists() and list((root / "logs").glob("*.json")):
        score += 2
    if audit.get("status") == "ready":
        score += 2
    elif audit.get("status") == "needs_review":
        score += 1
    status = _criterion_status(score, 15)
    return EvidenceCriterion(
        name="pipeline_integrity",
        category="reproducibility",
        score=score,
        max_score=15,
        status=status,
        detail=", ".join(notes) or "Pipeline summary is missing or incomplete.",
        recommendation="" if status == "pass" else "Fix failed/skipped pipeline steps and rerun audit_run.py.",
        artifact="pipeline_summary.json",
    )


def _real_run_criterion(
    dry_run: bool,
    preflight: dict[str, Any],
    environment: dict[str, Any],
) -> EvidenceCriterion:
    score = 0
    details: list[str] = []
    if dry_run:
        score += 3
        details.append("dry-run smoke artifact")
    else:
        score += 7
        details.append("real-mode run")
    preflight_status = str(preflight.get("status") or "")
    if preflight_status == "ready":
        score += 5
    elif preflight_status == "needs_attention":
        score += 3
    env_ok = environment.get("ok") is True
    if env_ok:
        score += 3
    status = _criterion_status(score, 15)
    return EvidenceCriterion(
        name="real_run_readiness",
        category="environment",
        score=score,
        max_score=15,
        status=status,
        detail=f"{'; '.join(details)}, preflight={preflight_status or 'missing'}, env_ok={env_ok}",
        recommendation=(
            ""
            if status == "pass"
            else "Run on a CUDA machine with upstream tools installed and review preflight_report.md."
        ),
        artifact="preflight_report.md",
    )


def _capture_manifest_criterion(validation: dict[str, Any]) -> EvidenceCriterion:
    status_raw = str(validation.get("status") or "missing")
    fail_count = _safe_int(validation.get("fail_count"))
    warn_count = _safe_int(validation.get("warn_count"))
    if status_raw == "ready":
        score = 10
        detail = "capture manifest is ready"
    elif status_raw == "needs_review":
        score = 5
        detail = f"capture manifest needs review; warnings={warn_count}"
    elif status_raw == "blocked" or fail_count:
        score = 0
        detail = f"capture manifest is blocked; failures={fail_count}, warnings={warn_count}"
    else:
        score = 0
        detail = "capture manifest validation is missing"
    status = _criterion_status(score, 10)
    return EvidenceCriterion(
        name="capture_manifest_quality",
        category="capture",
        score=score,
        max_score=10,
        status=status,
        detail=detail,
        recommendation=(
            ""
            if status == "pass"
            else "Complete capture_manifest.json and confirm static-scene, overlap, and privacy-review fields."
        ),
        artifact="capture_manifest_validation.md",
    )


def _scene_criterion(scene: dict[str, Any]) -> EvidenceCriterion:
    score = 0
    if scene.get("ready_for_training") is True:
        score += 6
    score += round(5 * min(max(_safe_float(scene.get("quality_score")), 0.0), 1.0))
    frame_count = _safe_int(scene.get("frame_count"))
    if frame_count >= 50:
        score += 2
    elif frame_count >= 20:
        score += 1
    pose_score = _safe_float(scene.get("pose_coverage_score"))
    if pose_score >= 1.0:
        score += 2
    elif pose_score >= 0.5:
        score += 1
    status = _criterion_status(score, 15)
    return EvidenceCriterion(
        name="scene_data_quality",
        category="scene",
        score=score,
        max_score=15,
        status=status,
        detail=(
            f"ready={scene.get('ready_for_training')}, quality={scene.get('quality_score')}, "
            f"frames={frame_count}, pose_coverage={scene.get('pose_coverage_score')}"
        ),
        recommendation="" if status == "pass" else "Recapture/reprocess with more overlap, parallax, and frames.",
        artifact="scene_data_inspection.md",
    )


def _query_criterion(
    query_count: int,
    query_report_count: int,
    overlay_count: int,
    evaluation: dict[str, Any],
) -> EvidenceCriterion:
    score = 0
    if query_count >= 3:
        score += 6
    elif query_count == 2:
        score += 3
    elif query_count:
        score += 1
    if query_count and query_report_count >= query_count:
        score += 5
    elif query_report_count:
        score += 3
    if overlay_count >= max(query_count, 1):
        score += 4
    elif overlay_count:
        score += 3
    relevancy = _safe_float(evaluation.get("average_relevancy_score"))
    if relevancy > 0:
        score += 2
    if query_count >= 3 and overlay_count >= query_count:
        score += 3
    status = _criterion_status(score, 20)
    return EvidenceCriterion(
        name="semantic_query_evidence",
        category="querying",
        score=score,
        max_score=20,
        status=status,
        detail=(
            f"queries={query_count}, query_reports={query_report_count}, overlays={overlay_count}, "
            f"average_relevancy={evaluation.get('average_relevancy_score')}"
        ),
        recommendation="" if status == "pass" else "Run at least 3 representative queries and keep RGB/relevancy overlays.",
        artifact="queries/",
    )


def _evaluation_criterion(
    query_count: int,
    annotation_validation: dict[str, Any],
    evaluation: dict[str, Any],
) -> EvidenceCriterion:
    score = 0
    if annotation_validation.get("ok") is True:
        score += 4
    bbox_count = _safe_int(evaluation.get("num_bbox_annotated_queries"))
    if bbox_count >= 3:
        score += 6
    elif bbox_count >= 2:
        score += 4
    elif bbox_count:
        score += 2
    evaluated = _safe_int(evaluation.get("num_evaluated_queries"))
    if evaluated >= 3:
        score += 4
    elif evaluated >= 2:
        score += 3
    elif evaluated:
        score += 2
    has_metrics = evaluation.get("top_k_hit_rate") is not None and evaluation.get("mean_iou_2d") is not None
    if has_metrics:
        score += 4
    if evaluation.get("num_result_queries") is not None:
        score += 2
    status = _criterion_status(score, 20)
    return EvidenceCriterion(
        name="annotation_and_evaluation",
        category="evaluation",
        score=score,
        max_score=20,
        status=status,
        detail=(
            f"annotation_ok={annotation_validation.get('ok')}, bbox_queries={bbox_count}, "
            f"evaluated={evaluated}, metrics_present={has_metrics}"
        ),
        recommendation=(
            ""
            if status == "pass"
            else "Fill manual bbox_2d labels for core queries and rerun evaluate_queries.py."
        ),
        artifact="evaluation/eval_summary.json",
    )


def _portfolio_criterion(root: Path) -> EvidenceCriterion:
    expected = {
        "project_report.md": 3,
        "portfolio_result_card.md": 3,
        "run_recommendations.md": 2,
        "reproduction_manifest.json": 2,
        "reproduction_report.md": 2,
        "reproduce_run.sh": 1,
        "demo_assets/query_grid.png": 2,
    }
    score = sum(points for relative, points in expected.items() if (root / relative).exists())
    missing = [relative for relative in expected if not (root / relative).exists()]
    status = _criterion_status(score, 15)
    return EvidenceCriterion(
        name="portfolio_packaging",
        category="presentation",
        score=score,
        max_score=15,
        status=status,
        detail="missing: " + ", ".join(missing) if missing else "portfolio-facing artifacts found",
        recommendation="" if status == "pass" else "Regenerate demo assets, reports, recommendations, and reproduction bundle.",
        artifact="portfolio_result_card.md",
    )


def _evidence_level(
    *,
    score: int,
    max_score: int,
    dry_run: bool,
    pipeline_summary: dict[str, Any],
    preflight: dict[str, Any],
    capture_validation: dict[str, Any],
    audit: dict[str, Any],
) -> EvidenceLevel:
    ratio = score / max(max_score, 1)
    capture_status = str(capture_validation.get("status") or "")
    if (
        pipeline_summary.get("success") is False
        or audit.get("status") == "blocked"
        or preflight.get("status") == "blocked"
        or capture_status == "blocked"
    ):
        return "blocked"
    if dry_run and ratio >= 0.65:
        return "dry_run_demo_ready"
    if not dry_run and capture_status != "ready":
        return "needs_review"
    if not dry_run and ratio >= 0.85:
        return "portfolio_ready_real_run"
    if ratio >= 0.65:
        return "needs_review"
    return "needs_evidence"


def _summary(level: EvidenceLevel, score: int, max_score: int) -> str:
    if level == "portfolio_ready_real_run":
        return "Run has strong real-scene evidence for a portfolio, subject to qualitative review."
    if level == "dry_run_demo_ready":
        return "Run is a useful smoke demo, but it is not real trained NeRF/LERF evidence."
    if level == "needs_review":
        return "Run has useful artifacts but needs review, annotations, or stronger real-run evidence."
    if level == "blocked":
        return "Run has blocker-level issues; fix them before sharing."
    return f"Run evidence is incomplete ({score}/{max_score}); add query, annotation, and report artifacts."


def _top_recommendations(
    criteria: list[EvidenceCriterion],
    recommendations: dict[str, Any],
) -> list[str]:
    items = [
        criterion.recommendation
        for criterion in sorted(criteria, key=lambda item: item.score / max(item.max_score, 1))
        if criterion.recommendation
    ]
    for item in recommendations.get("recommendations") or []:
        if isinstance(item, dict) and item.get("action"):
            items.append(str(item["action"]))
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
        if len(deduped) >= 5:
            break
    return deduped


def _count_overlays(root: Path) -> int:
    patterns = [
        "demo_assets/**/*overlay.png",
        "queries/**/*overlay.png",
        "queries/**/*relevancy.png",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        seen.update(path for path in root.glob(pattern) if path.is_file())
    if (root / "demo_assets" / "query_grid.png").exists():
        seen.add(root / "demo_assets" / "query_grid.png")
    return len(seen)


def _criterion_status(score: int, max_score: int) -> str:
    ratio = score / max(max_score, 1)
    if ratio >= 0.8:
        return "pass"
    if ratio >= 0.5:
        return "warn"
    return "fail"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _criterion_lines(criteria: list[EvidenceCriterion]) -> list[str]:
    if not criteria:
        return ["- None."]
    lines: list[str] = []
    for criterion in criteria:
        lines.append(
            f"- [{criterion.status}] {criterion.category}/{criterion.name}: "
            f"{criterion.score}/{criterion.max_score} - {criterion.detail}"
        )
        if criterion.recommendation:
            lines.append(f"  Recommendation: {criterion.recommendation}")
    return lines


def _markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]
