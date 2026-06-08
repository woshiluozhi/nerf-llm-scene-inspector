"""One-page result cards for externally reviewing a pipeline run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


@dataclass
class ResultCardCheck:
    """One reviewer-facing readiness check."""

    name: str
    status: str
    evidence: str
    action: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RunResultCard:
    """Compact, claim-calibrated summary for one run."""

    scene_name: str
    backend: str
    dry_run: bool
    generated_at: str
    result_status: str
    headline: str
    primary_takeaway: str
    shareable_blurb: str
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    demonstrated_capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    do_not_claim: list[str] = field(default_factory=list)
    checks: list[ResultCardCheck] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Run Result Card",
            "",
            self.headline,
            "",
            "## Review Snapshot",
            "",
            f"- Scene: `{self.scene_name}`",
            f"- Backend: `{self.backend}`",
            f"- Run mode: `{'dry-run smoke demo' if self.dry_run else 'real captured-scene run'}`",
            f"- Result status: `{self.result_status}`",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Primary Takeaway",
            "",
            self.primary_takeaway,
            "",
            "## Shareable Blurb",
            "",
            self.shareable_blurb,
            "",
            "## Evidence Snapshot",
            "",
            *_key_value_lines(self.evidence_snapshot),
            "",
            "## Metrics",
            "",
            *_key_value_lines(self.metrics),
            "",
            "## Demonstrated Capabilities",
            "",
            *_list_lines(self.demonstrated_capabilities),
            "",
            "## Limitations",
            "",
            *_list_lines(self.limitations),
            "",
            "## Do Not Claim",
            "",
            *_list_lines(self.do_not_claim),
            "",
            "## Readiness Checks",
            "",
            *_check_lines(self.checks),
            "",
            "## Next Actions",
            "",
            *_list_lines(self.next_actions),
            "",
            "## Artifact Links",
            "",
            *_artifact_lines(self.artifacts),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_run_result_card(run_dir: str | Path) -> RunResultCard:
    """Build a concise result card from existing run artifacts."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    scorecard = _read_json(root / "evidence_scorecard.json")
    quality = _read_json(root / "quality_gate.json")
    audit = _read_json(root / "run_audit.json")
    claim_audit = _read_json(root / "claim_audit.json")
    submission = _read_json(root / "submission_packet" / "submission_packet.json")
    evaluation = _read_json(root / "evaluation" / "eval_summary.json")
    annotations = _read_json(root / "evaluation" / "annotation_validation.json")
    scene = _read_json(root / "scene_data_inspection.json")
    recommendations = _read_json(root / "run_recommendations.json")
    relations = _read_json(root / "scene_relations" / "scene_relations_summary.json")

    scene_name = str(summary.get("scene_name") or scorecard.get("scene_name") or root.name)
    backend = str(summary.get("backend") or scorecard.get("backend") or "unknown")
    dry_run = bool(summary.get("dry_run", scorecard.get("dry_run", False)))
    status = _result_status(
        dry_run=dry_run,
        success=summary.get("success") is True,
        evidence_level=str(scorecard.get("evidence_level") or ""),
        quality_status=str(quality.get("status") or ""),
        claim_status=str(claim_audit.get("status") or ""),
        readiness=str(submission.get("readiness_level") or ""),
    )
    metrics = _metrics(evaluation, scorecard, scene, relations)
    evidence_snapshot = _evidence_snapshot(summary, scorecard, quality, audit, claim_audit, submission, annotations)
    limitations = _limitations(dry_run, evaluation, annotations, relations)
    next_actions = _next_actions(submission, recommendations, dry_run)
    do_not_claim = _do_not_claim(submission, dry_run)
    checks = _checks(summary, scorecard, quality, audit, claim_audit, submission, annotations)
    return RunResultCard(
        scene_name=scene_name,
        backend=backend,
        dry_run=dry_run,
        generated_at=utc_timestamp(),
        result_status=status,
        headline=_headline(scene_name, backend, status, dry_run),
        primary_takeaway=_primary_takeaway(status, dry_run, scorecard, evaluation),
        shareable_blurb=_shareable_blurb(scene_name, backend, dry_run, status),
        evidence_snapshot=evidence_snapshot,
        metrics=metrics,
        demonstrated_capabilities=_capabilities(summary, scorecard, relations),
        limitations=limitations,
        next_actions=next_actions,
        do_not_claim=do_not_claim,
        checks=checks,
        artifacts=_artifacts(root),
    )


def write_run_result_card(
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    json_output: str | Path | None = None,
) -> RunResultCard:
    """Build and write JSON plus Markdown run result card artifacts."""

    root = Path(run_dir)
    card = build_run_result_card(root)
    card.to_json(json_output or root / "run_result_card.json")
    card.to_markdown(output or root / "run_result_card.md")
    return card


def _result_status(
    *,
    dry_run: bool,
    success: bool,
    evidence_level: str,
    quality_status: str,
    claim_status: str,
    readiness: str,
) -> str:
    if not success or quality_status == "fail" or claim_status == "fail":
        return "blocked"
    if readiness == "portfolio_ready" or (not dry_run and evidence_level == "portfolio_ready_real_run"):
        return "portfolio_ready"
    if not dry_run and readiness in {"real_run_review_ready", "shareable_smoke_demo"}:
        return "real_run_review_ready"
    if dry_run and readiness in {"shareable_smoke_demo", "needs_pack_validation"}:
        return "shareable_smoke_demo"
    if dry_run or evidence_level == "dry_run_demo_ready":
        return "dry_run_smoke_demo"
    return "needs_evidence"


def _headline(scene_name: str, backend: str, status: str, dry_run: bool) -> str:
    mode = "CPU-safe dry-run" if dry_run else "real-scene"
    return f"**{scene_name}** is a {mode} NeRF-LLM Scene Inspector run using `{backend}` with status `{status}`."


def _primary_takeaway(
    status: str,
    dry_run: bool,
    scorecard: dict[str, Any],
    evaluation: dict[str, Any],
) -> str:
    score = _score_text(scorecard.get("score"), scorecard.get("max_score"))
    if status == "portfolio_ready":
        return f"The run has enough recorded evidence for portfolio sharing, with evidence score {score}."
    if status == "real_run_review_ready":
        return "The run has real-mode artifacts but still needs qualitative review before strong external claims."
    if dry_run:
        return (
            f"The run demonstrates the full artifact and evaluation workflow with evidence score {score}; "
            "it does not prove trained NeRF/LERF scene understanding quality."
        )
    evaluated = evaluation.get("num_evaluated_queries")
    return f"The run is structurally present but needs stronger evidence or review; evaluated queries: {evaluated}."


def _shareable_blurb(scene_name: str, backend: str, dry_run: bool, status: str) -> str:
    if dry_run:
        return (
            f"For `{scene_name}`, I generated a CPU-only smoke run of a Nerfstudio/LERF-style "
            f"open-vocabulary 3D scene inspection pipeline using `{backend}`. The artifacts show "
            "reproducible orchestration, query reports, visualizations, evaluation scaffolding, "
            "and sharing gates; real trained-scene claims require a CUDA-backed run."
        )
    qualifier = "portfolio-ready" if status == "portfolio_ready" else "review-ready"
    return (
        f"For `{scene_name}`, I ran a {qualifier} open-vocabulary 3D scene inspection workflow "
        f"with `{backend}`, preserving training/query/evaluation artifacts and claim-calibrated "
        "reports for review."
    )


def _evidence_snapshot(
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    quality: dict[str, Any],
    audit: dict[str, Any],
    claim_audit: dict[str, Any],
    submission: dict[str, Any],
    annotations: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_success": summary.get("success"),
        "evidence_level": scorecard.get("evidence_level"),
        "evidence_score": _score_text(scorecard.get("score"), scorecard.get("max_score")),
        "quality_gate": quality.get("status"),
        "run_audit": audit.get("status"),
        "claim_audit": claim_audit.get("status"),
        "submission_readiness": submission.get("readiness_level"),
        "annotation_validation_ok": annotations.get("ok"),
        "query_count": len(summary.get("queries") or []),
    }


def _metrics(
    evaluation: dict[str, Any],
    scorecard: dict[str, Any],
    scene: dict[str, Any],
    relations: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        "top_k_hit_rate": evaluation.get("top_k_hit_rate"),
        "mean_iou_2d": evaluation.get("mean_iou_2d"),
        "semantic_success_rate": evaluation.get("semantic_success_rate"),
        "average_relevancy_score": evaluation.get("average_relevancy_score"),
        "num_evaluated_queries": evaluation.get("num_evaluated_queries"),
        "num_bbox_annotated_queries": evaluation.get("num_bbox_annotated_queries"),
        "scene_quality_score": scene.get("quality_score"),
        "pose_coverage_score": scene.get("pose_coverage_score"),
        "scene_relation_entities": relations.get("num_entities"),
        "scene_relation_edges": relations.get("num_relations"),
    }
    score_metrics = scorecard.get("metrics") if isinstance(scorecard.get("metrics"), dict) else {}
    for key in ("preflight_status", "capture_manifest_status"):
        if key in score_metrics:
            metrics[key] = score_metrics[key]
    return {key: value for key, value in metrics.items() if value not in {None, ""}}


def _capabilities(
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    relations: dict[str, Any],
) -> list[str]:
    capabilities = [
        "Nerfstudio/LERF-style pipeline orchestration with typed run artifacts.",
        "Open-vocabulary query reports and visual evidence packaging.",
        "Annotation validation, evaluation summaries, and external-sharing gates.",
    ]
    if scorecard.get("overlay_count", 0):
        capabilities.append("RGB/relevancy overlay artifacts for qualitative review.")
    if relations:
        capabilities.append("Heuristic scene-relation extraction from query evidence.")
    if summary.get("dry_run") is not True:
        capabilities.append("Real-mode command path for upstream training/query execution.")
    return capabilities


def _limitations(
    dry_run: bool,
    evaluation: dict[str, Any],
    annotations: dict[str, Any],
    relations: dict[str, Any],
) -> list[str]:
    limitations = [
        "This is a research engineering system built on upstream Nerfstudio and LERF components.",
        "Single-scene portfolio metrics are not benchmark results.",
    ]
    if dry_run:
        limitations.append("Dry-run outputs are synthetic and validate workflow wiring only.")
    if not evaluation or not evaluation.get("num_bbox_annotated_queries"):
        limitations.append("Localization metrics need manual bbox annotations before quantitative claims.")
    if annotations.get("warnings"):
        limitations.append("Annotation warnings should be resolved before reporting numeric localization performance.")
    if relations:
        limitations.append("Scene-relation outputs are heuristic and should be treated as qualitative evidence.")
    return limitations


def _next_actions(
    submission: dict[str, Any],
    recommendations: dict[str, Any],
    dry_run: bool,
) -> list[str]:
    actions = [str(item) for item in submission.get("next_actions") or [] if str(item).strip()]
    if not actions:
        actions = [
            str(item.get("action"))
            for item in recommendations.get("recommendations") or []
            if isinstance(item, dict) and item.get("action")
        ]
    if dry_run:
        actions.insert(0, "Run the same pipeline without --dry-run on a CUDA machine with Nerfstudio/LERF installed.")
    deduped: list[str] = []
    for action in actions:
        if action and action not in deduped:
            deduped.append(action)
        if len(deduped) >= 6:
            break
    return deduped or ["Review run artifacts and rerun quality gates."]


def _do_not_claim(submission: dict[str, Any], dry_run: bool) -> list[str]:
    claims = [str(item) for item in submission.get("avoid_claims") or [] if str(item).strip()]
    if not claims:
        claims = [
            "Do not claim a new NeRF architecture.",
            "Do not claim state-of-the-art segmentation, detection, or 3D grounding performance.",
            "Do not present single-scene metrics as benchmark results.",
        ]
    if dry_run and not any("dry-run" in item.lower() for item in claims):
        claims.append("Do not describe dry-run overlays as trained LERF outputs from a real scene.")
    return claims


def _checks(
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    quality: dict[str, Any],
    audit: dict[str, Any],
    claim_audit: dict[str, Any],
    submission: dict[str, Any],
    annotations: dict[str, Any],
) -> list[ResultCardCheck]:
    return [
        ResultCardCheck(
            "pipeline_success",
            "pass" if summary.get("success") is True else "fail",
            f"pipeline_summary.success={summary.get('success')}",
            "Rerun or debug pipeline_summary.json." if summary.get("success") is not True else "",
            "pipeline_summary.json",
        ),
        ResultCardCheck(
            "evidence_scorecard",
            "pass" if scorecard.get("evidence_level") in {"dry_run_demo_ready", "portfolio_ready_real_run"} else "warn",
            f"level={scorecard.get('evidence_level')}, score={_score_text(scorecard.get('score'), scorecard.get('max_score'))}",
            "Review evidence_scorecard.md recommendations." if scorecard.get("evidence_level") not in {"dry_run_demo_ready", "portfolio_ready_real_run"} else "",
            "evidence_scorecard.md",
        ),
        ResultCardCheck(
            "quality_gate",
            "pass" if quality.get("status") == "pass" else "warn" if quality.get("passed") is True else "fail",
            f"profile={quality.get('profile')}, status={quality.get('status')}, passed={quality.get('passed')}",
            "Review or fix quality_gate.md criteria." if quality.get("status") != "pass" else "",
            "quality_gate.md",
        ),
        ResultCardCheck(
            "run_audit",
            "pass" if audit.get("status") == "ready" else "warn" if audit.get("status") == "needs_review" else "fail",
            f"status={audit.get('status')}, score={audit.get('score')}",
            "Review run_audit.md warnings before sharing." if audit.get("status") != "ready" else "",
            "run_audit.md",
        ),
        ResultCardCheck(
            "claim_audit",
            "pass" if claim_audit.get("status") == "pass" else "warn" if claim_audit.get("status") == "warn" else "fail",
            f"status={claim_audit.get('status')}, fails={claim_audit.get('fail_count', 0)}, warnings={claim_audit.get('warn_count', 0)}",
            "Fix unsupported claims before outreach." if claim_audit.get("status") == "fail" else "",
            "claim_audit.md",
        ),
        ResultCardCheck(
            "submission_packet",
            "pass" if submission.get("readiness_level") in {"shareable_smoke_demo", "portfolio_ready"} else "warn",
            f"readiness={submission.get('readiness_level')}",
            "Regenerate submission packet with a validated pack." if not submission.get("readiness_level") else "",
            "submission_packet/submission_checklist.md",
        ),
        ResultCardCheck(
            "annotations",
            "pass" if annotations.get("ok") is True and not annotations.get("warnings") else "warn",
            f"ok={annotations.get('ok')}, warnings={len(annotations.get('warnings') or [])}",
            "Resolve annotation warnings before reporting quantitative localization metrics." if annotations.get("warnings") else "",
            "evaluation/annotation_validation.json",
        ),
    ]


def _artifacts(root: Path) -> dict[str, str]:
    candidates = {
        "pipeline_summary": "pipeline_summary.json",
        "portfolio_page": "portfolio_page.html",
        "research_report": "research_report.md",
        "evidence_scorecard": "evidence_scorecard.md",
        "quality_gate": "quality_gate.md",
        "claim_audit": "claim_audit.md",
        "submission_checklist": "submission_packet/submission_checklist.md",
        "real_run_plan": "real_run_plan/real_run_plan.md",
        "reproduction_report": "reproduction_report.md",
        "query_grid": "demo_assets/query_grid.png",
        "annotation_review": "evaluation/annotation_review.md",
        "annotation_finalize": "annotation_finalize_report.md",
        "scene_relations": "scene_relations/scene_relations_report.md",
    }
    return {name: path for name, path in candidates.items() if (root / path).exists()}


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _score_text(value: Any, maximum: Any) -> str:
    if value is None or maximum in {None, 0, "0"}:
        return ""
    return f"{value}/{maximum}"


def _key_value_lines(payload: dict[str, Any]) -> list[str]:
    if not payload:
        return ["- None."]
    return [f"- {key}: `{_format_value(value)}`" for key, value in payload.items()]


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _check_lines(checks: list[ResultCardCheck]) -> list[str]:
    if not checks:
        return ["- None."]
    lines: list[str] = []
    for check in checks:
        lines.append(f"- [{check.status}] {check.name}: {check.evidence}")
        if check.action:
            lines.append(f"  Action: {check.action}")
        if check.artifact:
            lines.append(f"  Artifact: `{check.artifact}`")
    return lines


def _artifact_lines(artifacts: dict[str, str]) -> list[str]:
    if not artifacts:
        return ["- None."]
    return [f"- {name}: `{path}`" for name, path in artifacts.items()]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
