"""Actionable next-step recommendations for a pipeline run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp

RecommendationSeverity = Literal["critical", "high", "medium", "low"]
ReadinessLevel = Literal[
    "blocked",
    "dry_run_ready_for_smoke_demo",
    "needs_review",
    "ready_for_portfolio",
]


@dataclass
class RecommendationItem:
    """One concrete next action for improving a run."""

    severity: RecommendationSeverity
    category: str
    action: str
    rationale: str
    command: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunRecommendationReport:
    """Action plan derived from run audit, environment, scene, and evaluation artifacts."""

    run_dir: str
    scene_name: str
    readiness_level: ReadinessLevel
    summary: str
    top_next_action: str
    recommendations: list[RecommendationItem] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    @property
    def critical_count(self) -> int:
        return sum(1 for item in self.recommendations if item.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for item in self.recommendations if item.severity == "high")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "readiness_level": self.readiness_level,
            "summary": self.summary,
            "top_next_action": self.top_next_action,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "recommendations": [item.to_dict() for item in self.recommendations],
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
            "# Run Recommendations",
            "",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Readiness: {self.readiness_level}",
            f"- Summary: {self.summary}",
            f"- Top next action: {self.top_next_action or 'None'}",
            "",
            "## Actions",
            "",
            *_recommendation_lines(self.recommendations),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_run_recommendations(run_dir: str | Path) -> RunRecommendationReport:
    """Build a concrete action plan for one pipeline run directory."""

    root = Path(run_dir)
    pipeline_summary = _read_json(root / "pipeline_summary.json")
    run_audit = _read_json(root / "run_audit.json")
    query_evidence_audit = _read_json(root / "query_evidence_audit.json")
    capture_validation = _read_json(root / "capture_manifest_validation.json")
    preflight_report = _read_json(root / "preflight_report.json")
    failure_diagnostics = _read_json(root / "failure_diagnostics.json")
    environment_report = _read_json(root / "environment_report.json")
    scene_inspection = _read_json(root / "scene_data_inspection.json")
    annotation_validation = _read_json(root / "evaluation" / "annotation_validation.json")
    eval_summary = _read_json(root / "evaluation" / "eval_summary.json")

    scene_name = str(pipeline_summary.get("scene_name") or root.name)
    dry_run = bool(pipeline_summary.get("dry_run"))
    recommendations: list[RecommendationItem] = []

    _add_audit_findings(run_audit, recommendations)
    _add_query_evidence_actions(query_evidence_audit, recommendations)
    _add_failure_diagnostics_actions(failure_diagnostics, recommendations)
    _add_capture_manifest_actions(capture_validation, recommendations, dry_run=dry_run)
    _add_preflight_actions(preflight_report, recommendations, dry_run=dry_run)
    _add_environment_actions(environment_report, recommendations, dry_run=dry_run)
    _add_scene_actions(scene_inspection, recommendations, dry_run=dry_run)
    _add_annotation_actions(annotation_validation, eval_summary, recommendations)
    _add_pipeline_step_actions(pipeline_summary, recommendations)
    _add_run_mode_actions(scene_name, dry_run, recommendations)
    _add_export_action(root, run_audit, recommendations)
    recommendations = _dedupe_recommendations(recommendations)
    readiness = _readiness_level(recommendations, dry_run=dry_run, audit_status=str(run_audit.get("status") or ""))
    summary = _summary(readiness, recommendations)
    top_next_action = recommendations[0].action if recommendations else "Run is ready to present as a portfolio artifact."
    return RunRecommendationReport(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        readiness_level=readiness,
        summary=summary,
        top_next_action=top_next_action,
        recommendations=recommendations,
    )


def _add_audit_findings(audit: dict[str, Any], recommendations: list[RecommendationItem]) -> None:
    for finding in audit.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        severity = str(finding.get("severity") or "")
        if severity not in {"blocker", "warning"}:
            continue
        if severity == "warning" and str(finding.get("category") or "") in {"annotations", "evaluation"}:
            continue
        recommendations.append(
            RecommendationItem(
                severity="critical" if severity == "blocker" else "medium",
                category=str(finding.get("category") or "audit"),
                action=str(finding.get("recommendation") or finding.get("message") or "Review run audit finding."),
                rationale=str(finding.get("message") or "Run audit reported a finding."),
                artifact=str(finding.get("artifact") or "run_audit.json"),
            )
        )


def _add_query_evidence_actions(
    audit: dict[str, Any],
    recommendations: list[RecommendationItem],
) -> None:
    if not audit:
        return
    status = str(audit.get("status") or "")
    failed = _safe_int(audit.get("fail_count"))
    warned = _safe_int(audit.get("warn_count"))
    tasks = audit.get("tasks") if isinstance(audit.get("tasks"), list) else []
    totals = audit.get("totals") if isinstance(audit.get("totals"), dict) else {}
    counter_evidence_count = _safe_int(totals.get("counter_evidence_count"))
    risk_flag_count = _safe_int(totals.get("risk_flag_count"))
    if not counter_evidence_count:
        counter_evidence_count = sum(
            _safe_int(task.get("counter_evidence_count"))
            for task in tasks
            if isinstance(task, dict)
        )
    if not risk_flag_count:
        risk_flag_count = sum(
            _safe_int(task.get("risk_flag_count"))
            for task in tasks
            if isinstance(task, dict)
        )
    if status not in {"fail", "warn"} and audit.get("ok") is not False and not (
        counter_evidence_count or risk_flag_count
    ):
        return
    weak_modes = sorted(
        {
            str(task.get("evidence_mode") or "")
            for task in tasks
            if isinstance(task, dict) and str(task.get("evidence_mode") or "") in {"2d_fallback", "render_only", "missing"}
        }
    )
    recommendations.append(
        RecommendationItem(
            severity="critical" if status == "fail" or audit.get("ok") is False else "medium",
            category="query_evidence",
            action=(
                "Repair missing query artifacts before sharing this run."
                if status == "fail" or audit.get("ok") is False
                else "Review query fallback modes and missing visual evidence before making scene-understanding claims."
            ),
            rationale=(
                f"query_evidence_audit.json reports status={status}, failed={failed}, "
                f"warnings={warned}, modes={', '.join(weak_modes) or 'none'}."
            ),
            command="python scripts/audit_query_evidence.py --run-dir results/pipeline_runs/<scene>",
            artifact="query_evidence_audit.md",
        )
    )
    if counter_evidence_count or risk_flag_count:
        recommendations.append(
            RecommendationItem(
                severity="high" if risk_flag_count else "medium",
                category="query_evidence",
                action="Review query counter-evidence and risk flags before using scene answers for physical-action decisions.",
                rationale=(
                    f"query_evidence_audit.json reports counter_evidence={counter_evidence_count}, "
                    f"risk_flags={risk_flag_count}."
                ),
                command="python scripts/audit_query_evidence.py --run-dir results/pipeline_runs/<scene>",
                artifact="query_evidence_audit.md",
            )
        )


def _add_failure_diagnostics_actions(
    diagnostics: dict[str, Any],
    recommendations: list[RecommendationItem],
) -> None:
    for item in diagnostics.get("diagnostics") or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "")
        if severity not in {"blocker", "warning"}:
            continue
        recommendations.append(
            RecommendationItem(
                severity="critical" if severity == "blocker" else "high",
                category=str(item.get("category") or "failure_diagnostics"),
                action=str(item.get("recommendation") or "Inspect failure_diagnostics.md."),
                rationale=str(item.get("message") or "Failure diagnostics reported an issue."),
                command=str(item.get("command") or ""),
                artifact=str(item.get("artifact") or "failure_diagnostics.md"),
            )
        )


def _add_preflight_actions(
    report: dict[str, Any],
    recommendations: list[RecommendationItem],
    *,
    dry_run: bool,
) -> None:
    status = str(report.get("status") or "")
    if not status or status == "ready":
        return
    failed = _preflight_check_names(report, "fail")
    warned = _preflight_check_names(report, "warn")
    if failed:
        recommendations.append(
            RecommendationItem(
                severity="high" if dry_run else "critical",
                category="preflight",
                action="Fix failed real-run preflight checks before launching training.",
                rationale="Failed checks: " + ", ".join(failed[:6]),
                command="python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --data data/processed/<scene> --require-gpu",
                artifact="preflight_report.md",
            )
        )
    elif warned and not dry_run:
        recommendations.append(
            RecommendationItem(
                severity="medium",
                category="preflight",
                action="Review warning-level preflight checks before spending GPU time.",
                rationale="Warning checks: " + ", ".join(warned[:6]),
                command="python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --data data/processed/<scene>",
                artifact="preflight_report.md",
            )
        )


def _add_capture_manifest_actions(
    validation: dict[str, Any],
    recommendations: list[RecommendationItem],
    *,
    dry_run: bool,
) -> None:
    status = str(validation.get("status") or "")
    if not status or status == "ready":
        return
    fail_count = _safe_int(validation.get("fail_count"))
    warn_count = _safe_int(validation.get("warn_count"))
    if status == "blocked" or fail_count:
        recommendations.append(
            RecommendationItem(
                severity="critical",
                category="capture_manifest",
                action="Fix blocked capture-manifest checks before treating this run as reproducible evidence.",
                rationale=(
                    f"capture_manifest_validation.json reports status={status}, "
                    f"failures={fail_count}, warnings={warn_count}."
                ),
                command=(
                    "python scripts/create_capture_manifest.py --input path/to/video.mp4 "
                    "--type video --scene-name <scene> --output results/capture_manifest "
                    "--capture-device \"phone model\" --static-scene --high-overlap --privacy-reviewed"
                ),
                artifact="capture_manifest_validation.md",
            )
        )
    elif status == "needs_review":
        recommendations.append(
            RecommendationItem(
                severity="medium" if dry_run else "high",
                category="capture_manifest",
                action="Complete capture metadata and privacy review before sharing real-scene results.",
                rationale=(
                    f"capture_manifest_validation.json reports {warn_count} warning-level capture checks."
                ),
                command=(
                    "python scripts/create_capture_manifest.py --input path/to/video.mp4 "
                    "--type video --scene-name <scene> --output results/capture_manifest "
                    "--capture-device \"phone model\" --lighting \"bright indoor\" "
                    "--static-scene --high-overlap --privacy-reviewed"
                ),
                artifact="capture_manifest_validation.md",
            )
        )


def _add_environment_actions(
    report: dict[str, Any],
    recommendations: list[RecommendationItem],
    *,
    dry_run: bool,
) -> None:
    failures = [str(name) for name in report.get("strict_failures") or []]
    if failures:
        recommendations.append(
            RecommendationItem(
                severity="high" if dry_run else "critical",
                category="environment",
                action="Install or fix required upstream runtime dependencies before real training.",
                rationale="Environment checks reported: " + ", ".join(failures),
                command="python scripts/check_env.py --check-upstream --require-gpu --verbose",
                artifact="environment_report.json",
            )
        )


def _add_scene_actions(
    inspection: dict[str, Any],
    recommendations: list[RecommendationItem],
    *,
    dry_run: bool,
) -> None:
    if not inspection:
        return
    if inspection.get("ready_for_training") is False:
        recommendations.append(
            RecommendationItem(
                severity="high" if dry_run else "critical",
                category="scene_capture",
                action="Recapture or reprocess the scene until pose coverage passes readiness checks.",
                rationale="The processed scene is not marked ready for training.",
                command="python scripts/inspect_scene_data.py --data data/processed/<scene> --min-frames 50 --min-pose-extent 0.05",
                artifact="scene_data_inspection.md",
            )
        )
    quality = _safe_float(inspection.get("quality_score"))
    if 0 < quality < 0.75:
        recommendations.append(
            RecommendationItem(
                severity="medium",
                category="scene_capture",
                action="Improve capture quality with slower motion, more overlap, stronger parallax, and less blur.",
                rationale=f"Scene quality score is {quality:.2f}.",
                artifact="scene_data_inspection.json",
            )
        )


def _add_annotation_actions(
    validation: dict[str, Any],
    evaluation: dict[str, Any],
    recommendations: list[RecommendationItem],
) -> None:
    if validation.get("ok") is False:
        recommendations.append(
            RecommendationItem(
                severity="critical",
                category="annotations",
                action="Fix invalid annotations before treating evaluation results as evidence.",
                rationale="Annotation validation failed.",
                command="python scripts/validate_annotations.py --annotations results/annotations_template.json --results results/query_outputs",
                artifact="evaluation/annotation_validation.json",
            )
        )
    warnings = [str(item) for item in validation.get("warnings") or []]
    if warnings:
        recommendations.append(
            RecommendationItem(
                severity="medium",
                category="annotations",
                action="Resolve annotation coverage and view-id warnings before reporting quantitative scores.",
                rationale="Annotation validation warnings: " + "; ".join(warnings[:3]),
                artifact="evaluation/annotation_validation.json",
            )
        )
    if _safe_int(evaluation.get("num_bbox_annotated_queries")) == 0 or _safe_int(
        evaluation.get("num_evaluated_queries")
    ) == 0:
        recommendations.append(
            RecommendationItem(
                severity="medium",
                category="evaluation",
                action=(
                    "Add manual bbox_2d annotations with the workbench, then run "
                    "finalize_annotations.py to refresh evaluation and reporting artifacts."
                ),
                rationale="No bbox-annotated queries were evaluated quantitatively.",
                command=(
                    "python scripts/finalize_annotations.py --run-dir results/pipeline_runs/<scene> "
                    "--filled path/to/annotations_filled.json --profile real-run --export-pack --zip-pack"
                ),
                artifact="results/pipeline_runs/<scene>/annotation_finalize_report.md",
            )
        )


def _add_pipeline_step_actions(summary: dict[str, Any], recommendations: list[RecommendationItem]) -> None:
    steps = [step for step in summary.get("steps") or [] if isinstance(step, dict)]
    failed_steps = [step for step in steps if str(step.get("status") or "") == "failed"]
    if summary.get("success") is False and not failed_steps:
        recommendations.append(
            RecommendationItem(
                severity="critical",
                category="pipeline",
                action="Inspect pipeline_summary.json and rerun the failed or stale pipeline stages.",
                rationale="pipeline_summary.json reports success=false but does not list an explicit failed step.",
                artifact="pipeline_summary.json",
            )
        )
    for step in steps:
        name = str(step.get("name") or "")
        status = str(step.get("status") or "")
        if status == "failed":
            recommendations.append(
                RecommendationItem(
                    severity="critical",
                    category="pipeline",
                    action=f"Fix the failed pipeline step and rerun it: {name}.",
                    rationale=str(step.get("error") or "Pipeline step failed."),
                    artifact="pipeline_summary.json",
                )
            )
        elif status == "skipped" and name in {"query_scene", "generate_demo_assets", "evaluate_queries"}:
            recommendations.append(
                RecommendationItem(
                    severity="high",
                    category="pipeline",
                    action=f"Run the skipped portfolio-facing step: {name}.",
                    rationale="Complete query, demo, and evaluation steps are needed for a CV-ready artifact.",
                    command="python scripts/run_scene_pipeline.py --dry-run --query mug",
                    artifact="pipeline_summary.json",
                )
            )


def _add_run_mode_actions(
    scene_name: str,
    dry_run: bool,
    recommendations: list[RecommendationItem],
) -> None:
    if dry_run:
        recommendations.append(
            RecommendationItem(
                severity="high",
                category="run_mode",
                action="Run the same pipeline on a real captured scene with Nerfstudio/LERF installed on an NVIDIA GPU machine.",
                rationale="Dry-run artifacts verify wiring only; they are not trained NeRF/LERF evidence.",
                command=(
                    "python scripts/run_scene_pipeline.py --input path/to/video.mp4 "
                    f"--scene-name {scene_name} --type video --backend lerf --variant lerf-lite --strict"
                ),
                artifact="pipeline_summary.json",
            )
        )


def _add_export_action(
    root: Path,
    audit: dict[str, Any],
    recommendations: list[RecommendationItem],
) -> None:
    if audit.get("status") == "blocked":
        return
    recommendations.append(
        RecommendationItem(
            severity="low",
            category="portfolio_export",
            action="Finalize annotations and export a validated portfolio pack after reviewing warnings.",
            rationale=(
                "The finalizer refreshes evaluation, reporting, quality checks, submission materials, "
                "pack validation, and the final shareable zip in one run-scoped command."
            ),
            command=(
                f"python scripts/finalize_annotations.py --run-dir {root} "
                "--filled path/to/annotations_filled.json --profile real-run --export-pack --zip-pack"
            ),
            artifact="portfolio_pack_index.json",
        )
    )


def _readiness_level(
    recommendations: list[RecommendationItem],
    *,
    dry_run: bool,
    audit_status: str,
) -> ReadinessLevel:
    if audit_status == "blocked" or any(item.severity == "critical" for item in recommendations):
        return "blocked"
    if dry_run:
        return "dry_run_ready_for_smoke_demo"
    if audit_status == "needs_review" or any(item.severity in {"high", "medium"} for item in recommendations):
        return "needs_review"
    return "ready_for_portfolio"


def _summary(readiness: ReadinessLevel, recommendations: list[RecommendationItem]) -> str:
    if readiness == "blocked":
        return "Run has blocker-level issues that must be fixed before it is useful as evidence."
    if readiness == "dry_run_ready_for_smoke_demo":
        return "Run is useful as a smoke demo, but still needs a real GPU-backed scene run for research evidence."
    if readiness == "needs_review":
        return "Run is structurally complete but needs review or annotation cleanup before portfolio use."
    if recommendations:
        return "Run is ready; optional recommendations remain for packaging or presentation."
    return "Run is ready for portfolio presentation."


def _dedupe_recommendations(items: list[RecommendationItem]) -> list[RecommendationItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[RecommendationItem] = []
    for item in sorted(items, key=_recommendation_sort_key):
        key = (item.category, item.action)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _recommendation_sort_key(item: RecommendationItem) -> tuple[int, str, str]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return order[item.severity], item.category, item.action


def _recommendation_lines(items: list[RecommendationItem]) -> list[str]:
    if not items:
        return ["- None."]
    lines: list[str] = []
    for item in items:
        lines.append(f"- [{item.severity}] {item.category}: {item.action}")
        lines.append(f"  Rationale: {item.rationale}")
        if item.command:
            lines.append(f"  Command: `{item.command}`")
        if item.artifact:
            lines.append(f"  Artifact: `{item.artifact}`")
    return lines


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _preflight_check_names(report: dict[str, Any], status: str) -> list[str]:
    names: list[str] = []
    for check in report.get("checks") or []:
        if isinstance(check, dict) and check.get("status") == status:
            names.append(str(check.get("name") or check.get("category") or "unknown"))
    return names


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


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
