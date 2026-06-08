"""Run-level health audit for pipeline outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp

Severity = Literal["blocker", "warning", "info"]
AuditStatus = Literal["ready", "needs_review", "blocked"]


@dataclass
class AuditFinding:
    """One actionable run-audit finding."""

    severity: Severity
    category: str
    message: str
    recommendation: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunAuditReport:
    """Portable health report for one pipeline run directory."""

    status: AuditStatus
    score: int
    run_dir: str
    scene_name: str = ""
    dry_run: bool = False
    backend: str = ""
    pipeline_success: bool = False
    query_count: int = 0
    query_report_count: int = 0
    evaluated_query_count: int = 0
    key_artifacts: dict[str, str] = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    @property
    def blocker_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "blocker")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "score": self.score,
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "pipeline_success": self.pipeline_success,
            "query_count": self.query_count,
            "query_report_count": self.query_report_count,
            "evaluated_query_count": self.evaluated_query_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "key_artifacts": dict(self.key_artifacts),
            "findings": [finding.to_dict() for finding in self.findings],
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
            "# Pipeline Run Audit",
            "",
            f"- Status: {self.status}",
            f"- Score: {self.score}/100",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Backend: {self.backend or 'unknown'}",
            f"- Dry run: {self.dry_run}",
            f"- Pipeline success: {self.pipeline_success}",
            f"- Query reports: {self.query_report_count}/{self.query_count}",
            f"- Evaluated queries: {self.evaluated_query_count}",
            "",
            "## Findings",
            "",
            *_finding_lines(self.findings),
            "",
            "## Key Artifacts",
            "",
            *_artifact_lines(self.key_artifacts),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def audit_pipeline_run(run_dir: str | Path) -> RunAuditReport:
    """Audit a pipeline run directory and return an actionable health report."""

    root = Path(run_dir)
    findings: list[AuditFinding] = []
    pipeline_summary = _read_json(root / "pipeline_summary.json")
    preflight_report = _read_json(root / "preflight_report.json")
    scene_inspection = _read_json(root / "scene_data_inspection.json")
    environment_report = _read_json(root / "environment_report.json")
    annotation_validation = _read_json(root / "evaluation" / "annotation_validation.json")
    eval_summary = _read_json(root / "evaluation" / "eval_summary.json")

    scene_name = str(pipeline_summary.get("scene_name") or scene_inspection.get("scene_name") or "")
    dry_run = bool(pipeline_summary.get("dry_run", False))
    backend = str(pipeline_summary.get("backend") or "")
    pipeline_success = bool(pipeline_summary.get("success", False))
    queries = [str(query) for query in pipeline_summary.get("queries") or []]
    query_report_count = len(list((root / "queries").rglob("scene_query_report.json")))
    evaluated_query_count = _safe_int(eval_summary.get("num_evaluated_queries"))

    _check_required_files(root, pipeline_summary, findings)
    _check_pipeline_summary(pipeline_summary, findings)
    _check_steps(pipeline_summary, findings)
    _check_declared_command_logs(root, pipeline_summary, findings)
    _check_preflight(preflight_report, findings, dry_run=dry_run)
    _check_environment(environment_report, findings, dry_run=dry_run)
    _check_scene_inspection(scene_inspection, findings, dry_run=dry_run)
    _check_training(root, pipeline_summary, findings)
    _check_queries(queries, query_report_count, _step_status(pipeline_summary, "query_scene"), findings)
    _check_annotation_validation(annotation_validation, findings)
    _check_evaluation(eval_summary, findings)

    if dry_run:
        findings.append(
            AuditFinding(
                severity="info",
                category="run_mode",
                message="This is a dry-run artifact. It validates workflow wiring, not trained LERF quality.",
                recommendation="Run on a CUDA machine with Nerfstudio and LERF before presenting quantitative scene results.",
            )
        )

    score = _score(findings)
    status = _status(findings)
    return RunAuditReport(
        status=status,
        score=score,
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        dry_run=dry_run,
        backend=backend,
        pipeline_success=pipeline_success,
        query_count=len(queries),
        query_report_count=query_report_count,
        evaluated_query_count=evaluated_query_count,
        key_artifacts=_key_artifacts(root),
        findings=findings,
    )


def _check_required_files(
    root: Path,
    summary: dict[str, Any],
    findings: list[AuditFinding],
) -> None:
    required = [
        "pipeline_summary.json",
        "preflight_report.json",
        "environment_report.json",
        "scene_data_inspection.json",
        "queries.yaml",
    ]
    if _step_status(summary, "create_annotation_template") == "success":
        required.append("annotation_template.json")
    if _step_status(summary, "evaluate_queries") == "success":
        required.extend(
            [
                "evaluation/annotation_validation.json",
                "evaluation/eval_summary.json",
            ]
        )
    if _step_status(summary, "generate_demo_assets") == "success":
        required.extend(
            [
                "demo_assets/query_grid.png",
                "project_report.md",
                "portfolio_result_card.md",
            ]
        )
    for relative in required:
        if not (root / relative).exists():
            findings.append(
                AuditFinding(
                    severity="blocker",
                    category="missing_artifact",
                    message=f"Missing expected run artifact: {relative}.",
                    recommendation="Rerun scripts/run_scene_pipeline.py or inspect the failed pipeline step.",
                    artifact=relative,
                )
            )


def _check_pipeline_summary(summary: dict[str, Any], findings: list[AuditFinding]) -> None:
    if not summary:
        findings.append(
            AuditFinding(
                severity="blocker",
                category="pipeline",
                message="pipeline_summary.json is missing or unreadable.",
                recommendation="Run scripts/run_scene_pipeline.py before auditing.",
                artifact="pipeline_summary.json",
            )
        )
        return
    if not summary.get("success"):
        findings.append(
            AuditFinding(
                severity="blocker",
                category="pipeline",
                message="Pipeline summary reports success=false.",
                recommendation="Inspect failed step entries and rerun after fixing the root cause.",
                artifact="pipeline_summary.json",
            )
        )


def _check_preflight(
    report: dict[str, Any],
    findings: list[AuditFinding],
    *,
    dry_run: bool,
) -> None:
    if not report:
        return
    status = str(report.get("status") or "")
    if status == "blocked":
        findings.append(
            AuditFinding(
                severity="blocker",
                category="preflight",
                message="Real-run preflight reported blocker-level checks.",
                recommendation="Open preflight_report.md and fix failed environment, input, scene, or config checks.",
                artifact="preflight_report.md",
            )
        )
    elif status == "needs_attention" and not dry_run:
        findings.append(
            AuditFinding(
                severity="warning",
                category="preflight",
                message="Real-run preflight reported checks that need attention.",
                recommendation="Review preflight_report.md before spending GPU time on the run.",
                artifact="preflight_report.md",
            )
        )


def _check_steps(summary: dict[str, Any], findings: list[AuditFinding]) -> None:
    for step in summary.get("steps") or []:
        if not isinstance(step, dict):
            continue
        name = str(step.get("name") or "unknown")
        if name == "audit_run":
            continue
        status = str(step.get("status") or "unknown")
        if status == "failed":
            findings.append(
                AuditFinding(
                    severity="blocker",
                    category="pipeline_step",
                    message=f"Pipeline step failed: {name}.",
                    recommendation=str(step.get("error") or "Inspect the step summary and rerun."),
                    artifact="pipeline_summary.json",
                )
            )
        elif status == "warning":
            findings.append(
                AuditFinding(
                    severity="warning",
                    category="pipeline_step",
                    message=f"Pipeline step completed with warnings: {name}.",
                    recommendation="Inspect the corresponding artifact before using this run as evidence.",
                    artifact="pipeline_summary.json",
                )
            )
        elif status == "skipped" and name in {"query_scene", "generate_demo_assets", "evaluate_queries"}:
            findings.append(
                AuditFinding(
                    severity="warning",
                    category="pipeline_step",
                    message=f"Portfolio-facing step was skipped: {name}.",
                    recommendation="Run this step before treating the run as a complete demo/evaluation package.",
                    artifact="pipeline_summary.json",
                )
            )


def _check_declared_command_logs(
    root: Path,
    summary: dict[str, Any],
    findings: list[AuditFinding],
) -> None:
    for step in summary.get("steps") or []:
        if not isinstance(step, dict):
            continue
        outputs = step.get("outputs") if isinstance(step.get("outputs"), dict) else {}
        raw_log = outputs.get("command_log") if isinstance(outputs, dict) else None
        if not raw_log:
            continue
        log_path = _resolve_declared_log_path(root, str(raw_log))
        if not log_path.exists():
            findings.append(
                AuditFinding(
                    severity="blocker",
                    category="command_logs",
                    message=f"Declared command log is missing for step {step.get('name')}: {raw_log}.",
                    recommendation="Rerun the pipeline so full stdout/stderr logs are preserved.",
                    artifact=str(raw_log),
                )
            )


def _check_environment(
    report: dict[str, Any],
    findings: list[AuditFinding],
    *,
    dry_run: bool,
) -> None:
    if not report:
        return
    strict_failures = [str(name) for name in report.get("strict_failures") or []]
    if strict_failures:
        findings.append(
            AuditFinding(
                severity="blocker" if not dry_run else "warning",
                category="environment",
                message="Required environment checks failed: " + ", ".join(strict_failures),
                recommendation="Run python scripts/check_env.py --check-upstream --require-gpu and follow the hints.",
                artifact="environment_report.json",
            )
        )


def _check_scene_inspection(
    inspection: dict[str, Any],
    findings: list[AuditFinding],
    *,
    dry_run: bool,
) -> None:
    if not inspection:
        return
    if not inspection.get("ready_for_training"):
        findings.append(
            AuditFinding(
                severity="warning" if dry_run else "blocker",
                category="scene_data",
                message="Processed scene is not marked ready for training.",
                recommendation="Review scene_data_inspection.md and recapture/reprocess if pose coverage or frames are weak.",
                artifact="scene_data_inspection.md",
            )
        )
    quality_score = _safe_float(inspection.get("quality_score"))
    if quality_score and quality_score < 0.75:
        findings.append(
            AuditFinding(
                severity="warning",
                category="scene_data",
                message=f"Scene quality score is low: {quality_score:.2f}.",
                recommendation="Use slower capture, more overlap, less blur, and stronger parallax.",
                artifact="scene_data_inspection.json",
            )
        )


def _check_training(root: Path, summary: dict[str, Any], findings: list[AuditFinding]) -> None:
    expected = {
        "train_baseline_nerf": root / "training" / "baseline_train_summary.json",
        "train_language_field": root / "training" / "language_train_summary.json",
    }
    for step_name, path in expected.items():
        if _step_status(summary, step_name) == "success" and not path.exists():
            findings.append(
                AuditFinding(
                    severity="blocker",
                    category="training",
                    message=f"{step_name} succeeded but {path.name} is missing.",
                    recommendation="Rerun the pipeline so training provenance is preserved.",
                    artifact=str(path.relative_to(root)).replace("\\", "/"),
                )
            )


def _check_queries(
    queries: list[str],
    query_report_count: int,
    query_step_status: str,
    findings: list[AuditFinding],
) -> None:
    if query_step_status != "success":
        return
    if queries and query_report_count < len(queries):
        findings.append(
            AuditFinding(
                severity="blocker",
                category="queries",
                message=f"Only {query_report_count} query reports were found for {len(queries)} planned queries.",
                recommendation="Rerun query_scene or inspect backend errors.",
                artifact="queries/",
            )
        )


def _check_annotation_validation(validation: dict[str, Any], findings: list[AuditFinding]) -> None:
    if not validation:
        return
    if validation.get("ok") is False:
        findings.append(
            AuditFinding(
                severity="blocker",
                category="annotations",
                message="Annotation validation failed.",
                recommendation="Open evaluation/annotation_validation.json and fix duplicate queries, invalid boxes, or schema errors.",
                artifact="evaluation/annotation_validation.json",
            )
        )
    for warning in validation.get("warnings") or []:
        findings.append(
            AuditFinding(
                severity="warning",
                category="annotations",
                message=str(warning),
                recommendation="Resolve annotation coverage or view-id mismatches before treating scores as quantitative evidence.",
                artifact="evaluation/annotation_validation.json",
            )
        )


def _check_evaluation(summary: dict[str, Any], findings: list[AuditFinding]) -> None:
    if not summary:
        return
    evaluated = _safe_int(summary.get("num_evaluated_queries"))
    if evaluated == 0:
        findings.append(
            AuditFinding(
                severity="warning",
                category="evaluation",
                message="No bbox-annotated queries were evaluated quantitatively.",
                recommendation="Fill annotation_template.json with manual bbox_2d labels and rerun evaluate_queries.py.",
                artifact="evaluation/eval_summary.json",
            )
        )


def _key_artifacts(root: Path) -> dict[str, str]:
    candidates = {
        "pipeline_summary": "pipeline_summary.json",
        "preflight_report": "preflight_report.md",
        "evidence_scorecard": "evidence_scorecard.md",
        "run_audit": "run_audit.json",
        "command_logs": "logs/",
        "environment_report": "environment_report.json",
        "scene_data_inspection": "scene_data_inspection.md",
        "annotation_validation": "evaluation/annotation_validation.json",
        "evaluation_summary": "evaluation/eval_summary.json",
        "query_grid": "demo_assets/query_grid.png",
        "demo_montage": "demo_assets/demo_montage.gif",
        "portfolio_card": "portfolio_result_card.md",
        "portfolio_page": "portfolio_page.html",
        "project_report": "project_report.md",
    }
    return {name: relative for name, relative in candidates.items() if (root / relative).exists()}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _resolve_declared_log_path(root: Path, raw_log: str) -> Path:
    log_path = Path(raw_log)
    if log_path.is_absolute():
        return log_path
    candidates = [root / log_path, project_root() / log_path, log_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _step_status(summary: dict[str, Any], step_name: str) -> str:
    for step in summary.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == step_name:
            return str(step.get("status") or "")
    return ""


def _score(findings: list[AuditFinding]) -> int:
    score = 100
    for finding in findings:
        if finding.severity == "blocker":
            score -= 30
        elif finding.severity == "warning":
            score -= 8
    return max(0, score)


def _status(findings: list[AuditFinding]) -> AuditStatus:
    if any(finding.severity == "blocker" for finding in findings):
        return "blocked"
    if any(finding.severity == "warning" for finding in findings):
        return "needs_review"
    return "ready"


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


def _finding_lines(findings: list[AuditFinding]) -> list[str]:
    if not findings:
        return ["- None."]
    lines: list[str] = []
    for finding in findings:
        lines.append(f"- [{finding.severity}] {finding.category}: {finding.message}")
        if finding.recommendation:
            lines.append(f"  Recommendation: {finding.recommendation}")
    return lines


def _artifact_lines(artifacts: dict[str, str]) -> list[str]:
    if not artifacts:
        return ["- None."]
    return [f"- {name}: `{path}`" for name, path in artifacts.items()]
