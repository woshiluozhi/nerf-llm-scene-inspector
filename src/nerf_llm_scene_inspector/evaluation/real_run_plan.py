"""Generate actionable real-scene run plans from existing pipeline artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp
from nerf_llm_scene_inspector.utils.shell import format_command


PlanSeverity = Literal["blocker", "warning", "info"]
StepStatus = Literal["ready", "needs_input", "blocked", "optional"]


@dataclass
class RealRunIssue:
    """One blocker, warning, or informational note for a real run."""

    category: str
    severity: PlanSeverity
    message: str
    action: str
    artifact: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RealRunCommand:
    """One concrete command or manual step in the real-run playbook."""

    phase: str
    name: str
    command: str
    purpose: str
    status: StepStatus = "ready"
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RealRunPlan:
    """Portfolio-oriented plan for upgrading a smoke run into real evidence."""

    run_dir: str
    scene_name: str
    generated_at: str
    current_mode: str
    readiness_level: str
    backend: str
    variant: str
    input_path: str
    input_type: str
    processed_data: str
    require_gpu: bool
    query_count: int
    issues: list[RealRunIssue] = field(default_factory=list)
    commands: list[RealRunCommand] = field(default_factory=list)
    claim_upgrade_path: list[str] = field(default_factory=list)
    evidence_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def blocker_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "blocker")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["blocker_count"] = self.blocker_count
        payload["warning_count"] = self.warning_count
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        payload["commands"] = [command.to_dict() for command in self.commands]
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
            "# Real-Run Action Plan",
            "",
            "This plan turns the current run artifacts into a concrete next-run checklist for a real CUDA/Nerfstudio/LERF scene. It is a planning artifact, not evidence that training has already succeeded.",
            "",
            "## Current State",
            "",
            f"- Scene: `{self.scene_name}`",
            f"- Run directory: `{self.run_dir}`",
            f"- Current mode: `{self.current_mode}`",
            f"- Readiness level: `{self.readiness_level}`",
            f"- Backend target: `{self.backend}` / `{self.variant}`",
            f"- Input target: `{self.input_path}` ({self.input_type})",
            f"- Processed data target: `{self.processed_data}`",
            f"- Require GPU for real run: {self.require_gpu}",
            f"- Query count: {self.query_count}",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Issues To Resolve",
            "",
            *_issue_lines(self.issues),
            "",
            "## Command Playbook",
            "",
            *_command_lines(self.commands),
            "",
            "## Evidence Targets",
            "",
            *_list_lines(self.evidence_targets),
            "",
            "## Claim Upgrade Path",
            "",
            *_list_lines(self.claim_upgrade_path),
            "",
            "## Notes",
            "",
            *_list_lines(self.notes),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_real_run_plan(
    run_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    input_type: str | None = None,
    processed_data: str | Path | None = None,
    backend: str | None = None,
    variant: str | None = None,
    queries_path: str | Path | None = None,
    submission_packet_path: str | Path | None = None,
    output_root: str | Path = "results/pipeline_runs",
    require_gpu: bool = True,
    repo_url: str = "",
) -> RealRunPlan:
    """Build an actionable next-run plan from one pipeline run directory."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    capture = _read_json(root / "capture_manifest.json")
    capture_validation = _read_json(root / "capture_manifest_validation.json")
    preflight = _read_json(root / "preflight_report.json")
    scene = _read_json(root / "scene_data_inspection.json")
    quality = _read_json(root / "quality_gate.json")
    run_audit = _read_json(root / "run_audit.json")
    failure_diagnostics = _read_json(root / "failure_diagnostics.json")
    query_evidence = _read_json(root / "query_evidence_audit.json")
    run_readiness = _read_json(root / "run_readiness.json")
    recommendations = _read_json(root / "run_recommendations.json")
    submission = (
        _read_json(submission_packet_path)
        if submission_packet_path
        else _read_json(root / "submission_packet" / "submission_packet.json")
    )

    scene_name = str(summary.get("scene_name") or capture.get("scene_name") or root.name)
    chosen_backend = str(backend or summary.get("backend") or "lerf")
    chosen_variant = str(variant or _language_variant(root) or "lerf-lite")
    chosen_input = (
        _display_path(Path(input_path))
        if input_path
        else _normalize_path_text(str(capture.get("input_path") or "path/to/video_or_images"))
    )
    chosen_type = str(input_type or capture.get("input_type") or "video")
    chosen_data = (
        _display_path(Path(processed_data))
        if processed_data
        else _normalize_path_text(str(_summary_path(summary, "processed_data") or f"data/processed/{scene_name}"))
    )
    queries = [str(query) for query in summary.get("queries") or []]
    chosen_queries_path = _display_path(Path(queries_path)) if queries_path else _display_path(root / "queries.yaml")
    current_mode = "dry-run smoke demo" if bool(summary.get("dry_run")) else "real run"
    readiness = str(
        submission.get("readiness_level")
        or run_readiness.get("readiness_level")
        or recommendations.get("readiness_level")
        or quality.get("status")
        or "unknown"
    )

    issues = _issues(
        summary=summary,
        capture_validation=capture_validation,
        preflight=preflight,
        scene=scene,
        quality=quality,
        run_audit=run_audit,
        failure_diagnostics=failure_diagnostics,
        query_evidence=query_evidence,
        run_readiness=run_readiness,
        submission=submission,
    )
    commands = _commands(
        scene_name=scene_name,
        input_path=chosen_input,
        input_type=chosen_type,
        processed_data=chosen_data,
        backend=chosen_backend,
        variant=chosen_variant,
        queries_path=chosen_queries_path,
        output_root=_display_path(Path(output_root)),
        require_gpu=require_gpu,
        repo_url=repo_url,
    )
    return RealRunPlan(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        generated_at=utc_timestamp(),
        current_mode=current_mode,
        readiness_level=readiness,
        backend=chosen_backend,
        variant=chosen_variant,
        input_path=chosen_input,
        input_type=chosen_type,
        processed_data=chosen_data,
        require_gpu=require_gpu,
        query_count=len(queries),
        issues=issues,
        commands=commands,
        claim_upgrade_path=_claim_upgrade_path(bool(summary.get("dry_run"))),
        evidence_targets=_evidence_targets(scene_name),
        notes=_notes(require_gpu=require_gpu),
    )


def write_real_run_plan(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    input_path: str | Path | None = None,
    input_type: str | None = None,
    processed_data: str | Path | None = None,
    backend: str | None = None,
    variant: str | None = None,
    queries_path: str | Path | None = None,
    submission_packet_path: str | Path | None = None,
    output_root: str | Path = "results/pipeline_runs",
    require_gpu: bool = True,
    repo_url: str = "",
) -> RealRunPlan:
    """Build and write JSON plus Markdown real-run planning artifacts."""

    root = Path(run_dir)
    output = Path(output_dir) if output_dir else root / "real_run_plan"
    plan = build_real_run_plan(
        root,
        input_path=input_path,
        input_type=input_type,
        processed_data=processed_data,
        backend=backend,
        variant=variant,
        queries_path=queries_path,
        submission_packet_path=submission_packet_path,
        output_root=output_root,
        require_gpu=require_gpu,
        repo_url=repo_url,
    )
    plan.to_json(output / "real_run_plan.json")
    plan.to_markdown(output / "real_run_plan.md")
    return plan


def _issues(
    *,
    summary: dict[str, Any],
    capture_validation: dict[str, Any],
    preflight: dict[str, Any],
    scene: dict[str, Any],
    quality: dict[str, Any],
    run_audit: dict[str, Any],
    failure_diagnostics: dict[str, Any],
    query_evidence: dict[str, Any],
    run_readiness: dict[str, Any],
    submission: dict[str, Any],
) -> list[RealRunIssue]:
    issues: list[RealRunIssue] = []
    if bool(summary.get("dry_run")):
        issues.append(
            RealRunIssue(
                "run_mode",
                "warning",
                "The current artifacts are dry-run smoke evidence.",
                "Run the same pipeline without --dry-run on a CUDA machine before claiming trained-scene results.",
                "pipeline_summary.json",
            )
        )
    if summary and summary.get("success") is not True:
        issues.append(
            RealRunIssue(
                "pipeline",
                "blocker",
                "pipeline_summary.json does not report success.",
                "Debug failed pipeline steps before using this run as a real-run template.",
                "pipeline_summary.json",
            )
        )
    capture_status = str(capture_validation.get("status") or "")
    capture_fail_count = _safe_int(capture_validation.get("fail_count"))
    capture_warn_count = _safe_int(capture_validation.get("warn_count"))
    if not capture_validation:
        issues.append(
            RealRunIssue(
                "capture",
                "warning",
                "Capture manifest validation is missing.",
                "Create or refresh capture_manifest_validation.json before using this run as a real-run template.",
                "capture_manifest_validation.md",
            )
        )
    elif capture_status == "blocked" or capture_fail_count:
        issues.append(
            RealRunIssue(
                "capture",
                "blocker",
                f"Capture manifest validation is blocked or has {capture_fail_count} failed checks.",
                "Fix missing/invalid capture metadata and privacy review before training.",
                "capture_manifest_validation.md",
            )
        )
    elif capture_status and capture_status != "ready":
        issues.append(
            RealRunIssue(
                "capture",
                "warning",
                f"Capture manifest status is {capture_status}.",
                "Record device, lighting, static-scene, overlap, and privacy-review fields.",
                "capture_manifest_validation.md",
            )
        )
    elif capture_warn_count:
        issues.append(
            RealRunIssue(
                "capture",
                "warning",
                f"Capture manifest validation has {capture_warn_count} warning-level checks.",
                "Review capture metadata warnings before spending GPU time.",
                "capture_manifest_validation.md",
            )
        )
    preflight_status = str(preflight.get("status") or "")
    preflight_fail_count = _safe_int(preflight.get("fail_count"))
    preflight_warn_count = _safe_int(preflight.get("warn_count"))
    if preflight_status == "blocked" or preflight_fail_count:
        issues.append(
            RealRunIssue(
                "preflight",
                "blocker",
                f"Real-run preflight reported blocker-level checks or {preflight_fail_count} failures.",
                "Resolve missing upstream tools, GPU, input, processed data, or backend registration.",
                "preflight_report.md",
            )
        )
    elif preflight_status and preflight_status != "ready":
        issues.append(
            RealRunIssue(
                "preflight",
                "warning",
                f"Real-run preflight status is {preflight_status}.",
                "Review preflight warnings before spending GPU time.",
                "preflight_report.md",
            )
        )
    elif preflight_warn_count:
        issues.append(
            RealRunIssue(
                "preflight",
                "warning",
                f"Real-run preflight has {preflight_warn_count} warning-level checks.",
                "Review preflight warnings before spending GPU time.",
                "preflight_report.md",
            )
        )
    if scene and scene.get("ready_for_training") is False:
        issues.append(
            RealRunIssue(
                "scene_data",
                "warning",
                "Processed scene inspection is not ready for training.",
                "Recapture or rerun ns-process-data until frame count and pose coverage pass.",
                "scene_data_inspection.md",
            )
        )
    quality_status = str(quality.get("status") or "")
    quality_fail_count = _safe_int(quality.get("fail_count"))
    quality_warn_count = _safe_int(quality.get("warn_count"))
    if quality_status == "fail" or quality_fail_count:
        issues.append(
            RealRunIssue(
                "quality_gate",
                "blocker",
                f"Quality gate reports a failed profile or {quality_fail_count} failed checks.",
                "Open quality_gate.md and resolve failed criteria before sharing.",
                "quality_gate.md",
            )
        )
    elif quality_status == "warn" or quality_warn_count:
        issues.append(
            RealRunIssue(
                "quality_gate",
                "warning",
                f"Quality gate has warning-level criteria or {quality_warn_count} warnings.",
                "Review quality_gate.md and decide whether each warning is acceptable for a smoke demo or real run.",
                "quality_gate.md",
            )
        )
    audit_status = str(run_audit.get("status") or "")
    audit_blocker_count = _safe_int(run_audit.get("blocker_count"))
    if audit_status == "blocked" or audit_blocker_count:
        issues.append(
            RealRunIssue(
                "run_audit",
                "blocker",
                f"Run audit reports status={audit_status or 'unknown'} with {audit_blocker_count} blockers.",
                "Open run_audit.md and resolve run-level blockers before treating the run as real-run evidence.",
                "run_audit.md",
            )
        )
    elif audit_status and audit_status != "ready":
        issues.append(
            RealRunIssue(
                "run_audit",
                "warning",
                f"Run audit status is {audit_status}.",
                "Review run_audit.md before sharing or rerunning.",
                "run_audit.md",
            )
        )
    diagnostics_status = str(failure_diagnostics.get("status") or "")
    diagnostics_blocker_count = _safe_int(failure_diagnostics.get("blocker_count"))
    diagnostics_warning_count = _safe_int(failure_diagnostics.get("warning_count"))
    if diagnostics_status == "blocked" or diagnostics_blocker_count:
        issues.append(
            RealRunIssue(
                "failure_diagnostics",
                "blocker",
                (
                    f"Failure diagnostics report status={diagnostics_status or 'unknown'} "
                    f"with {diagnostics_blocker_count} blockers."
                ),
                "Open failure_diagnostics.md and resolve classified runtime or artifact failures first.",
                "failure_diagnostics.md",
            )
        )
    elif diagnostics_status and diagnostics_status != "clear":
        issues.append(
            RealRunIssue(
                "failure_diagnostics",
                "warning",
                f"Failure diagnostics status is {diagnostics_status}.",
                "Review failure_diagnostics.md warnings before rerunning or sharing.",
                "failure_diagnostics.md",
            )
        )
    elif diagnostics_warning_count:
        issues.append(
            RealRunIssue(
                "failure_diagnostics",
                "warning",
                f"Failure diagnostics contain {diagnostics_warning_count} warning-level items.",
                "Review failure_diagnostics.md warnings before rerunning or sharing.",
                "failure_diagnostics.md",
            )
        )
    query_status = str(query_evidence.get("status") or "")
    query_risk_count = _query_risk_count(query_evidence)
    if query_status == "fail" or query_risk_count:
        issues.append(
            RealRunIssue(
                "query_evidence",
                "blocker",
                f"Query evidence reports status={query_status or 'unknown'} with {query_risk_count} risk flags.",
                "Resolve overlapping counter-evidence or missing query artifacts before external sharing.",
                "query_evidence_audit.md",
            )
        )
    run_readiness_level = str(run_readiness.get("readiness_level") or "")
    if run_readiness_level == "blocked":
        issues.append(
            RealRunIssue(
                "run_readiness",
                "blocker",
                "Run readiness gate reports blocked.",
                "Open run_readiness.md and resolve failed launch or external-review gates.",
                "run_readiness.md",
            )
        )
    readiness = str(submission.get("readiness_level") or "")
    if readiness in {"blocked", "needs_pack_validation"}:
        issues.append(
            RealRunIssue(
                "sharing",
                "warning" if readiness == "needs_pack_validation" else "blocker",
                f"Submission packet readiness is {readiness}.",
                "Export and validate the portfolio pack, then regenerate the submission packet.",
                "submission_packet/submission_checklist.md",
            )
        )
    if not issues:
        issues.append(
            RealRunIssue(
                "status",
                "info",
                "No blocker-level issues were inferred from the current artifacts.",
                "Proceed through the command playbook and review outputs manually.",
            )
        )
    return issues


def _commands(
    *,
    scene_name: str,
    input_path: str,
    input_type: str,
    processed_data: str,
    backend: str,
    variant: str,
    queries_path: str,
    output_root: str,
    require_gpu: bool,
    repo_url: str,
) -> list[RealRunCommand]:
    require_flag = ["--require-gpu"] if require_gpu else []
    upstream_flag = ["--check-upstream"] if require_gpu else ["--no-check-upstream"]
    run_dir = f"{output_root.rstrip('/')}/{scene_name}"
    capture_dir = f"{run_dir}/capture_manifest_real"
    return [
        RealRunCommand(
            "environment",
            "check_cuda_upstream",
            _cmd(["python", "scripts/check_env.py", *upstream_flag, *require_flag, "--verbose"]),
            "Confirm Python, CUDA/PyTorch, Nerfstudio, LERF, COLMAP, FFmpeg, and backend method registration before training.",
            "blocked" if require_gpu else "ready",
            ["environment_report.json"],
        ),
        RealRunCommand(
            "capture",
            "create_capture_manifest",
            _cmd(
                [
                    "python",
                    "scripts/create_capture_manifest.py",
                    "--input",
                    input_path,
                    "--type",
                    input_type,
                    "--scene-name",
                    scene_name,
                    "--capture-device",
                    "phone model",
                    "--lighting",
                    "bright diffuse indoor",
                    "--camera-motion",
                    "slow orbit with high overlap",
                    "--static-scene",
                    "--high-overlap",
                    "--privacy-reviewed",
                    "--output",
                    capture_dir,
                ]
            ),
            "Record capture conditions that affect COLMAP, NeRF quality, privacy, and reproducibility.",
            "needs_input" if "path/to/" in input_path else "ready",
            [f"{capture_dir}/capture_manifest.json", f"{capture_dir}/capture_manifest_validation.md"],
        ),
        RealRunCommand(
            "preflight",
            "preflight_before_processing",
            _cmd(
                [
                    "python",
                    "scripts/preflight_real_run.py",
                    "--input",
                    input_path,
                    "--type",
                    input_type,
                    "--capture-manifest",
                    f"{capture_dir}/capture_manifest.json",
                    *require_flag,
                ]
            ),
            "Catch missing raw input, capture metadata, GPU, and upstream tools before running COLMAP or training.",
            "needs_input" if "path/to/" in input_path else "ready",
            ["preflight_report.json", "preflight_report.md"],
        ),
        RealRunCommand(
            "processing",
            "prepare_data",
            _cmd(
                [
                    "python",
                    "scripts/prepare_data.py",
                    "--input",
                    input_path,
                    "--output",
                    processed_data,
                    "--type",
                    input_type,
                ]
            ),
            "Run Nerfstudio ns-process-data and validate transforms.json.",
            "needs_input" if "path/to/" in input_path else "ready",
            [f"{processed_data}/transforms.json", f"{processed_data}/scene_inspector_metadata.json"],
        ),
        RealRunCommand(
            "processing",
            "inspect_processed_scene",
            _cmd(["python", "scripts/inspect_scene_data.py", "--data", processed_data, "--output", run_dir]),
            "Review frame count, missing images, camera pose validity, and pose coverage before training.",
            "ready",
            [f"{run_dir}/scene_data_inspection.json", f"{run_dir}/scene_data_inspection.md"],
        ),
        RealRunCommand(
            "training",
            "train_language_pipeline",
            _cmd(
                [
                    "python",
                    "scripts/run_scene_pipeline.py",
                    "--input",
                    input_path,
                    "--scene-name",
                    scene_name,
                    "--type",
                    input_type,
                    "--backend",
                    backend,
                    "--variant",
                    variant,
                    "--queries-file",
                    queries_path,
                    "--capture-manifest",
                    f"{capture_dir}/capture_manifest.json",
                    "--output-root",
                    output_root,
                    "--strict",
                    "--analyze-relations",
                ]
            ),
            "Run the full real pipeline without --dry-run, including baseline training, language-field training, queries, demo assets, evaluation, and reports.",
            "blocked" if require_gpu else "ready",
            [f"{run_dir}/pipeline_summary.json", f"{run_dir}/training/language_train_summary.json"],
        ),
        RealRunCommand(
            "review",
            "create_annotation_workbench",
            _cmd(
                [
                    "python",
                    "scripts/create_annotation_workbench.py",
                    "--annotations",
                    f"{run_dir}/annotation_template.json",
                    "--results",
                    f"{run_dir}/queries",
                    "--output",
                    f"{run_dir}/evaluation/annotation_workbench",
                ]
            ),
            "Create the offline browser workbench used to draw or adjust manual bbox labels.",
            "ready",
            [f"{run_dir}/evaluation/annotation_workbench/annotation_workbench.html"],
        ),
        RealRunCommand(
            "review",
            "finalize_annotations",
            _cmd(
                [
                    "python",
                    "scripts/finalize_annotations.py",
                    "--run-dir",
                    run_dir,
                    "--filled",
                    "path/to/annotations_filled.json",
                    "--profile",
                    "real-run",
                    "--export-pack",
                    "--zip-pack",
                    "--repo-url",
                    repo_url or "https://github.com/woshiluozhi/nerf-llm-scene-inspector",
                ]
            ),
            (
                "Merge browser-edited annotations and refresh evaluation, visual QA, audits, scorecards, "
                "quality gates, reports, result cards, portfolio pages, reproduction bundles, pack validation, "
                "claim audit, and submission materials."
            ),
            "needs_input",
            [
                f"{run_dir}/annotation_finalize_report.md",
                f"{run_dir}/annotations_merged.json",
                f"{run_dir}/evaluation/eval_summary.json",
                f"{run_dir}/evaluation/annotation_review.md",
                f"{run_dir}/quality_gate.md",
                "results/portfolio_pack/portfolio_pack_validation.json",
                f"{run_dir}/submission_packet/submission_checklist.md",
            ],
        ),
    ]


def _claim_upgrade_path(dry_run: bool) -> list[str]:
    prefix = "Current dry-run smoke demo" if dry_run else "Current real-run artifacts"
    return [
        f"{prefix}: use only to show pipeline architecture, artifact format, and reproducibility wiring.",
        "Real-scene run: capture a static, well-lit scene and run Nerfstudio/LERF without --dry-run on a CUDA machine.",
        "Reviewed evidence: inspect overlays, fill annotation boxes, run finalize_annotations.py, and pass the real-run quality gate.",
        "Portfolio-ready claim: share only after portfolio pack validation and submission packet readiness are clean.",
    ]


def _evidence_targets(scene_name: str) -> list[str]:
    run_dir = f"results/pipeline_runs/{scene_name}"
    return [
        f"{run_dir}/training/language_train_summary.json with a real config.yml path.",
        f"{run_dir}/queries/*/scene_query_report.json and overlay images generated from the trained model.",
        f"{run_dir}/evaluation/annotation_review_contact_sheet.png for qualitative localization review.",
        f"{run_dir}/evaluation/eval_summary.json with reviewed annotations for any quantitative metrics.",
        f"{run_dir}/quality_gate.md using profile real-run or portfolio.",
        "results/portfolio_pack/portfolio_pack_index.json after validation.",
    ]


def _notes(*, require_gpu: bool) -> list[str]:
    notes = [
        "Commands assume they are run from the repository root.",
        "Keep real-scene claims separate from dry-run smoke evidence.",
        "Single-scene metrics are portfolio diagnostics, not benchmark results.",
    ]
    if require_gpu:
        notes.append("Full Nerfstudio/LERF training requires an NVIDIA GPU with compatible CUDA/PyTorch.")
    return notes


def _language_variant(root: Path) -> str:
    summary = _read_json(root / "training" / "language_train_summary.json")
    return str(summary.get("variant") or summary.get("method") or "")


def _summary_path(summary: dict[str, Any], key: str) -> str:
    paths = summary.get("paths")
    if isinstance(paths, dict) and paths.get(key):
        return str(paths[key])
    return ""


def _cmd(parts: list[str]) -> str:
    return format_command([part for part in parts if part != ""])


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _query_risk_count(query_evidence: dict[str, Any]) -> int:
    totals = query_evidence.get("totals")
    if isinstance(totals, dict) and totals.get("risk_flag_count") is not None:
        return _safe_int(totals.get("risk_flag_count"))
    tasks = query_evidence.get("tasks")
    if isinstance(tasks, list):
        return sum(_safe_int(task.get("risk_flag_count")) for task in tasks if isinstance(task, dict))
    return _safe_int(query_evidence.get("risk_flag_count"))


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return _display_path(path)


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _normalize_path_text(value: str) -> str:
    return value.replace("\\", "/")


def _issue_lines(issues: list[RealRunIssue]) -> list[str]:
    if not issues:
        return ["- None."]
    return [
        f"- `{issue.severity}` {issue.category}: {issue.message} Action: {issue.action}"
        + (f" Artifact: `{issue.artifact}`" if issue.artifact else "")
        for issue in issues
    ]


def _command_lines(commands: list[RealRunCommand]) -> list[str]:
    if not commands:
        return ["- No commands were generated."]
    lines: list[str] = []
    for index, command in enumerate(commands, start=1):
        lines.extend(
            [
                f"### {index}. {command.phase}: {command.name}",
                "",
                f"- Status: `{command.status}`",
                f"- Purpose: {command.purpose}",
                "",
                "```bash",
                command.command,
                "```",
            ]
        )
        if command.expected_outputs:
            lines.extend(["", "Expected outputs:", *[f"- `{item}`" for item in command.expected_outputs]])
        lines.append("")
    return lines


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]
