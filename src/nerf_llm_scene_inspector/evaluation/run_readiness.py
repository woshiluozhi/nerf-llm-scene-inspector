"""Run-level readiness gates for real-run and external-review decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack
from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


GateStatus = Literal["pass", "warn", "fail"]
ReadinessLevel = Literal[
    "blocked",
    "dry_run_needs_real_run",
    "ready_for_gpu_run",
    "shareable_smoke_demo",
    "real_run_review_ready",
    "portfolio_ready",
    "needs_review",
]


@dataclass
class ReadinessGate:
    """One pass/warn/fail readiness gate for a run."""

    name: str
    status: GateStatus
    message: str
    action: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RunReadinessReport:
    """Machine-readable and reviewer-facing readiness summary for one run."""

    run_dir: str
    scene_name: str
    generated_at: str
    dry_run: bool
    backend: str
    readiness_level: ReadinessLevel
    ready_to_start_real_run: bool
    ready_for_external_review: bool
    query_evidence_status: str = ""
    query_counter_evidence_count: int = 0
    query_risk_flag_count: int = 0
    gates: list[ReadinessGate] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    evidence_notes: list[str] = field(default_factory=list)

    @property
    def fail_count(self) -> int:
        return sum(1 for gate in self.gates if gate.status == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for gate in self.gates if gate.status == "warn")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "generated_at": self.generated_at,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "readiness_level": self.readiness_level,
            "ready_to_start_real_run": self.ready_to_start_real_run,
            "ready_for_external_review": self.ready_for_external_review,
            "query_evidence_status": self.query_evidence_status,
            "query_counter_evidence_count": self.query_counter_evidence_count,
            "query_risk_flag_count": self.query_risk_flag_count,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "gates": [gate.to_dict() for gate in self.gates],
            "next_actions": list(self.next_actions),
            "evidence_notes": list(self.evidence_notes),
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
            "# Run Readiness Gate",
            "",
            "This report summarizes whether a run is ready for a real GPU-backed Nerfstudio/LERF run and whether its artifacts are ready for external review.",
            "",
            "## Decision",
            "",
            f"- Scene: `{self.scene_name}`",
            f"- Run directory: `{self.run_dir}`",
            f"- Backend: `{self.backend}`",
            f"- Dry run: {self.dry_run}",
            f"- Readiness level: `{self.readiness_level}`",
            f"- Ready to start real run: {self.ready_to_start_real_run}",
            f"- Ready for external review: {self.ready_for_external_review}",
            f"- Query evidence: {self.query_evidence_status or 'unknown'}",
            f"- Query counter-evidence items: {self.query_counter_evidence_count}",
            f"- Query risk flags: {self.query_risk_flag_count}",
            f"- Failed gates: {self.fail_count}",
            f"- Warning gates: {self.warn_count}",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Gates",
            "",
            *_gate_lines(self.gates),
            "",
            "## Next Actions",
            "",
            *_list_lines(self.next_actions),
            "",
            "## Evidence Notes",
            "",
            *_list_lines(self.evidence_notes),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_run_readiness(
    run_dir: str | Path,
    *,
    pack_dir: str | Path | None = None,
    pack_validation_path: str | Path | None = None,
) -> RunReadinessReport:
    """Build a run-level readiness summary from existing artifacts."""

    root = Path(run_dir)
    pipeline = _read_json(root / "pipeline_summary.json")
    preflight = _read_json(root / "preflight_report.json")
    environment = _read_json(root / "environment_report.json")
    capture = _read_json(root / "capture_manifest_validation.json")
    scene = _read_json(root / "scene_data_inspection.json")
    language = _read_json(root / "training" / "language_train_summary.json")
    query_evidence = _read_json(root / "query_evidence_audit.json")
    failure_diagnostics = _read_json(root / "failure_diagnostics.json")
    quality = _read_json(root / "quality_gate.json")
    claim_audit = _read_json(root / "claim_audit.json")
    submission = _read_json(root / "submission_packet" / "submission_packet.json")
    result_card = _read_json(root / "run_result_card.json")
    pack_validation = _pack_validation(pack_dir, pack_validation_path)

    dry_run = bool(pipeline.get("dry_run", result_card.get("dry_run", False)))
    scene_name = str(pipeline.get("scene_name") or root.name)
    backend = str(pipeline.get("backend") or language.get("backend") or "unknown")
    counter_evidence_count, risk_flag_count = _query_evidence_counts(query_evidence)
    gates = [
        _pipeline_gate(pipeline),
        _evidence_mode_gate(dry_run),
        _capture_gate(capture),
        _preflight_gate(preflight),
        _environment_gate(environment, dry_run=dry_run),
        _scene_gate(scene),
        _language_training_gate(language, dry_run=dry_run),
        _query_evidence_gate(query_evidence),
        _failure_diagnostics_gate(failure_diagnostics),
        _quality_gate(quality),
        _claim_audit_gate(claim_audit),
        _submission_gate(submission),
        _portfolio_pack_gate(submission, pack_validation),
    ]
    ready_to_start_real_run = _ready_to_start_real_run(gates, dry_run=dry_run)
    ready_for_external_review = _ready_for_external_review(gates, submission)
    readiness_level = _readiness_level(
        gates,
        dry_run=dry_run,
        ready_to_start_real_run=ready_to_start_real_run,
        ready_for_external_review=ready_for_external_review,
        submission=submission,
    )
    return RunReadinessReport(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        generated_at=utc_timestamp(),
        dry_run=dry_run,
        backend=backend,
        readiness_level=readiness_level,
        ready_to_start_real_run=ready_to_start_real_run,
        ready_for_external_review=ready_for_external_review,
        query_evidence_status=str(query_evidence.get("status") or ""),
        query_counter_evidence_count=counter_evidence_count,
        query_risk_flag_count=risk_flag_count,
        gates=gates,
        next_actions=_next_actions(gates, readiness_level),
        evidence_notes=_evidence_notes(dry_run=dry_run, readiness_level=readiness_level),
    )


def write_run_readiness(
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    pack_dir: str | Path | None = None,
    pack_validation_path: str | Path | None = None,
) -> RunReadinessReport:
    """Build and write JSON plus Markdown readiness reports."""

    root = Path(run_dir)
    report = build_run_readiness(root, pack_dir=pack_dir, pack_validation_path=pack_validation_path)
    report.to_json(output or root / "run_readiness.json")
    report.to_markdown(markdown_output or root / "run_readiness.md")
    return report


def _pipeline_gate(pipeline: dict[str, Any]) -> ReadinessGate:
    if not pipeline:
        return ReadinessGate(
            "pipeline_summary",
            "fail",
            "pipeline_summary.json is missing or unreadable.",
            "Run scripts/run_scene_pipeline.py before checking readiness.",
            "pipeline_summary.json",
        )
    if pipeline.get("success") is True:
        return ReadinessGate("pipeline_summary", "pass", "Pipeline summary reports success.", artifact="pipeline_summary.json")
    return ReadinessGate(
        "pipeline_summary",
        "fail",
        f"Pipeline summary success={pipeline.get('success')}.",
        "Fix failed pipeline steps before spending GPU time or sharing externally.",
        "pipeline_summary.json",
    )


def _evidence_mode_gate(dry_run: bool) -> ReadinessGate:
    if dry_run:
        return ReadinessGate(
            "evidence_mode",
            "warn",
            "Current artifacts are CPU dry-run smoke evidence.",
            "Run without --dry-run on a CUDA/Nerfstudio/LERF machine before claiming trained-scene results.",
            "pipeline_summary.json",
        )
    return ReadinessGate("evidence_mode", "pass", "Run is recorded as a real-scene mode.", artifact="pipeline_summary.json")


def _capture_gate(capture: dict[str, Any]) -> ReadinessGate:
    status = str(capture.get("status") or "")
    if not capture:
        return ReadinessGate(
            "capture_manifest",
            "warn",
            "Capture manifest validation is missing.",
            "Create or copy capture metadata before a real run.",
            "capture_manifest_validation.md",
        )
    if status == "ready":
        return ReadinessGate("capture_manifest", "pass", "Capture manifest validation is ready.", artifact="capture_manifest_validation.md")
    if status == "blocked":
        return ReadinessGate(
            "capture_manifest",
            "fail",
            "Capture manifest validation is blocked.",
            "Fix capture metadata, privacy review, or static-scene fields.",
            "capture_manifest_validation.md",
        )
    return ReadinessGate(
        "capture_manifest",
        "warn",
        f"Capture manifest validation status is {status or 'unknown'}.",
        "Review capture_manifest_validation.md before real training.",
        "capture_manifest_validation.md",
    )


def _preflight_gate(preflight: dict[str, Any]) -> ReadinessGate:
    status = str(preflight.get("status") or "")
    if not preflight:
        return ReadinessGate(
            "preflight",
            "warn",
            "Real-run preflight report is missing.",
            "Run scripts/preflight_real_run.py with the intended input and backend.",
            "preflight_report.md",
        )
    if status == "ready" and preflight.get("ready_for_real_run") is not False:
        return ReadinessGate("preflight", "pass", "Preflight checks are ready.", artifact="preflight_report.md")
    if status == "blocked":
        return ReadinessGate(
            "preflight",
            "fail",
            "Preflight reported blocker-level checks.",
            "Resolve failed input, data, dependency, or backend checks before training.",
            "preflight_report.md",
        )
    return ReadinessGate(
        "preflight",
        "warn",
        f"Preflight status is {status or 'unknown'}.",
        "Review preflight_report.md before spending GPU time.",
        "preflight_report.md",
    )


def _environment_gate(environment: dict[str, Any], *, dry_run: bool) -> ReadinessGate:
    if not environment:
        return ReadinessGate(
            "environment_gpu_upstream",
            "warn",
            "Environment report is missing.",
            "Run python scripts/check_env.py --check-upstream --require-gpu --verbose.",
            "environment_report.json",
        )
    strict_failures = [str(item) for item in environment.get("strict_failures") or []]
    if strict_failures:
        return ReadinessGate(
            "environment_gpu_upstream",
            "fail",
            "Required environment checks failed: " + ", ".join(strict_failures),
            "Install CUDA-compatible PyTorch, Nerfstudio, LERF, COLMAP, FFmpeg, and registered ns-train methods.",
            "environment_report.json",
        )
    checks = environment.get("checks") if isinstance(environment.get("checks"), list) else []
    if _has_required_gpu_and_upstream(checks):
        return ReadinessGate(
            "environment_gpu_upstream",
            "pass",
            "Environment report includes required GPU and upstream tool checks.",
            artifact="environment_report.json",
        )
    status = "warn" if dry_run or environment.get("ok") is True else "fail"
    return ReadinessGate(
        "environment_gpu_upstream",
        status,
        "Environment report was not generated with required GPU/upstream checks.",
        "Run python scripts/check_env.py --check-upstream --require-gpu --verbose before a real run.",
        "environment_report.json",
    )


def _scene_gate(scene: dict[str, Any]) -> ReadinessGate:
    if not scene:
        return ReadinessGate(
            "scene_data",
            "warn",
            "Processed scene inspection is missing.",
            "Run scripts/inspect_scene_data.py after ns-process-data.",
            "scene_data_inspection.md",
        )
    if scene.get("ready_for_training") is True:
        return ReadinessGate("scene_data", "pass", "Scene inspection is ready for training.", artifact="scene_data_inspection.md")
    return ReadinessGate(
        "scene_data",
        "warn",
        "Scene inspection is not ready for training.",
        "Recapture or rerun ns-process-data until frame count, image files, and pose coverage are acceptable.",
        "scene_data_inspection.md",
    )


def _language_training_gate(language: dict[str, Any], *, dry_run: bool) -> ReadinessGate:
    if not language:
        return ReadinessGate(
            "language_training",
            "warn",
            "Language-field training summary is missing.",
            "Run train_language_field.py or run_scene_pipeline.py with language training enabled.",
            "training/language_train_summary.json",
        )
    if language.get("success") is False:
        return ReadinessGate(
            "language_training",
            "fail",
            "Language-field training summary reports failure.",
            "Inspect training/language_train_summary.json and rerun training.",
            "training/language_train_summary.json",
        )
    if dry_run or language.get("dry_run") is True:
        return ReadinessGate(
            "language_training",
            "warn",
            "Language-field summary is dry-run evidence.",
            "Run real LERF/OpenNeRF training and record a real config.yml path.",
            "training/language_train_summary.json",
        )
    if language.get("config_path"):
        return ReadinessGate(
            "language_training",
            "pass",
            "Language-field training recorded a config path.",
            artifact="training/language_train_summary.json",
        )
    return ReadinessGate(
        "language_training",
        "warn",
        "Language-field training summary has no config_path.",
        "Confirm ns-train completed and write the generated config.yml path.",
        "training/language_train_summary.json",
    )


def _query_evidence_gate(audit: dict[str, Any]) -> ReadinessGate:
    if not audit:
        return ReadinessGate(
            "query_evidence",
            "warn",
            "Query evidence audit is missing.",
            "Run scripts/audit_query_evidence.py before external review.",
            "query_evidence_audit.md",
        )
    status = str(audit.get("status") or "")
    counter_evidence_count, risk_flag_count = _query_evidence_counts(audit)
    if audit.get("ok") is False or status == "fail":
        return ReadinessGate(
            "query_evidence",
            "fail",
            "Query evidence audit reports failed query artifacts.",
            "Regenerate missing query reports, overlays, or visual summaries.",
            "query_evidence_audit.md",
        )
    if risk_flag_count:
        return ReadinessGate(
            "query_evidence",
            "fail",
            f"Query evidence audit reports {risk_flag_count} risk flag(s).",
            "Resolve or document counter-evidence conflicts before external review.",
            "query_evidence_audit.md",
        )
    if counter_evidence_count:
        return ReadinessGate(
            "query_evidence",
            "warn",
            f"Query evidence audit reports {counter_evidence_count} counter-evidence item(s).",
            "Review disambiguation prompts before sharing scene-answer claims.",
            "query_evidence_audit.md",
        )
    if status == "warn":
        return ReadinessGate(
            "query_evidence",
            "warn",
            "Query evidence audit reports warning-level evidence.",
            "Inspect fallback modes, missing artifacts, and query warnings.",
            "query_evidence_audit.md",
        )
    if status == "pass":
        return ReadinessGate("query_evidence", "pass", "Query evidence audit passed.", artifact="query_evidence_audit.md")
    return ReadinessGate(
        "query_evidence",
        "warn",
        f"Query evidence audit status is {status or 'unknown'}.",
        "Regenerate query_evidence_audit.json.",
        "query_evidence_audit.json",
    )


def _quality_gate(quality: dict[str, Any]) -> ReadinessGate:
    status = str(quality.get("status") or "")
    if not quality:
        return ReadinessGate(
            "quality_gate",
            "warn",
            "Run quality gate is missing.",
            "Run scripts/check_run_quality.py before sharing.",
            "quality_gate.md",
        )
    if status == "pass":
        return ReadinessGate("quality_gate", "pass", "Quality gate passed.", artifact="quality_gate.md")
    if status == "fail" or quality.get("passed") is False:
        return ReadinessGate(
            "quality_gate",
            "fail",
            "Quality gate failed.",
            "Open quality_gate.md and fix failed criteria.",
            "quality_gate.md",
        )
    return ReadinessGate(
        "quality_gate",
        "warn",
        f"Quality gate status is {status or 'unknown'}.",
        "Review warning criteria before sharing externally.",
        "quality_gate.md",
    )


def _failure_diagnostics_gate(diagnostics: dict[str, Any]) -> ReadinessGate:
    status = str(diagnostics.get("status") or "")
    if not diagnostics:
        return ReadinessGate(
            "failure_diagnostics",
            "warn",
            "Failure diagnostics report is missing.",
            "Run python scripts/diagnose_run_failures.py --run-dir <run> before launch or sharing.",
            "failure_diagnostics.md",
        )
    if status == "clear":
        return ReadinessGate(
            "failure_diagnostics",
            "pass",
            "No known failure signatures were detected.",
            artifact="failure_diagnostics.md",
        )
    if status == "blocked" or diagnostics.get("blocker_count"):
        return ReadinessGate(
            "failure_diagnostics",
            "fail",
            "Failure diagnostics found blocker-level issues.",
            "Open failure_diagnostics.md and resolve the listed root-cause fixes.",
            "failure_diagnostics.md",
        )
    return ReadinessGate(
        "failure_diagnostics",
        "warn",
        f"Failure diagnostics status is {status or 'unknown'}.",
        "Review failure_diagnostics.md before spending GPU time or sharing externally.",
        "failure_diagnostics.md",
    )


def _claim_audit_gate(claim_audit: dict[str, Any]) -> ReadinessGate:
    status = str(claim_audit.get("status") or "")
    if not claim_audit:
        return ReadinessGate(
            "claim_audit",
            "warn",
            "Claim audit is missing.",
            "Run scripts/audit_claims.py before external sharing.",
            "claim_audit.md",
        )
    if status == "pass" and claim_audit.get("ok") is not False:
        return ReadinessGate("claim_audit", "pass", "Claim audit passed.", artifact="claim_audit.md")
    if status == "fail" or (claim_audit.get("ok") is False and status not in {"warn", "pass"}):
        return ReadinessGate(
            "claim_audit",
            "fail",
            "Claim audit reports unsupported external-facing claims.",
            "Fix claim_audit.md findings before sharing.",
            "claim_audit.md",
        )
    return ReadinessGate(
        "claim_audit",
        "warn",
        f"Claim audit status is {status or 'unknown'}.",
        "Review claim_audit.md warnings before outreach.",
        "claim_audit.md",
    )


def _submission_gate(submission: dict[str, Any]) -> ReadinessGate:
    readiness = str(submission.get("readiness_level") or "")
    if not submission:
        return ReadinessGate(
            "submission_packet",
            "warn",
            "Submission packet is missing.",
            "Run scripts/create_submission_packet.py.",
            "submission_packet/submission_checklist.md",
        )
    if readiness in {"shareable_smoke_demo", "real_run_review_ready", "portfolio_ready"}:
        return ReadinessGate(
            "submission_packet",
            "pass",
            f"Submission packet readiness is {readiness}.",
            artifact="submission_packet/submission_checklist.md",
        )
    if readiness == "blocked":
        return ReadinessGate(
            "submission_packet",
            "fail",
            "Submission packet is blocked.",
            "Resolve failed submission checklist items before sharing.",
            "submission_packet/submission_checklist.md",
        )
    return ReadinessGate(
        "submission_packet",
        "warn",
        f"Submission packet readiness is {readiness or 'unknown'}.",
        "Refresh submission packet after pack validation.",
        "submission_packet/submission_checklist.md",
    )


def _portfolio_pack_gate(submission: dict[str, Any], pack_validation: dict[str, Any]) -> ReadinessGate:
    if pack_validation:
        if pack_validation.get("ok") is True:
            return ReadinessGate(
                "portfolio_pack",
                "pass",
                "Portfolio pack validation passed.",
                artifact="portfolio_pack_validation.json",
            )
        return ReadinessGate(
            "portfolio_pack",
            "fail",
            "Portfolio pack validation did not pass.",
            "Open portfolio_pack_validation.json and fix missing artifacts, path leaks, or quality issues.",
            "portfolio_pack_validation.json",
        )
    if submission.get("pack_ok") is True:
        return ReadinessGate(
            "portfolio_pack",
            "pass",
            "Submission packet records a validated portfolio pack.",
            artifact="portfolio_pack_validation.json",
        )
    return ReadinessGate(
        "portfolio_pack",
        "warn",
        "No validated portfolio pack was recorded for this readiness report.",
        "Run finalize_annotations.py with --export-pack --zip-pack, then regenerate readiness with --pack.",
        "portfolio_pack_validation.json",
    )


def _ready_to_start_real_run(gates: list[ReadinessGate], *, dry_run: bool) -> bool:
    if dry_run:
        return False
    needed = {"pipeline_summary", "capture_manifest", "preflight", "environment_gpu_upstream", "scene_data"}
    gate_map = {gate.name: gate.status for gate in gates}
    return all(gate_map.get(name) == "pass" for name in needed)


def _ready_for_external_review(gates: list[ReadinessGate], submission: dict[str, Any]) -> bool:
    if any(gate.status == "fail" for gate in gates):
        return False
    readiness = str(submission.get("readiness_level") or "")
    return readiness in {"shareable_smoke_demo", "real_run_review_ready", "portfolio_ready"}


def _readiness_level(
    gates: list[ReadinessGate],
    *,
    dry_run: bool,
    ready_to_start_real_run: bool,
    ready_for_external_review: bool,
    submission: dict[str, Any],
) -> ReadinessLevel:
    if any(gate.status == "fail" for gate in gates):
        return "blocked"
    submission_level = str(submission.get("readiness_level") or "")
    if submission_level == "portfolio_ready":
        return "portfolio_ready"
    if ready_for_external_review and dry_run:
        return "shareable_smoke_demo"
    if ready_for_external_review:
        return "real_run_review_ready"
    if ready_to_start_real_run:
        return "ready_for_gpu_run"
    if dry_run:
        return "dry_run_needs_real_run"
    return "needs_review"


def _next_actions(gates: list[ReadinessGate], readiness_level: ReadinessLevel) -> list[str]:
    actions = [gate.action for gate in gates if gate.status in {"fail", "warn"} and gate.action]
    if readiness_level == "dry_run_needs_real_run":
        actions.insert(0, "Run a CUDA-backed Nerfstudio/LERF pipeline without --dry-run.")
    if readiness_level == "ready_for_gpu_run":
        actions.insert(0, "Start the real training/query pipeline and monitor generated configs and overlays.")
    if readiness_level in {"shareable_smoke_demo", "real_run_review_ready", "portfolio_ready"}:
        actions.insert(0, "Attach the repository, portfolio pack, and latest passing CI run when sharing.")
    return _dedupe(actions)[:8]


def _evidence_notes(*, dry_run: bool, readiness_level: ReadinessLevel) -> list[str]:
    notes = [
        "This readiness gate is a project artifact, not an independent benchmark result.",
        "Single-scene evidence should be described as portfolio evidence unless broader evaluation is added.",
    ]
    if dry_run:
        notes.append("Dry-run artifacts validate orchestration and artifact shape, not trained LERF scene quality.")
    if readiness_level == "blocked":
        notes.append("Blocked readiness means at least one required evidence or safety gate failed.")
    return notes


def _query_evidence_counts(audit: dict[str, Any]) -> tuple[int, int]:
    totals = audit.get("totals") if isinstance(audit.get("totals"), dict) else {}
    counter = _safe_int(totals.get("counter_evidence_count"))
    risk = _safe_int(totals.get("risk_flag_count"))
    tasks = audit.get("tasks") if isinstance(audit.get("tasks"), list) else []
    if not counter:
        counter = sum(
            _safe_int(task.get("counter_evidence_count"))
            for task in tasks
            if isinstance(task, dict)
        )
    if not risk:
        risk = sum(
            _safe_int(task.get("risk_flag_count"))
            for task in tasks
            if isinstance(task, dict)
        )
    return counter, risk


def _pack_validation(
    pack_dir: str | Path | None,
    pack_validation_path: str | Path | None,
) -> dict[str, Any]:
    if pack_validation_path:
        return _read_json(pack_validation_path)
    if not pack_dir:
        return {}
    pack = Path(pack_dir)
    validation_path = pack / "portfolio_pack_validation.json" if pack.is_dir() else pack.with_name(f"{pack.stem}_validation.json")
    if validation_path.exists():
        return _read_json(validation_path)
    if pack.exists():
        return validate_portfolio_pack(pack).to_dict()
    return {"ok": False, "errors": [f"Portfolio pack path does not exist: {pack.name or 'portfolio_pack'}"]}


def _has_required_gpu_and_upstream(checks: list[Any]) -> bool:
    required_ok = {
        str(check.get("name")): bool(check.get("ok"))
        for check in checks
        if isinstance(check, dict) and check.get("required") is True
    }
    has_cuda = required_ok.get("cuda") is True
    has_ns_train = required_ok.get("ns-train") is True
    has_process = required_ok.get("ns-process-data") is True
    has_lerf_method = any(name.startswith("ns-train method:lerf") and ok for name, ok in required_ok.items())
    return has_cuda and has_ns_train and has_process and has_lerf_method


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _gate_lines(gates: list[ReadinessGate]) -> list[str]:
    if not gates:
        return ["- No gates were evaluated."]
    return [
        f"- `{gate.status}` {gate.name}: {gate.message}"
        + (f" Action: {gate.action}" if gate.action else "")
        + (f" Artifact: `{gate.artifact}`" if gate.artifact else "")
        for gate in gates
    ]


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
