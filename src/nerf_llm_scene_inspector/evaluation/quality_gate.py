"""Run-level quality gates for deciding whether evidence is share-ready."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack
from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp

GateProfile = Literal["smoke", "real-run", "portfolio"]
GateStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class GatePolicy:
    """Profile-specific threshold policy."""

    min_query_reports: int
    min_evaluated_queries: int
    min_evidence_ratio: float
    allow_dry_run: bool
    require_audit_ready: bool
    require_capture_ready: bool
    require_portfolio_level: bool
    require_pack: bool
    fail_on_annotation_warnings: bool


PROFILE_POLICIES: dict[GateProfile, GatePolicy] = {
    "smoke": GatePolicy(
        min_query_reports=1,
        min_evaluated_queries=0,
        min_evidence_ratio=0.60,
        allow_dry_run=True,
        require_audit_ready=False,
        require_capture_ready=False,
        require_portfolio_level=False,
        require_pack=False,
        fail_on_annotation_warnings=False,
    ),
    "real-run": GatePolicy(
        min_query_reports=3,
        min_evaluated_queries=1,
        min_evidence_ratio=0.65,
        allow_dry_run=False,
        require_audit_ready=False,
        require_capture_ready=True,
        require_portfolio_level=False,
        require_pack=False,
        fail_on_annotation_warnings=False,
    ),
    "portfolio": GatePolicy(
        min_query_reports=3,
        min_evaluated_queries=3,
        min_evidence_ratio=0.85,
        allow_dry_run=False,
        require_audit_ready=True,
        require_capture_ready=True,
        require_portfolio_level=True,
        require_pack=True,
        fail_on_annotation_warnings=True,
    ),
}


@dataclass
class QualityGateCriterion:
    """One pass/warn/fail criterion in a run quality gate."""

    name: str
    status: GateStatus
    message: str
    recommendation: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class QualityGateReport:
    """Portable quality-gate decision for one pipeline run."""

    profile: GateProfile
    status: GateStatus
    passed: bool
    run_dir: str
    scene_name: str = ""
    dry_run: bool = False
    evidence_level: str = ""
    evidence_score: int = 0
    evidence_max_score: int = 0
    audit_status: str = ""
    capture_manifest_status: str = ""
    query_report_count: int = 0
    evaluated_query_count: int = 0
    criteria: list[QualityGateCriterion] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    @property
    def fail_count(self) -> int:
        return sum(1 for criterion in self.criteria if criterion.status == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for criterion in self.criteria if criterion.status == "warn")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "status": self.status,
            "passed": self.passed,
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "dry_run": self.dry_run,
            "evidence_level": self.evidence_level,
            "evidence_score": self.evidence_score,
            "evidence_max_score": self.evidence_max_score,
            "audit_status": self.audit_status,
            "capture_manifest_status": self.capture_manifest_status,
            "query_report_count": self.query_report_count,
            "evaluated_query_count": self.evaluated_query_count,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
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
            "# Run Quality Gate",
            "",
            f"- Profile: {self.profile}",
            f"- Status: {self.status}",
            f"- Passed: {self.passed}",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Dry run: {self.dry_run}",
            f"- Evidence level: {self.evidence_level or 'unknown'}",
            f"- Evidence score: {self.evidence_score}/{self.evidence_max_score}",
            f"- Audit status: {self.audit_status or 'unknown'}",
            f"- Capture manifest: {self.capture_manifest_status or 'unknown'}",
            f"- Query reports: {self.query_report_count}",
            f"- Evaluated queries: {self.evaluated_query_count}",
            "",
            "## Criteria",
            "",
            *_criterion_lines(self.criteria),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def check_run_quality(
    run_dir: str | Path,
    *,
    profile: GateProfile = "smoke",
    pack_dir: str | Path | None = None,
    min_query_reports: int | None = None,
    min_evaluated_queries: int | None = None,
    min_evidence_ratio: float | None = None,
    allow_dry_run: bool | None = None,
    require_pack: bool | None = None,
) -> QualityGateReport:
    """Evaluate a pipeline run against a share-readiness profile."""

    root = Path(run_dir)
    policy = _policy_with_overrides(
        profile,
        min_query_reports=min_query_reports,
        min_evaluated_queries=min_evaluated_queries,
        min_evidence_ratio=min_evidence_ratio,
        allow_dry_run=allow_dry_run,
        require_pack=require_pack,
    )
    pipeline = _read_json(root / "pipeline_summary.json")
    audit = _read_json(root / "run_audit.json")
    diagnostics = _read_json(root / "failure_diagnostics.json")
    scorecard = _read_json(root / "evidence_scorecard.json")
    capture = _read_json(root / "capture_manifest_validation.json")
    annotations = _read_json(root / "evaluation" / "annotation_validation.json")
    evaluation = _read_json(root / "evaluation" / "eval_summary.json")

    criteria = [
        _pipeline_criterion(pipeline),
        _dry_run_criterion(bool(pipeline.get("dry_run")), policy),
        _audit_criterion(audit, policy),
        _failure_diagnostics_criterion(diagnostics),
        _evidence_criterion(scorecard, policy),
        _capture_criterion(capture, policy),
        _query_criterion(root, pipeline, scorecard, policy),
        _evaluation_criterion(scorecard, evaluation, policy),
        _annotation_criterion(annotations, policy),
        _pack_criterion(pack_dir, policy),
    ]
    status = _overall_status(criteria)
    return QualityGateReport(
        profile=profile,
        status=status,
        passed=status != "fail",
        run_dir=_display_run_dir(root),
        scene_name=str(pipeline.get("scene_name") or scorecard.get("scene_name") or root.name),
        dry_run=bool(pipeline.get("dry_run")),
        evidence_level=str(scorecard.get("evidence_level") or ""),
        evidence_score=_safe_int(scorecard.get("score")),
        evidence_max_score=_safe_int(scorecard.get("max_score")),
        audit_status=str(audit.get("status") or ""),
        capture_manifest_status=str(capture.get("status") or ""),
        query_report_count=_query_report_count(root, scorecard),
        evaluated_query_count=_evaluated_query_count(scorecard, evaluation),
        criteria=criteria,
    )


def _policy_with_overrides(
    profile: GateProfile,
    *,
    min_query_reports: int | None,
    min_evaluated_queries: int | None,
    min_evidence_ratio: float | None,
    allow_dry_run: bool | None,
    require_pack: bool | None,
) -> GatePolicy:
    policy = PROFILE_POLICIES[profile]
    return replace(
        policy,
        min_query_reports=policy.min_query_reports
        if min_query_reports is None
        else min_query_reports,
        min_evaluated_queries=policy.min_evaluated_queries
        if min_evaluated_queries is None
        else min_evaluated_queries,
        min_evidence_ratio=policy.min_evidence_ratio
        if min_evidence_ratio is None
        else min_evidence_ratio,
        allow_dry_run=policy.allow_dry_run if allow_dry_run is None else allow_dry_run,
        require_pack=policy.require_pack if require_pack is None else require_pack,
    )


def _pipeline_criterion(summary: dict[str, Any]) -> QualityGateCriterion:
    if not summary:
        return QualityGateCriterion(
            "pipeline_summary",
            "fail",
            "pipeline_summary.json is missing or unreadable.",
            "Run scripts/run_scene_pipeline.py before applying the quality gate.",
            "pipeline_summary.json",
        )
    if summary.get("success") is True:
        return QualityGateCriterion(
            "pipeline_summary",
            "pass",
            "Pipeline summary reports success=true.",
            artifact="pipeline_summary.json",
        )
    return QualityGateCriterion(
        "pipeline_summary",
        "fail",
        "Pipeline summary does not report success=true.",
        "Inspect failed steps in pipeline_summary.json and rerun the pipeline.",
        "pipeline_summary.json",
    )


def _dry_run_criterion(dry_run: bool, policy: GatePolicy) -> QualityGateCriterion:
    if dry_run and not policy.allow_dry_run:
        return QualityGateCriterion(
            "run_mode",
            "fail",
            "This is a dry-run artifact, but the selected profile requires a real run.",
            "Run on a CUDA machine with Nerfstudio/LERF installed before using this gate.",
            "pipeline_summary.json",
        )
    if dry_run:
        return QualityGateCriterion(
            "run_mode",
            "warn",
            "Dry-run artifacts validate wiring only, not trained NeRF/LERF quality.",
            "Use the smoke profile for dry-run demos and real-run/portfolio profiles for evidence.",
            "pipeline_summary.json",
        )
    return QualityGateCriterion("run_mode", "pass", "Run is marked as real-mode.", artifact="pipeline_summary.json")


def _audit_criterion(audit: dict[str, Any], policy: GatePolicy) -> QualityGateCriterion:
    if not audit:
        return QualityGateCriterion(
            "run_audit",
            "fail",
            "run_audit.json is missing or unreadable.",
            "Run python scripts/audit_run.py --run-dir <run-dir>.",
            "run_audit.json",
        )
    status = str(audit.get("status") or "")
    if status == "blocked":
        return QualityGateCriterion(
            "run_audit",
            "fail",
            "Run audit reports blocker-level issues.",
            "Open run_audit.md and fix blocker findings.",
            "run_audit.md",
        )
    if status == "needs_review":
        gate_status: GateStatus = "fail" if policy.require_audit_ready else "warn"
        return QualityGateCriterion(
            "run_audit",
            gate_status,
            "Run audit still has warning-level findings.",
            "Review run_audit.md before sharing or switch to a stricter profile after cleanup.",
            "run_audit.md",
        )
    if status == "ready":
        return QualityGateCriterion("run_audit", "pass", "Run audit is ready.", artifact="run_audit.md")
    return QualityGateCriterion(
        "run_audit",
        "warn",
        f"Run audit has unrecognized status: {status or 'missing'}.",
        "Regenerate run_audit.json.",
        "run_audit.json",
    )


def _failure_diagnostics_criterion(diagnostics: dict[str, Any]) -> QualityGateCriterion:
    if not diagnostics:
        return QualityGateCriterion(
            "failure_diagnostics",
            "warn",
            "failure_diagnostics.json is missing or unreadable.",
            "Run python scripts/diagnose_run_failures.py --run-dir <run-dir>.",
            "failure_diagnostics.json",
        )
    status = str(diagnostics.get("status") or "")
    if status == "blocked" or diagnostics.get("blocker_count"):
        return QualityGateCriterion(
            "failure_diagnostics",
            "fail",
            "Failure diagnostics report blocker-level issues.",
            "Open failure_diagnostics.md and fix the listed root causes.",
            "failure_diagnostics.md",
        )
    if status == "needs_attention" or diagnostics.get("warning_count"):
        return QualityGateCriterion(
            "failure_diagnostics",
            "warn",
            "Failure diagnostics report warning-level issues.",
            "Review failure_diagnostics.md before sharing or spending more GPU time.",
            "failure_diagnostics.md",
        )
    if status == "clear":
        return QualityGateCriterion(
            "failure_diagnostics",
            "pass",
            "No known failure signatures were detected.",
            artifact="failure_diagnostics.md",
        )
    return QualityGateCriterion(
        "failure_diagnostics",
        "warn",
        f"Failure diagnostics has unrecognized status: {status or 'missing'}.",
        "Regenerate failure_diagnostics.json.",
        "failure_diagnostics.json",
    )


def _evidence_criterion(scorecard: dict[str, Any], policy: GatePolicy) -> QualityGateCriterion:
    if not scorecard:
        return QualityGateCriterion(
            "evidence_scorecard",
            "fail",
            "evidence_scorecard.json is missing or unreadable.",
            "Run python scripts/create_evidence_scorecard.py --run-dir <run-dir>.",
            "evidence_scorecard.json",
        )
    score = _safe_int(scorecard.get("score"))
    max_score = max(_safe_int(scorecard.get("max_score")), 1)
    ratio = score / max_score
    level = str(scorecard.get("evidence_level") or "")
    if policy.require_portfolio_level and level != "portfolio_ready_real_run":
        return QualityGateCriterion(
            "evidence_scorecard",
            "fail",
            f"Portfolio profile requires portfolio_ready_real_run, got {level or 'missing'}.",
            "Improve real-run evidence, annotations, and packaging until the scorecard is portfolio-ready.",
            "evidence_scorecard.md",
        )
    if level in {"blocked", "needs_evidence", ""}:
        return QualityGateCriterion(
            "evidence_scorecard",
            "fail",
            f"Evidence level is {level or 'missing'}.",
            "Open evidence_scorecard.md and address the weakest criteria.",
            "evidence_scorecard.md",
        )
    if ratio < policy.min_evidence_ratio:
        return QualityGateCriterion(
            "evidence_scorecard",
            "fail",
            f"Evidence score ratio {ratio:.2f} is below required {policy.min_evidence_ratio:.2f}.",
            "Add missing query, annotation, scene-quality, or portfolio artifacts.",
            "evidence_scorecard.md",
        )
    if level == "needs_review":
        return QualityGateCriterion(
            "evidence_scorecard",
            "warn",
            "Evidence scorecard is useful but still marked needs_review.",
            "Resolve scorecard recommendations before treating this as final evidence.",
            "evidence_scorecard.md",
        )
    return QualityGateCriterion(
        "evidence_scorecard",
        "pass",
        f"Evidence scorecard passes profile threshold ({score}/{max_score}).",
        artifact="evidence_scorecard.md",
    )


def _capture_criterion(capture: dict[str, Any], policy: GatePolicy) -> QualityGateCriterion:
    if not capture:
        status: GateStatus = "fail" if policy.require_capture_ready else "warn"
        return QualityGateCriterion(
            "capture_manifest",
            status,
            "capture_manifest_validation.json is missing or unreadable.",
            "Create or regenerate capture metadata before sharing real-scene results.",
            "capture_manifest_validation.json",
        )
    status_raw = str(capture.get("status") or "")
    if status_raw == "blocked":
        return QualityGateCriterion(
            "capture_manifest",
            "fail",
            "Capture manifest validation is blocked.",
            "Fix capture-manifest failures and rerun validation.",
            "capture_manifest_validation.md",
        )
    if status_raw == "needs_review":
        gate_status: GateStatus = "fail" if policy.require_capture_ready else "warn"
        return QualityGateCriterion(
            "capture_manifest",
            gate_status,
            "Capture manifest still needs review.",
            "Fill device, overlap, lighting, static-scene, and privacy-review fields.",
            "capture_manifest_validation.md",
        )
    if status_raw == "ready":
        return QualityGateCriterion(
            "capture_manifest",
            "pass",
            "Capture manifest is ready.",
            artifact="capture_manifest_validation.md",
        )
    return QualityGateCriterion(
        "capture_manifest",
        "warn",
        f"Capture manifest has unrecognized status: {status_raw or 'missing'}.",
        "Regenerate capture_manifest_validation.json.",
        "capture_manifest_validation.json",
    )


def _query_criterion(
    root: Path,
    pipeline: dict[str, Any],
    scorecard: dict[str, Any],
    policy: GatePolicy,
) -> QualityGateCriterion:
    planned = len([query for query in pipeline.get("queries") or [] if str(query).strip()])
    query_reports = _query_report_count(root, scorecard)
    overlays = _safe_int(scorecard.get("overlay_count"))
    if query_reports < policy.min_query_reports:
        return QualityGateCriterion(
            "query_coverage",
            "fail",
            f"Only {query_reports} query reports found; profile requires {policy.min_query_reports}.",
            "Run representative semantic queries and keep scene_query_report.json outputs.",
            "queries/",
        )
    if overlays < min(policy.min_query_reports, max(planned, 1)):
        return QualityGateCriterion(
            "query_coverage",
            "warn",
            f"Query reports are present, but only {overlays} overlay artifacts were counted.",
            "Generate demo overlays so reviewers can inspect relevancy qualitatively.",
            "demo_assets/",
        )
    return QualityGateCriterion(
        "query_coverage",
        "pass",
        f"Query evidence passes profile threshold ({query_reports} reports, {overlays} overlays).",
        artifact="queries/",
    )


def _evaluation_criterion(
    scorecard: dict[str, Any],
    evaluation: dict[str, Any],
    policy: GatePolicy,
) -> QualityGateCriterion:
    evaluated = _evaluated_query_count(scorecard, evaluation)
    if evaluated < policy.min_evaluated_queries:
        return QualityGateCriterion(
            "evaluation_coverage",
            "fail",
            f"Only {evaluated} evaluated queries found; profile requires {policy.min_evaluated_queries}.",
            (
                "Add manual bbox_2d annotations with the workbench, then run "
                "finalize_annotations.py to refresh evaluation and reports."
            ),
            "evaluation/eval_summary.json",
        )
    if evaluated == 0:
        return QualityGateCriterion(
            "evaluation_coverage",
            "pass",
            "Quantitative evaluation is optional for the smoke profile.",
            artifact="evaluation/eval_summary.json",
        )
    return QualityGateCriterion(
        "evaluation_coverage",
        "pass",
        f"Evaluation coverage passes profile threshold ({evaluated} evaluated queries).",
        artifact="evaluation/eval_summary.json",
    )


def _annotation_criterion(
    annotations: dict[str, Any],
    policy: GatePolicy,
) -> QualityGateCriterion:
    if not annotations:
        return QualityGateCriterion(
            "annotations",
            "warn",
            "annotation_validation.json is missing or unreadable.",
            "Validate manual annotations before reporting quantitative scores.",
            "evaluation/annotation_validation.json",
        )
    if annotations.get("ok") is False:
        return QualityGateCriterion(
            "annotations",
            "fail",
            "Annotation validation reports invalid annotations.",
            "Fix annotation schema, duplicate queries, view ids, or bbox values.",
            "evaluation/annotation_validation.json",
        )
    warnings = [str(warning) for warning in annotations.get("warnings") or []]
    if warnings:
        gate_status: GateStatus = "fail" if policy.fail_on_annotation_warnings else "warn"
        return QualityGateCriterion(
            "annotations",
            gate_status,
            "Annotation validation has warnings: " + "; ".join(warnings[:3]),
            "Resolve annotation coverage and view-id mismatches before final reporting.",
            "evaluation/annotation_validation.json",
        )
    return QualityGateCriterion(
        "annotations",
        "pass",
        "Annotation validation is clean.",
        artifact="evaluation/annotation_validation.json",
    )


def _pack_criterion(pack_dir: str | Path | None, policy: GatePolicy) -> QualityGateCriterion:
    if pack_dir is None:
        status: GateStatus = "fail" if policy.require_pack else "warn"
        return QualityGateCriterion(
            "portfolio_pack",
            status,
            "No exported portfolio pack was provided for validation.",
            "Run finalize_annotations.py with --export-pack --zip-pack, then pass --pack for final sharing checks.",
            "results/portfolio_pack",
        )
    validation = validate_portfolio_pack(pack_dir)
    if not validation.ok:
        return QualityGateCriterion(
            "portfolio_pack",
            "fail",
            "Portfolio pack validation failed.",
            "Open portfolio_pack_validation.json and fix missing artifacts, bad links, or path leaks.",
            "portfolio_pack_validation.json",
        )
    if validation.warnings:
        return QualityGateCriterion(
            "portfolio_pack",
            "warn",
            "Portfolio pack is structurally valid but has warnings.",
            "Review pack warnings before sharing externally.",
            "portfolio_pack_validation.json",
        )
    return QualityGateCriterion(
        "portfolio_pack",
        "pass",
        "Portfolio pack validation passed without warnings.",
        artifact="portfolio_pack_validation.json",
    )


def _query_report_count(root: Path, scorecard: dict[str, Any]) -> int:
    count = _safe_int(scorecard.get("query_report_count"))
    return count if count else len(list((root / "queries").rglob("scene_query_report.json")))


def _evaluated_query_count(scorecard: dict[str, Any], evaluation: dict[str, Any]) -> int:
    return _safe_int(scorecard.get("evaluated_query_count")) or _safe_int(
        evaluation.get("num_evaluated_queries")
    )


def _overall_status(criteria: list[QualityGateCriterion]) -> GateStatus:
    if any(criterion.status == "fail" for criterion in criteria):
        return "fail"
    if any(criterion.status == "warn" for criterion in criteria):
        return "warn"
    return "pass"


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


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _criterion_lines(criteria: list[QualityGateCriterion]) -> list[str]:
    if not criteria:
        return ["- None."]
    lines: list[str] = []
    for criterion in criteria:
        lines.append(f"- [{criterion.status}] {criterion.name}: {criterion.message}")
        if criterion.recommendation:
            lines.append(f"  Recommendation: {criterion.recommendation}")
        if criterion.artifact:
            lines.append(f"  Artifact: `{criterion.artifact}`")
    return lines
