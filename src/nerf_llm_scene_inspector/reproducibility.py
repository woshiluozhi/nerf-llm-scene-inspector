"""Create portable reproduction manifests from pipeline run outputs."""

from __future__ import annotations

import json
import hashlib
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


MUTABLE_ARTIFACT_NAMES = {"annotation_finalize"}


@dataclass
class ReproductionArtifact:
    """One artifact expected inside a reproducible run directory."""

    name: str
    path: str
    exists: bool
    purpose: str
    kind: str = "missing"
    size_bytes: int | None = None
    sha256: str | None = None
    file_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class ReproductionBundle:
    """Portable instructions and evidence paths for reproducing one pipeline run."""

    run_dir: str
    scene_name: str
    dry_run: bool
    backend: str
    generated_at: str
    source_command: list[str] = field(default_factory=list)
    replay_command: str = ""
    prerequisites: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    artifacts: list[ReproductionArtifact] = field(default_factory=list)
    artifact_summary: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "generated_at": self.generated_at,
            "source_command": list(self.source_command),
            "replay_command": self.replay_command,
            "prerequisites": list(self.prerequisites),
            "verification_commands": list(self.verification_commands),
            "queries": list(self.queries),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "artifact_summary": dict(self.artifact_summary),
            "notes": list(self.notes),
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
            "# Reproduction Report",
            "",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Backend: {self.backend or 'unknown'}",
            f"- Dry run: {self.dry_run}",
            f"- Run directory: `{self.run_dir}`",
            "",
            "## Replay Command",
            "",
            "```bash",
            self.replay_command or "# No replay command recorded.",
            "```",
            "",
            "## Verification",
            "",
            *_command_lines(self.verification_commands),
            "",
            "## Key Artifacts",
            "",
            *_artifact_summary_lines(self.artifact_summary),
            "",
            *_artifact_lines(self.artifacts),
            "",
            "## Notes",
            "",
            *_note_lines(self.notes),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def to_shell_script(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "python -m pip install -e \".[dev,video]\"",
            "python scripts/check_env.py --json",
        ]
        if not self.dry_run:
            lines.append("python scripts/check_env.py --check-upstream --require-gpu --verbose")
        if self.replay_command:
            lines.append(self.replay_command)
        lines.extend(self.verification_commands)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


@dataclass
class ReproductionManifestIssue:
    """One integrity issue found while checking a reproduction manifest."""

    artifact_name: str
    path: str
    category: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ReproductionManifestValidation:
    """Integrity report for a run-local reproduction manifest."""

    ok: bool
    run_dir: str
    manifest_path: str
    timestamp: str
    require_complete: bool
    checked_artifacts: int = 0
    matched_files: int = 0
    matched_directories: int = 0
    recorded_missing: int = 0
    issues: list[ReproductionManifestIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ReproductionManifestIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ReproductionManifestIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "run_dir": self.run_dir,
            "manifest_path": self.manifest_path,
            "timestamp": self.timestamp,
            "require_complete": self.require_complete,
            "checked_artifacts": self.checked_artifacts,
            "matched_files": self.matched_files,
            "matched_directories": self.matched_directories,
            "recorded_missing": self.recorded_missing,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
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
            "# Reproduction Manifest Validation",
            "",
            f"- Status: {'pass' if self.ok else 'fail'}",
            f"- Run directory: `{self.run_dir}`",
            f"- Manifest: `{self.manifest_path}`",
            f"- Require complete: {self.require_complete}",
            f"- Checked artifacts: {self.checked_artifacts}",
            f"- Matched files: {self.matched_files}",
            f"- Matched directories: {self.matched_directories}",
            f"- Recorded missing artifacts: {self.recorded_missing}",
            f"- Errors: {len(self.errors)}",
            f"- Warnings: {len(self.warnings)}",
            "",
            "## Issues",
            "",
            *_manifest_issue_lines(self.issues),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_reproduction_bundle(run_dir: str | Path) -> ReproductionBundle:
    """Build replay and verification instructions from a pipeline run directory."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    scene_name = str(summary.get("scene_name") or root.name)
    dry_run = bool(summary.get("dry_run"))
    backend = str(summary.get("backend") or "")
    source_command = _source_command(summary)
    artifacts = _artifacts(root)
    return ReproductionBundle(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        dry_run=dry_run,
        backend=backend,
        generated_at=utc_timestamp(),
        source_command=source_command,
        replay_command=_replay_command(source_command, summary),
        prerequisites=_prerequisites(dry_run=dry_run),
        verification_commands=_verification_commands(root),
        queries=[str(query) for query in summary.get("queries") or []],
        artifacts=artifacts,
        artifact_summary=_artifact_summary(artifacts),
        notes=_notes(dry_run=dry_run),
    )


def verify_reproduction_manifest(
    run_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    require_complete: bool = False,
) -> ReproductionManifestValidation:
    """Verify file sizes and SHA256 digests recorded in a reproduction manifest."""

    root = Path(run_dir)
    manifest = Path(manifest_path) if manifest_path is not None else root / "reproduction_manifest.json"
    issues: list[ReproductionManifestIssue] = []
    if not manifest.exists():
        issues.append(
            _manifest_issue(
                "",
                str(manifest),
                "missing_manifest",
                "error",
                "Reproduction manifest does not exist.",
            )
        )
        return _validation_report(root, manifest, require_complete, 0, 0, 0, 0, issues)

    payload = _read_manifest_payload(manifest, issues)
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(artifacts, list):
        issues.append(
            _manifest_issue(
                "",
                _display_validation_path(manifest),
                "invalid_manifest",
                "error",
                "Manifest field 'artifacts' must be a list.",
            )
        )
        return _validation_report(root, manifest, require_complete, 0, 0, 0, 0, issues)

    matched_files = 0
    matched_directories = 0
    recorded_missing = 0
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, dict):
            issues.append(
                _manifest_issue(
                    "",
                    "",
                    "invalid_artifact",
                    "error",
                    "Manifest artifact entries must be objects.",
                )
            )
            continue
        name = str(raw_artifact.get("name") or "")
        relative_path = str(raw_artifact.get("path") or "")
        if not relative_path:
            issues.append(
                _manifest_issue(name, "", "invalid_artifact_path", "error", "Artifact path is missing.")
            )
            continue
        if Path(relative_path).is_absolute():
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "absolute_artifact_path",
                    "error",
                    "Artifact path must be relative to the run directory.",
                )
            )
            continue
        artifact_path = root / relative_path
        expected_exists = bool(raw_artifact.get("exists"))
        expected_kind = str(raw_artifact.get("kind") or "")
        if not expected_exists:
            recorded_missing += 1
            severity = "error" if require_complete else "warning"
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "recorded_missing_artifact",
                    severity,
                    "Artifact was recorded as missing in the manifest.",
                )
            )
            continue
        if not artifact_path.exists():
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "missing_artifact",
                    "error",
                    "Artifact existed when the manifest was written but is missing now.",
                )
            )
            continue
        if expected_kind == "file":
            matched_files += _verify_file_artifact(raw_artifact, artifact_path, name, relative_path, issues)
        elif expected_kind == "directory":
            matched_directories += _verify_directory_artifact(
                raw_artifact, artifact_path, name, relative_path, issues
            )
        else:
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "unknown_artifact_kind",
                    "warning",
                    f"Artifact kind is {expected_kind or 'unset'}; only existence was checked.",
                )
            )
    _check_artifact_summary(payload, artifacts, issues)
    return _validation_report(
        root,
        manifest,
        require_complete,
        len(artifacts),
        matched_files,
        matched_directories,
        recorded_missing,
        issues,
    )


def _read_manifest_payload(path: Path, issues: list[ReproductionManifestIssue]) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(
            _manifest_issue(
                "",
                _display_validation_path(path),
                "invalid_json",
                "error",
                f"Could not parse reproduction manifest JSON: {exc}",
            )
        )
        return {}
    return raw if isinstance(raw, dict) else {}


def _verify_file_artifact(
    raw_artifact: dict[str, Any],
    artifact_path: Path,
    name: str,
    relative_path: str,
    issues: list[ReproductionManifestIssue],
) -> int:
    start_error_count = sum(1 for issue in issues if issue.severity == "error")
    if not artifact_path.is_file():
        issues.append(
            _manifest_issue(
                name,
                relative_path,
                "artifact_kind_mismatch",
                "error",
                "Manifest recorded a file, but the path is not a file.",
            )
        )
        return 0
    expected_size = raw_artifact.get("size_bytes")
    if isinstance(expected_size, int):
        current_size = artifact_path.stat().st_size
        if current_size != expected_size:
            severity = _integrity_mismatch_severity(name)
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "size_mismatch",
                    severity,
                    f"Expected {expected_size} bytes, found {current_size} bytes.",
                )
            )
    else:
        issues.append(
            _manifest_issue(
                name,
                relative_path,
                "missing_size_digest",
                "warning",
                "File artifact does not record size_bytes.",
            )
        )
    expected_sha = raw_artifact.get("sha256")
    if isinstance(expected_sha, str) and expected_sha:
        current_sha = _sha256(artifact_path)
        if current_sha != expected_sha:
            severity = _integrity_mismatch_severity(name)
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "sha256_mismatch",
                    severity,
                    "SHA256 digest does not match the current file contents.",
                )
            )
    else:
        issues.append(
            _manifest_issue(
                name,
                relative_path,
                "missing_sha256_digest",
                "warning",
                "File artifact does not record a SHA256 digest.",
            )
        )
    end_error_count = sum(1 for issue in issues if issue.severity == "error")
    return 1 if end_error_count == start_error_count else 0


def _integrity_mismatch_severity(artifact_name: str) -> str:
    return "warning" if artifact_name in MUTABLE_ARTIFACT_NAMES else "error"


def _verify_directory_artifact(
    raw_artifact: dict[str, Any],
    artifact_path: Path,
    name: str,
    relative_path: str,
    issues: list[ReproductionManifestIssue],
) -> int:
    if not artifact_path.is_dir():
        issues.append(
            _manifest_issue(
                name,
                relative_path,
                "artifact_kind_mismatch",
                "error",
                "Manifest recorded a directory, but the path is not a directory.",
            )
        )
        return 0
    expected_count = raw_artifact.get("file_count")
    if isinstance(expected_count, int):
        current_count = sum(1 for item in artifact_path.rglob("*") if item.is_file())
        if current_count != expected_count:
            issues.append(
                _manifest_issue(
                    name,
                    relative_path,
                    "directory_file_count_mismatch",
                    "warning",
                    f"Expected {expected_count} files, found {current_count} files.",
                )
            )
            return 0
    else:
        issues.append(
            _manifest_issue(
                name,
                relative_path,
                "missing_directory_file_count",
                "warning",
                "Directory artifact does not record file_count.",
            )
        )
    return 1


def _check_artifact_summary(
    payload: dict[str, Any],
    artifacts: list[Any],
    issues: list[ReproductionManifestIssue],
) -> None:
    summary = payload.get("artifact_summary")
    if not isinstance(summary, dict):
        issues.append(
            _manifest_issue(
                "",
                "artifact_summary",
                "missing_artifact_summary",
                "warning",
                "Manifest does not contain an artifact_summary object.",
            )
        )
        return
    expected = {
        "total": len(artifacts),
        "existing": sum(1 for artifact in artifacts if isinstance(artifact, dict) and artifact.get("exists")),
        "missing": sum(1 for artifact in artifacts if isinstance(artifact, dict) and not artifact.get("exists")),
        "files": sum(1 for artifact in artifacts if isinstance(artifact, dict) and artifact.get("kind") == "file"),
        "directories": sum(
            1 for artifact in artifacts if isinstance(artifact, dict) and artifact.get("kind") == "directory"
        ),
        "total_size_bytes": sum(
            int(artifact.get("size_bytes"))
            for artifact in artifacts
            if isinstance(artifact, dict) and isinstance(artifact.get("size_bytes"), int)
        ),
    }
    for key, expected_value in expected.items():
        observed_value = summary.get(key)
        if observed_value != expected_value:
            issues.append(
                _manifest_issue(
                    "",
                    f"artifact_summary.{key}",
                    "artifact_summary_mismatch",
                    "warning",
                    f"Expected {expected_value}, found {observed_value}.",
                )
            )


def _validation_report(
    run_dir: Path,
    manifest_path: Path,
    require_complete: bool,
    checked_artifacts: int,
    matched_files: int,
    matched_directories: int,
    recorded_missing: int,
    issues: list[ReproductionManifestIssue],
) -> ReproductionManifestValidation:
    return ReproductionManifestValidation(
        ok=not any(issue.severity == "error" for issue in issues),
        run_dir=_display_run_dir(run_dir),
        manifest_path=_display_validation_path(manifest_path),
        timestamp=utc_timestamp(),
        require_complete=require_complete,
        checked_artifacts=checked_artifacts,
        matched_files=matched_files,
        matched_directories=matched_directories,
        recorded_missing=recorded_missing,
        issues=issues,
    )


def _manifest_issue(
    artifact_name: str,
    path: str,
    category: str,
    severity: str,
    message: str,
) -> ReproductionManifestIssue:
    return ReproductionManifestIssue(
        artifact_name=artifact_name,
        path=path.replace("\\", "/"),
        category=category,
        severity=severity,
        message=message,
    )


def _source_command(summary: dict[str, Any]) -> list[str]:
    raw = summary.get("provenance", {}).get("command") if isinstance(summary.get("provenance"), dict) else []
    return [str(item) for item in raw or []]


def _replay_command(source_command: list[str], summary: dict[str, Any]) -> str:
    if source_command:
        command = list(source_command)
        first = command[0].replace("\\", "/")
        if first.endswith(".py"):
            command = ["python", first, *command[1:]]
        else:
            command[0] = first
        return _format_command(command)
    scene = str(summary.get("scene_name") or "desk_scene")
    queries = [str(query) for query in summary.get("queries") or []]
    command = [
        "python",
        "scripts/run_scene_pipeline.py",
        "--scene-name",
        scene,
        "--backend",
        str(summary.get("backend") or "lerf"),
    ]
    if summary.get("dry_run"):
        command.append("--dry-run")
    for query in queries:
        command.extend(["--query", query])
    return _format_command(command)


def _verification_commands(root: Path) -> list[str]:
    run_dir = _display_run_dir(root)
    runs_root = _display_path(root.parent)
    return [
        _format_command(["python", "scripts/verify_reproduction_manifest.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/audit_query_evidence.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/diagnose_run_failures.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/audit_run.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/recommend_next_steps.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/generate_research_report.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_evidence_scorecard.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/check_run_quality.py", "--run-dir", run_dir, "--profile", "smoke"]),
        _format_command(["python", "scripts/generate_research_report.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_run_result_card.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/generate_portfolio_page.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_submission_packet.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_real_run_plan.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_run_readiness.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/audit_claims.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/compare_runs.py", "--root", runs_root]),
        _format_command(
            [
                "python",
                "scripts/analyze_prompt_sensitivity.py",
                "--suite",
                "examples/prompt_sensitivity.yaml",
                "--results",
                f"{run_dir}/queries",
                "--output",
                f"{run_dir}/prompt_sensitivity",
            ]
        ),
        _format_command(
            [
                "python",
                "scripts/analyze_scene_relations.py",
                "--results",
                f"{run_dir}/queries",
                "--output",
                f"{run_dir}/scene_relations",
                "--scene-name",
                root.name,
                "--dry-run",
            ]
        ),
        _format_command(
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
        _format_command(
            [
                "python",
                "scripts/finalize_annotations.py",
                "--run-dir",
                run_dir,
                "--filled",
                f"{run_dir}/evaluation/annotation_workbench/annotation_seed.json",
                "--profile",
                "smoke",
                "--export-pack",
                "--zip-pack",
            ]
        ),
        _format_command(["python", "scripts/validate_portfolio_pack.py", "--pack", "results/portfolio_pack"]),
    ]


def _prerequisites(*, dry_run: bool) -> list[str]:
    items = [
        "Python 3.10+",
        "pip install -e .[dev,video]",
        "Git checkout matching the recorded commit when available",
    ]
    if dry_run:
        items.append("No GPU is required for dry-run reproduction.")
    else:
        items.extend(
            [
                "NVIDIA GPU with CUDA-compatible PyTorch",
                "Nerfstudio CLI installed",
                "LERF installed and registered with Nerfstudio",
                "COLMAP and FFmpeg available on PATH",
            ]
        )
    return items


def _artifacts(root: Path) -> list[ReproductionArtifact]:
    candidates = [
        ("pipeline_summary", root / "pipeline_summary.json", "pipeline_summary.json", "Top-level run status and provenance."),
        ("capture_manifest", root / "capture_manifest.md", "capture_manifest.md", "Scene-capture metadata and reproducibility context."),
        (
            "capture_manifest_validation",
            root / "capture_manifest_validation.md",
            "capture_manifest_validation.md",
            "Validation of capture conditions, overlap, static scene, and privacy review.",
        ),
        ("preflight_report", root / "preflight_report.md", "preflight_report.md", "Real-run readiness checks before training."),
        (
            "failure_diagnostics",
            root / "failure_diagnostics.md",
            "failure_diagnostics.md",
            "Classified command-log and training/query failure diagnostics.",
        ),
        ("environment_report", root / "environment_report.json", "environment_report.json", "Runtime and upstream dependency checks."),
        ("scene_inspection", root / "scene_data_inspection.md", "scene_data_inspection.md", "Processed scene quality and pose readiness."),
        ("run_audit", root / "run_audit.md", "run_audit.md", "Run health audit."),
        (
            "query_evidence_audit",
            root / "query_evidence_audit.md",
            "query_evidence_audit.md",
            "Per-query audit of visual overlays, localization evidence, fallback mode, and missing artifacts.",
        ),
        (
            "query_evidence_audit_json",
            root / "query_evidence_audit.json",
            "query_evidence_audit.json",
            "Machine-readable per-query evidence audit.",
        ),
        ("recommendations", root / "run_recommendations.md", "run_recommendations.md", "Actionable next steps."),
        ("evidence_scorecard", root / "evidence_scorecard.md", "evidence_scorecard.md", "Portfolio evidence quality scorecard."),
        ("quality_gate", root / "quality_gate.md", "quality_gate.md", "Pass/warn/fail run quality gate report."),
        ("claim_audit", root / "claim_audit.md", "claim_audit.md", "Audit report for avoiding unsupported external-facing claims."),
        ("run_result_card", root / "run_result_card.md", "run_result_card.md", "One-page reviewer-facing summary of what this run proves and does not prove."),
        ("portfolio_page", root / "portfolio_page.html", "portfolio_page.html", "Static HTML page for sharing run evidence."),
        (
            "real_run_plan",
            root / "real_run_plan" / "real_run_plan.md",
            "real_run_plan/real_run_plan.md",
            "Action plan for upgrading smoke evidence into a real CUDA/Nerfstudio/LERF run.",
        ),
        (
            "run_readiness",
            root / "run_readiness.md",
            "run_readiness.md",
            "Run-level readiness gate for real-run and external-review decisions.",
        ),
        ("research_report", root / "research_report.md", "research_report.md", "Paper-style report summarizing method, evidence, limitations, and next steps."),
        (
            "submission_checklist",
            root / "submission_packet" / "submission_checklist.md",
            "submission_packet/submission_checklist.md",
            "Claim-calibrated checklist for CV, portfolio, and professor outreach.",
        ),
        (
            "run_comparison",
            root.parent / "run_comparison.md",
            "../run_comparison.md",
            "Ranked comparison across repeated captures/training attempts.",
        ),
        ("query_grid", root / "demo_assets" / "query_grid.png", "demo_assets/query_grid.png", "Qualitative query visualization."),
        (
            "prompt_sensitivity",
            root / "prompt_sensitivity" / "prompt_sensitivity_report.md",
            "prompt_sensitivity/prompt_sensitivity_report.md",
            "Prompt wording stability report for open-vocabulary query variants.",
        ),
        (
            "scene_relations",
            root / "scene_relations" / "scene_relations_report.md",
            "scene_relations/scene_relations_report.md",
            "Scene-level object relation graph inferred from query boxes or 3D points.",
        ),
        ("evaluation_summary", root / "evaluation" / "eval_summary.json", "evaluation/eval_summary.json", "Quantitative/qualitative metric summary."),
        ("annotation_review", root / "evaluation" / "annotation_review.md", "evaluation/annotation_review.md", "Visual QA report for manual bbox annotations."),
        (
            "annotation_review_contact_sheet",
            root / "evaluation" / "annotation_review_contact_sheet.png",
            "evaluation/annotation_review_contact_sheet.png",
            "Contact sheet of manual bbox annotations over rendered views.",
        ),
        (
            "annotation_workbench",
            root / "evaluation" / "annotation_workbench" / "annotation_workbench.html",
            "evaluation/annotation_workbench/annotation_workbench.html",
            "Offline browser workbench for drawing and exporting manual bbox annotations.",
        ),
        (
            "annotation_finalize",
            root / "annotation_finalize_report.md",
            "annotation_finalize_report.md",
            "Post-workbench refresh report for merged annotations, evaluation, QA, and portfolio artifacts.",
        ),
        (
            "annotations_merged",
            root / "annotations_merged.json",
            "annotations_merged.json",
            "Clean evaluation annotation JSON merged from a filled workbench export.",
        ),
        (
            "annotation_merge_report",
            root / "annotation_merge_report.json",
            "annotation_merge_report.json",
            "Structured report for changed fields, missing queries, invalid boxes, and validation results.",
        ),
        ("portfolio_card", root / "portfolio_result_card.md", "portfolio_result_card.md", "Short project-page result narrative."),
        ("command_logs", root / "logs", "logs", "Subprocess command stdout/stderr records."),
    ]
    artifacts: list[ReproductionArtifact] = []
    for name, path, relative_path, purpose in candidates:
        artifacts.append(_artifact(name, path, relative_path, purpose))
    artifacts.extend(_query_artifacts(root))
    return artifacts


def _query_artifacts(root: Path) -> list[ReproductionArtifact]:
    query_root = root / "queries"
    if not query_root.exists() or not query_root.is_dir():
        return []
    artifacts: list[ReproductionArtifact] = []
    for task_dir in sorted(path for path in query_root.iterdir() if path.is_dir()):
        task_slug = task_dir.name
        artifacts.extend(
            [
                _artifact(
                    _artifact_name("query", task_slug, "report"),
                    task_dir / "scene_query_report.json",
                    f"queries/{task_slug}/scene_query_report.json",
                    "Machine-readable scene query report for one natural-language task.",
                ),
                _artifact(
                    _artifact_name("query", task_slug, "markdown"),
                    task_dir / "scene_query_report.md",
                    f"queries/{task_slug}/scene_query_report.md",
                    "Human-readable scene query report with evidence and caveats.",
                ),
                _artifact(
                    _artifact_name("query", task_slug, "visual_summary"),
                    task_dir / "query_visual_summary.json",
                    f"queries/{task_slug}/query_visual_summary.json",
                    "Compact summary of expanded visual prompts and query-grid artifact.",
                ),
                _artifact(
                    _artifact_name("query", task_slug, "grid"),
                    task_dir / "query_grid.png",
                    f"queries/{task_slug}/query_grid.png",
                    "Run-scoped grid of rendered overlays for this query task.",
                ),
            ]
        )
        for result_path in sorted(task_dir.glob("*/query_result.json")):
            expanded_slug = result_path.parent.name
            artifacts.append(
                _artifact(
                    _artifact_name("query", task_slug, expanded_slug, "result"),
                    result_path,
                    f"queries/{task_slug}/{expanded_slug}/query_result.json",
                    "Backend QueryResult for one expanded visual prompt.",
                )
            )
    return artifacts


def _artifact(name: str, path: Path, relative_path: str, purpose: str) -> ReproductionArtifact:
    return ReproductionArtifact(
        name=name,
        path=relative_path.replace("\\", "/"),
        exists=path.exists(),
        purpose=purpose,
        **_artifact_metadata(path),
    )


def _artifact_metadata(path: Path) -> dict[str, int | str | None]:
    if not path.exists():
        return {"kind": "missing"}
    if path.is_file():
        return {
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if path.is_dir():
        return {
            "kind": "directory",
            "file_count": sum(1 for item in path.rglob("*") if item.is_file()),
        }
    return {"kind": "other"}


def _artifact_summary(artifacts: list[ReproductionArtifact]) -> dict[str, int]:
    return {
        "total": len(artifacts),
        "existing": sum(1 for artifact in artifacts if artifact.exists),
        "missing": sum(1 for artifact in artifacts if not artifact.exists),
        "files": sum(1 for artifact in artifacts if artifact.kind == "file"),
        "directories": sum(1 for artifact in artifacts if artifact.kind == "directory"),
        "total_size_bytes": sum(artifact.size_bytes or 0 for artifact in artifacts),
    }


def _artifact_name(*parts: str) -> str:
    cleaned = [re.sub(r"[^a-zA-Z0-9]+", "_", part).strip("_").lower() for part in parts]
    return "_".join(part for part in cleaned if part)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notes(*, dry_run: bool) -> list[str]:
    notes = [
        "Reproduction commands assume they are run from the repository root.",
        "Run artifacts are intentionally separated from checked-in source files.",
    ]
    if dry_run:
        notes.append("Dry-run reproduction validates pipeline wiring, not trained NeRF/LERF quality.")
    else:
        notes.append("Real-scene reproduction depends on upstream Nerfstudio/LERF/CUDA versions and capture quality.")
    return notes


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item).replace("\\", "/")) for item in command)


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return "."


def _display_validation_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _command_lines(commands: list[str]) -> list[str]:
    if not commands:
        return ["- None."]
    return [f"- `{command}`" for command in commands]


def _artifact_lines(artifacts: list[ReproductionArtifact]) -> list[str]:
    if not artifacts:
        return ["- None."]
    return [f"- {artifact.name}: `{artifact.path}` ({_artifact_status(artifact)}) - {artifact.purpose}" for artifact in artifacts]


def _artifact_summary_lines(summary: dict[str, int]) -> list[str]:
    if not summary:
        return ["- Artifact summary unavailable."]
    return [
        f"- Existing artifacts: {summary.get('existing', 0)}/{summary.get('total', 0)}",
        f"- Files: {summary.get('files', 0)}",
        f"- Directories: {summary.get('directories', 0)}",
        f"- Total file bytes: {summary.get('total_size_bytes', 0)}",
    ]


def _artifact_status(artifact: ReproductionArtifact) -> str:
    if not artifact.exists:
        return "missing"
    details = [artifact.kind]
    if artifact.size_bytes is not None:
        details.append(f"{artifact.size_bytes} bytes")
    if artifact.sha256:
        details.append(f"sha256 {artifact.sha256[:12]}")
    if artifact.file_count is not None:
        details.append(f"{artifact.file_count} files")
    return ", ".join(details)


def _manifest_issue_lines(issues: list[ReproductionManifestIssue]) -> list[str]:
    if not issues:
        return ["- None."]
    return [
        (
            f"- {issue.severity.upper()} `{issue.category}`"
            f" {issue.artifact_name or '(manifest)'} `{issue.path}`: {issue.message}"
        )
        for issue in issues
    ]


def _note_lines(notes: list[str]) -> list[str]:
    if not notes:
        return ["- None."]
    return [f"- {note}" for note in notes]
