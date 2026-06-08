"""Validate exported portfolio packs before sharing them."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


PROJECT_REQUIRED_FILES = [
    "portfolio_pack_index.json",
    "project/README.md",
    "project/LICENSE",
    "project/CITATION.cff",
    "project/docs/index.html",
    "project/docs/portfolio_result_card.md",
    "project/docs/project_report.md",
    "project/docs/method_summary.md",
    "project/docs/cv_bullets.md",
    "project/docs/cold_email_paragraph.md",
    "project/docs/real_scene_capture_checklist.md",
    "project/docs/real_run_reproducibility.md",
    "project/docs/assets/query_grid.png",
    "project/docs/assets/demo_montage.gif",
]

RUN_REQUIRED_FILES = [
    "run/pipeline_summary.json",
    "run/capture_manifest.json",
    "run/capture_manifest.md",
    "run/capture_manifest_validation.json",
    "run/capture_manifest_validation.md",
    "run/preflight_report.json",
    "run/preflight_report.md",
    "run/evidence_scorecard.json",
    "run/evidence_scorecard.md",
    "run/quality_gate.json",
    "run/quality_gate.md",
    "run/claim_audit.json",
    "run/claim_audit.md",
    "run/run_result_card.json",
    "run/run_result_card.md",
    "run/run_audit.json",
    "run/run_audit.md",
    "run/run_recommendations.json",
    "run/run_recommendations.md",
    "run/research_report.json",
    "run/research_report.md",
    "run/real_run_plan/real_run_plan.json",
    "run/real_run_plan/real_run_plan.md",
    "run/submission_packet/submission_packet.json",
    "run/submission_packet/submission_checklist.md",
    "run/submission_packet/cv_project_entry.md",
    "run/submission_packet/professor_email_brief.md",
    "run/reproduction_manifest.json",
    "run/reproduction_report.md",
    "run/reproduce_run.sh",
    "run/environment_report.json",
    "run/scene_data_inspection.json",
    "run/scene_data_inspection.md",
    "run/queries.yaml",
    "run/annotation_template.json",
    "run/project_report.md",
    "run/portfolio_result_card.md",
    "run/portfolio_page.html",
    "run/evaluation/eval_summary.json",
    "run/evaluation/eval_table.csv",
    "run/evaluation/annotation_validation.json",
    "run/evaluation/annotation_review.json",
    "run/evaluation/annotation_review.md",
    "run/evaluation/annotation_review_contact_sheet.png",
    "run/evaluation/annotation_workbench/annotation_workbench.html",
    "run/evaluation/annotation_workbench/annotation_workbench_manifest.json",
    "run/evaluation/annotation_workbench/annotation_seed.json",
    "run/evaluation/qualitative_report.md",
    "run/demo_assets/demo_summary.json",
    "run/demo_assets/query_grid.png",
    "run/demo_assets/demo_montage.gif",
]

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {"LICENSE", "README", "README.md"}
PATH_LEAK_PATTERNS = {
    "windows_user_path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/][^\s\"'`<>|]+", re.IGNORECASE),
    "windows_appdata": re.compile(r"[A-Za-z]:[\\/]+Users[\\/][^\s\"'`<>|]+[\\/]AppData[\\/]", re.IGNORECASE),
    "escaped_windows_users": re.compile(r"\\\\Users\\\\|\\Users\\", re.IGNORECASE),
    "appdata_path": re.compile(r"AppData[\\/]+", re.IGNORECASE),
    "github_actions_workspace": re.compile(r"/home/runner/work/[^\s\"'`<>|]+", re.IGNORECASE),
    "github_actions_python_cache": re.compile(r"/opt/hostedtoolcache/[^\s\"'`<>|]+", re.IGNORECASE),
    "pytest_tmp": re.compile(r"/tmp/pytest-[^\s\"'`<>|]+", re.IGNORECASE),
}


@dataclass
class PathLeak:
    """Potential local machine path retained in a shareable pack."""

    file: str
    pattern: str
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class PortfolioValidationReport:
    """Structured validation result for an exported portfolio pack."""

    ok: bool
    pack_dir: str
    timestamp: str
    checked_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    path_leaks: list[PathLeak] = field(default_factory=list)
    artifact_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path_leaks"] = [leak.to_dict() for leak in self.path_leaks]
        return payload

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path


def validate_portfolio_pack(pack_dir: str | Path) -> PortfolioValidationReport:
    """Validate that a portfolio pack directory or zip archive is complete and share-safe."""

    pack_path = Path(pack_dir)
    errors: list[str] = []
    if not pack_path.exists():
        errors.append(f"Portfolio pack path does not exist: {pack_path}")
        return _report(pack_path, [], [], [], [], [], errors)
    if pack_path.is_file():
        if pack_path.suffix.lower() != ".zip":
            errors.append(f"Portfolio pack path is not a directory or .zip archive: {pack_path}")
            return _report(pack_path, [], [], [], [], [], errors)
        return _validate_portfolio_zip(pack_path)
    if not pack_path.is_dir():
        errors.append(f"Portfolio pack path is not a directory or .zip archive: {pack_path}")
        return _report(pack_path, [], [], [], [], [], errors)
    return _validate_portfolio_directory(pack_path)


def _validate_portfolio_zip(zip_path: Path) -> PortfolioValidationReport:
    errors: list[str] = []
    if not zipfile.is_zipfile(zip_path):
        errors.append(f"Portfolio pack zip is not a valid zip archive: {zip_path}")
        return _report(zip_path, [], [], [], [], [], errors)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            unsafe_members = _unsafe_zip_members(archive)
            if unsafe_members:
                preview = ", ".join(unsafe_members[:5])
                errors.append(f"Portfolio pack zip contains unsafe member paths: {preview}")
                return _report(zip_path, [], [], [], [], [], errors)
            with tempfile.TemporaryDirectory(prefix="nerf-portfolio-pack-") as tmp_dir:
                extract_root = Path(tmp_dir) / "pack"
                extract_root.mkdir()
                archive.extractall(extract_root)
                pack_root = _extracted_pack_root(extract_root)
                return _validate_portfolio_directory(pack_root, report_path=zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"Could not validate portfolio pack zip {zip_path}: {exc}")
        return _report(zip_path, [], [], [], [], [], errors)


def _validate_portfolio_directory(
    pack_path: Path,
    *,
    report_path: Path | None = None,
) -> PortfolioValidationReport:
    """Validate an already-extracted portfolio pack directory."""

    errors: list[str] = []
    warnings: list[str] = []
    missing_files: list[str] = []
    artifact_issues: list[str] = []

    _check_required_files(pack_path, PROJECT_REQUIRED_FILES, missing_files)
    if (pack_path / "run").exists():
        _check_required_files(pack_path, RUN_REQUIRED_FILES, missing_files)
        _check_run_logs(pack_path, artifact_issues)
        _check_run_audit(pack_path, warnings, errors)
        _check_capture_manifest(pack_path, warnings, errors)
        _check_evidence_scorecard(pack_path, warnings, errors)
        _check_quality_gate(pack_path, warnings, errors)
        _check_claim_audit(pack_path, warnings, errors)
        _check_run_result_card(pack_path, warnings, errors)
        _check_annotation_validation(pack_path, warnings, errors)
    else:
        warnings.append("No run/ directory found; pack includes project materials but no run-scoped evidence.")

    _check_index(pack_path, artifact_issues, warnings, errors)
    checked_files, path_leaks = _scan_path_leaks(pack_path)

    return _report(
        report_path or pack_path,
        checked_files,
        missing_files,
        path_leaks,
        artifact_issues,
        warnings,
        errors,
    )


def _report(
    pack_path: Path,
    checked_files: list[str],
    missing_files: list[str],
    path_leaks: list[PathLeak],
    artifact_issues: list[str],
    warnings: list[str],
    errors: list[str],
) -> PortfolioValidationReport:
    ok = not (missing_files or path_leaks or artifact_issues or errors)
    return PortfolioValidationReport(
        ok=ok,
        pack_dir=_display_pack_dir(pack_path),
        timestamp=utc_timestamp(),
        checked_files=checked_files,
        missing_files=missing_files,
        path_leaks=path_leaks,
        artifact_issues=artifact_issues,
        warnings=warnings,
        errors=errors,
    )


def _check_required_files(pack_path: Path, relative_paths: list[str], missing_files: list[str]) -> None:
    for relative_path in relative_paths:
        if not (pack_path / relative_path).exists():
            missing_files.append(relative_path)


def _extracted_pack_root(extract_root: Path) -> Path:
    if (extract_root / "portfolio_pack_index.json").exists():
        return extract_root
    children = [path for path in extract_root.iterdir() if path.name != "__MACOSX"]
    directories = [path for path in children if path.is_dir()]
    files = [path for path in children if path.is_file()]
    if not files and len(directories) == 1 and (directories[0] / "portfolio_pack_index.json").exists():
        return directories[0]
    return extract_root


def _unsafe_zip_members(archive: zipfile.ZipFile) -> list[str]:
    unsafe: list[str] = []
    for info in archive.infolist():
        name = info.filename
        normalized = name.replace("\\", "/")
        stripped = normalized.rstrip("/")
        parts = [part for part in stripped.split("/") if part]
        if (
            not stripped
            or normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or "\x00" in normalized
            or any(part == ".." for part in parts)
        ):
            unsafe.append(name)
    return unsafe


def _check_index(
    pack_path: Path,
    artifact_issues: list[str],
    warnings: list[str],
    errors: list[str],
) -> None:
    index_path = pack_path / "portfolio_pack_index.json"
    index = _read_json(index_path, errors)
    if not isinstance(index, dict):
        return

    missing = index.get("missing")
    if missing:
        errors.append("portfolio_pack_index.json reports missing files: " + ", ".join(map(str, missing)))
    if not isinstance(missing, list):
        errors.append("portfolio_pack_index.json field 'missing' must be a list.")

    optional_missing = index.get("optional_missing") or []
    if optional_missing:
        warnings.append("Optional files were not exported: " + ", ".join(map(str, optional_missing)))

    for item in index.get("copied") or []:
        if not isinstance(item, dict):
            artifact_issues.append("portfolio_pack_index.json copied entries must be objects.")
            continue
        destination = item.get("destination")
        if not isinstance(destination, str) or not destination:
            artifact_issues.append("portfolio_pack_index.json copied entry is missing destination.")
            continue
        _check_relative_artifact(pack_path, destination, artifact_issues, label="copied destination")
        _check_copied_digest(pack_path, item, destination, artifact_issues, warnings)

    run_summary = index.get("run_summary")
    if run_summary is None:
        if (pack_path / "run").exists():
            warnings.append("Pack has run/ artifacts but no run_summary in portfolio_pack_index.json.")
        return
    if not isinstance(run_summary, dict):
        errors.append("portfolio_pack_index.json field 'run_summary' must be an object or null.")
        return
    if run_summary.get("success") is not True:
        errors.append("run_summary.success is not true; this pack should not be used as a successful demo.")
    artifacts = run_summary.get("artifacts")
    if not isinstance(artifacts, dict):
        artifact_issues.append("run_summary.artifacts must be an object.")
        return
    for name, relative_path in artifacts.items():
        if not isinstance(relative_path, str) or not relative_path:
            artifact_issues.append(f"run_summary artifact '{name}' must be a non-empty relative path.")
            continue
        _check_relative_artifact(pack_path, relative_path, artifact_issues, label=f"run_summary artifact '{name}'")


def _check_relative_artifact(
    pack_path: Path,
    relative_path: str,
    artifact_issues: list[str],
    *,
    label: str,
) -> None:
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        artifact_issues.append(f"{label} points outside the pack: {relative_path}")
        return
    candidate = (pack_path / relative_path).resolve()
    try:
        candidate.relative_to(pack_path.resolve())
    except ValueError:
        artifact_issues.append(f"{label} points outside the pack: {relative_path}")
        return
    if relative_path.endswith("/"):
        if not candidate.is_dir():
            artifact_issues.append(f"{label} directory is missing: {relative_path}")
        elif not any(candidate.iterdir()):
            artifact_issues.append(f"{label} directory is empty: {relative_path}")
    elif not candidate.exists():
        artifact_issues.append(f"{label} is missing: {relative_path}")


def _check_copied_digest(
    pack_path: Path,
    item: dict[str, Any],
    destination: str,
    artifact_issues: list[str],
    warnings: list[str],
) -> None:
    candidate = pack_path / destination
    if not candidate.is_file():
        return
    expected_sha = item.get("sha256")
    expected_size = item.get("size_bytes")
    if expected_sha is None and expected_size is None:
        warnings.append(f"copied destination has no integrity digest: {destination}")
        return
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        artifact_issues.append(f"copied destination has invalid sha256 digest: {destination}")
    else:
        actual_sha = _sha256(candidate)
        if actual_sha != expected_sha:
            artifact_issues.append(f"copied destination sha256 mismatch: {destination}")
    if not isinstance(expected_size, int):
        artifact_issues.append(f"copied destination has invalid size_bytes: {destination}")
    elif candidate.stat().st_size != expected_size:
        artifact_issues.append(f"copied destination size mismatch: {destination}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_run_logs(pack_path: Path, artifact_issues: list[str]) -> None:
    logs_dir = pack_path / "run" / "logs"
    if not logs_dir.exists():
        artifact_issues.append("run/logs/ is missing; command provenance was not packaged.")
        return
    if not list(logs_dir.glob("*.json")):
        artifact_issues.append("run/logs/ does not contain any JSON command logs.")


def _check_run_audit(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    audit = _read_json(pack_path / "run" / "run_audit.json", errors)
    if not isinstance(audit, dict):
        return
    status = str(audit.get("status") or "")
    if status == "blocked":
        errors.append("run_audit.json status is blocked.")
    elif status == "needs_review":
        warnings.append("run_audit.json status is needs_review; inspect warnings before sharing.")
    elif status and status != "ready":
        warnings.append(f"run_audit.json has unrecognized status: {status}")


def _check_capture_manifest(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    validation = _read_json(pack_path / "run" / "capture_manifest_validation.json", errors)
    if not isinstance(validation, dict):
        return
    status = str(validation.get("status") or "")
    if status == "blocked":
        errors.append("capture_manifest_validation.json status is blocked.")
    elif status == "needs_review":
        warnings.append("capture_manifest_validation.json status is needs_review; fill capture metadata before sharing real results.")
    elif status and status != "ready":
        warnings.append(f"capture_manifest_validation.json has unrecognized status: {status}")


def _check_evidence_scorecard(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    scorecard = _read_json(pack_path / "run" / "evidence_scorecard.json", errors)
    if not isinstance(scorecard, dict):
        return
    level = str(scorecard.get("evidence_level") or "")
    if level == "blocked":
        errors.append("evidence_scorecard.json level is blocked.")
    elif level == "dry_run_demo_ready":
        warnings.append("evidence_scorecard.json level is dry_run_demo_ready; share as a smoke demo, not a real trained-scene result.")
    elif level in {"needs_evidence", "needs_review"}:
        warnings.append(f"evidence_scorecard.json level is {level}; inspect scorecard before sharing.")
    elif level and level not in {"dry_run_demo_ready", "portfolio_ready_real_run"}:
        warnings.append(f"evidence_scorecard.json has unrecognized level: {level}")
    if scorecard.get("dry_run") is True and level == "portfolio_ready_real_run":
        errors.append("Dry-run scorecard cannot be portfolio_ready_real_run.")


def _check_quality_gate(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    gate = _read_json(pack_path / "run" / "quality_gate.json", errors)
    if not isinstance(gate, dict):
        return
    status = str(gate.get("status") or "")
    if gate.get("passed") is False or status == "fail":
        errors.append("quality_gate.json reports a failed run quality gate.")
    elif status == "warn":
        warnings.append("quality_gate.json status is warn; inspect criteria before sharing.")
    elif status and status != "pass":
        warnings.append(f"quality_gate.json has unrecognized status: {status}")


def _check_claim_audit(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    audit = _read_json(pack_path / "run" / "claim_audit.json", errors)
    if not isinstance(audit, dict):
        return
    status = str(audit.get("status") or "")
    if audit.get("ok") is False or status == "fail":
        errors.append("claim_audit.json reports unsupported external-facing claims.")
    elif status == "warn":
        warnings.append("claim_audit.json status is warn; inspect claim_audit.md before sharing.")
    elif status and status != "pass":
        warnings.append(f"claim_audit.json has unrecognized status: {status}")


def _check_run_result_card(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    card = _read_json(pack_path / "run" / "run_result_card.json", errors)
    if not isinstance(card, dict):
        return
    status = str(card.get("result_status") or "")
    if status == "blocked":
        errors.append("run_result_card.json result_status is blocked.")
    elif status in {"shareable_smoke_demo", "dry_run_smoke_demo"}:
        warnings.append("run_result_card.json marks this as smoke evidence, not a real trained-scene result.")
    elif status in {"needs_evidence", "real_run_review_ready"}:
        warnings.append(f"run_result_card.json result_status is {status}; inspect before sharing.")
    elif status and status != "portfolio_ready":
        warnings.append(f"run_result_card.json has unrecognized result_status: {status}")


def _check_annotation_validation(pack_path: Path, warnings: list[str], errors: list[str]) -> None:
    validation = _read_json(pack_path / "run" / "evaluation" / "annotation_validation.json", errors)
    if not isinstance(validation, dict):
        return
    if validation.get("ok") is False:
        errors.append("annotation_validation.json reports invalid annotations.")
    for warning in validation.get("warnings") or []:
        warnings.append(f"Annotation validation warning: {warning}")


def _scan_path_leaks(pack_path: Path) -> tuple[list[str], list[PathLeak]]:
    checked_files: list[str] = []
    leaks: list[PathLeak] = []
    for path in sorted(pack_path.rglob("*")):
        if not path.is_file() or not _is_text_file(path):
            continue
        relative_path = _relative_path(path, pack_path)
        checked_files.append(relative_path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATH_LEAK_PATTERNS.items():
            match = pattern.search(text)
            if match:
                leaks.append(
                    PathLeak(
                        file=relative_path,
                        pattern=name,
                        excerpt=_excerpt(text, match.start(), match.end()),
                    )
                )
    return checked_files, leaks


def _read_json(path: Path, errors: list[str]) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Could not parse JSON {path.name}: {exc}")
        return None


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES


def _excerpt(text: str, start: int, end: int) -> str:
    left = max(0, start - 40)
    right = min(len(text), end + 40)
    return text[left:right].replace("\n", " ")


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _display_pack_dir(path: Path) -> str:
    """Return a share-safe pack identifier for reports that may live inside the pack."""

    try:
        return path.resolve().name or "portfolio_pack"
    except OSError:
        return path.name or "portfolio_pack"
