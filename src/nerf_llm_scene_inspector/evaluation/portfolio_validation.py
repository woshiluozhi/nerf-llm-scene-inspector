"""Validate exported portfolio packs before sharing them."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


PROJECT_REQUIRED_FILES = [
    "portfolio_pack_index.json",
    "project/README.md",
    "project/LICENSE",
    "project/CITATION.cff",
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
    "run/preflight_report.json",
    "run/preflight_report.md",
    "run/evidence_scorecard.json",
    "run/evidence_scorecard.md",
    "run/run_audit.json",
    "run/run_audit.md",
    "run/run_recommendations.json",
    "run/run_recommendations.md",
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
    "run/evaluation/eval_summary.json",
    "run/evaluation/eval_table.csv",
    "run/evaluation/annotation_validation.json",
    "run/evaluation/qualitative_report.md",
    "run/demo_assets/demo_summary.json",
    "run/demo_assets/query_grid.png",
    "run/demo_assets/demo_montage.gif",
]

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
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
    """Validate that a portfolio pack is complete, portable, and share-safe."""

    pack_path = Path(pack_dir)
    errors: list[str] = []
    warnings: list[str] = []
    missing_files: list[str] = []
    artifact_issues: list[str] = []

    if not pack_path.exists():
        errors.append(f"Portfolio pack directory does not exist: {pack_path}")
        return _report(pack_path, [], missing_files, [], artifact_issues, warnings, errors)
    if not pack_path.is_dir():
        errors.append(f"Portfolio pack path is not a directory: {pack_path}")
        return _report(pack_path, [], missing_files, [], artifact_issues, warnings, errors)

    _check_required_files(pack_path, PROJECT_REQUIRED_FILES, missing_files)
    if (pack_path / "run").exists():
        _check_required_files(pack_path, RUN_REQUIRED_FILES, missing_files)
        _check_run_logs(pack_path, artifact_issues)
        _check_run_audit(pack_path, warnings, errors)
        _check_evidence_scorecard(pack_path, warnings, errors)
        _check_annotation_validation(pack_path, warnings, errors)
    else:
        warnings.append("No run/ directory found; pack includes project materials but no run-scoped evidence.")

    _check_index(pack_path, artifact_issues, warnings, errors)
    checked_files, path_leaks = _scan_path_leaks(pack_path)

    return _report(
        pack_path,
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
        pack_dir=_display_path(pack_path),
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
