"""Structured diagnostics for failed or degraded real pipeline runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


DiagnosticSeverity = Literal["blocker", "warning", "info"]
DiagnosticStatus = Literal["clear", "needs_attention", "blocked"]


@dataclass(frozen=True)
class LogPattern:
    """One low-level log signature mapped to a run-level action."""

    category: str
    severity: DiagnosticSeverity
    pattern: str
    message: str
    recommendation: str
    command: str = ""


@dataclass
class FailureDiagnostic:
    """One actionable issue inferred from logs, summaries, or query reports."""

    severity: DiagnosticSeverity
    category: str
    message: str
    recommendation: str
    artifact: str = ""
    command: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FailureDiagnosticsReport:
    """Portable diagnostic report for one pipeline run."""

    run_dir: str
    scene_name: str
    status: DiagnosticStatus
    command_log_count: int
    failed_command_count: int
    training_summary_count: int
    query_report_count: int
    generated_at: str = field(default_factory=utc_timestamp)
    diagnostics: list[FailureDiagnostic] = field(default_factory=list)

    @property
    def blocker_count(self) -> int:
        return sum(1 for item in self.diagnostics if item.severity == "blocker")

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.diagnostics if item.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "status": self.status,
            "generated_at": self.generated_at,
            "command_log_count": self.command_log_count,
            "failed_command_count": self.failed_command_count,
            "training_summary_count": self.training_summary_count,
            "query_report_count": self.query_report_count,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
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
            "# Failure Diagnostics",
            "",
            "This report converts command logs, training summaries, and query reports into actionable debugging signals.",
            "",
            "## Summary",
            "",
            f"- Scene: `{self.scene_name or 'unknown'}`",
            f"- Run directory: `{self.run_dir}`",
            f"- Status: `{self.status}`",
            f"- Command logs inspected: {self.command_log_count}",
            f"- Failed commands: {self.failed_command_count}",
            f"- Training summaries inspected: {self.training_summary_count}",
            f"- Query reports inspected: {self.query_report_count}",
            f"- Blockers: {self.blocker_count}",
            f"- Warnings: {self.warning_count}",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Diagnostics",
            "",
            *_diagnostic_lines(self.diagnostics),
            "",
            "## Notes",
            "",
            "- Raw stdout and stderr are preserved in `logs/`; this report stores categories and repair actions rather than copying long logs.",
            "- A clear report means no known failure signatures were found, not that upstream model quality is guaranteed.",
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


LOG_PATTERNS = [
    LogPattern(
        category="cuda_oom",
        severity="blocker",
        pattern=r"cuda out of memory|cublas_status_alloc_failed|cudnn_status_alloc_failed|out of memory",
        message="GPU memory exhaustion was detected in command output.",
        recommendation=(
            "Reduce training/rendering memory pressure: use lerf-lite, lower image resolution, "
            "reduce rays/batch size in the Nerfstudio config, or use a larger GPU."
        ),
    ),
    LogPattern(
        category="cuda_unavailable",
        severity="blocker",
        pattern=r"no nvidia driver|cuda.*not available|torch not compiled with cuda|found no nvidia|no cuda",
        message="CUDA or the NVIDIA runtime was not available.",
        recommendation=(
            "Run on an NVIDIA GPU machine with CUDA-compatible PyTorch, then rerun "
            "python scripts/check_env.py --check-upstream --require-gpu --verbose."
        ),
        command="python scripts/check_env.py --check-upstream --require-gpu --verbose",
    ),
    LogPattern(
        category="missing_nerfstudio_cli",
        severity="blocker",
        pattern=r"ns-train.*not found|required executable 'ns-train'|no such file.*ns-train",
        message="The Nerfstudio training CLI was not found.",
        recommendation="Install Nerfstudio in the active environment and run ns-install-cli.",
        command="python -m pip install nerfstudio && ns-install-cli && ns-train -h",
    ),
    LogPattern(
        category="missing_process_data_cli",
        severity="blocker",
        pattern=r"ns-process-data.*not found|required executable 'ns-process-data'|no such file.*ns-process-data",
        message="The Nerfstudio data-processing CLI was not found.",
        recommendation="Install Nerfstudio in the active environment and confirm ns-process-data --help works.",
        command="python -m pip install nerfstudio && ns-install-cli && ns-process-data --help",
    ),
    LogPattern(
        category="lerf_method_missing",
        severity="blocker",
        pattern=r"does not list method|invalid choice.*lerf|unknown method.*lerf|method.*lerf.*not",
        message="LERF methods were not registered with Nerfstudio.",
        recommendation="Install LERF editable in the same environment as Nerfstudio and rerun ns-install-cli.",
        command="git clone https://github.com/kerrj/lerf && cd lerf && python -m pip install -e . && ns-install-cli && ns-train -h",
    ),
    LogPattern(
        category="missing_colmap",
        severity="blocker",
        pattern=r"colmap.*not found|required executable 'colmap'|no such file.*colmap",
        message="COLMAP was not available for camera-pose reconstruction.",
        recommendation="Install COLMAP and rerun data preparation.",
        command="conda install -c conda-forge colmap",
    ),
    LogPattern(
        category="colmap_reconstruction_failed",
        severity="blocker",
        pattern=r"no good initial image pair|reconstruction failed|mapper failed|colmap.*failed",
        message="COLMAP reconstruction appears to have failed.",
        recommendation=(
            "Recapture with slower motion, higher overlap, less blur, and more parallax; "
            "then rerun ns-process-data."
        ),
    ),
    LogPattern(
        category="missing_ffmpeg",
        severity="blocker",
        pattern=r"ffmpeg.*not found|required executable 'ffmpeg'|no such file.*ffmpeg",
        message="FFmpeg was not available for video extraction.",
        recommendation="Install FFmpeg or provide an image directory instead of a video.",
        command="conda install -c conda-forge ffmpeg",
    ),
    LogPattern(
        category="ffmpeg_decode_failed",
        severity="blocker",
        pattern=r"ffmpeg.*error|invalid data found|moov atom not found|could not open.*video",
        message="Video decoding failed during data preparation.",
        recommendation="Verify the video file, transcode it to mp4 if needed, or capture a new video.",
    ),
    LogPattern(
        category="python_import_error",
        severity="blocker",
        pattern=r"modulenotfounderror|no module named|importerror|cannot import name",
        message="A Python import error was detected.",
        recommendation="Install the missing package in the active environment and rerun the failing command.",
    ),
    LogPattern(
        category="transforms_missing",
        severity="blocker",
        pattern=r"transforms\.json.*not found|expected .*transforms\.json",
        message="Processed data did not produce transforms.json.",
        recommendation="Inspect COLMAP/FFmpeg logs and recapture or rerun ns-process-data.",
        command="python scripts/prepare_data.py --input path/to/video.mp4 --type video --output data/processed/<scene>",
    ),
    LogPattern(
        category="input_path_missing",
        severity="blocker",
        pattern=r"input path does not exist|filenotfounderror",
        message="An input path referenced by the run was missing.",
        recommendation="Check the run command paths and rerun from the repository root with existing inputs.",
    ),
    LogPattern(
        category="permission_error",
        severity="blocker",
        pattern=r"permission denied|access is denied",
        message="A filesystem permission error was detected.",
        recommendation="Move the project and data to a writable workspace or fix file permissions.",
    ),
    LogPattern(
        category="lerf_render_fallback",
        severity="warning",
        pattern=r"automated lerf rendering failed|does not expose image_encoder",
        message="Automated LERF rendering fell back to the interactive viewer workflow.",
        recommendation=(
            "Use scripts/repair_scene_query_from_viewer.py for full scene reports or "
            "scripts/import_viewer_outputs.py for single-query viewer renders."
        ),
        command="python scripts/repair_scene_query_from_viewer.py --report results/pipeline_runs/desk_scene/queries/mug/scene_query_report.json --viewer-root results/manual_viewer",
    ),
]


def build_failure_diagnostics(run_dir: str | Path) -> FailureDiagnosticsReport:
    """Inspect a run directory and classify known failure modes."""

    root = Path(run_dir)
    pipeline = _read_json(root / "pipeline_summary.json")
    scene_name = str(pipeline.get("scene_name") or root.name)
    diagnostics: list[FailureDiagnostic] = []

    command_logs = _command_logs(root)
    training_summaries = _training_summaries(root)
    query_reports = _query_reports(root)
    viewer_repair_summaries = _viewer_repair_summaries(root)

    _diagnose_pipeline_steps(pipeline, diagnostics)
    _diagnose_environment(root, diagnostics)
    _diagnose_preflight(root, diagnostics)
    _diagnose_command_logs(command_logs, diagnostics, root)
    _diagnose_training_summaries(training_summaries, diagnostics, root)
    _diagnose_query_reports(query_reports, diagnostics, root)
    _diagnose_viewer_repair_summaries(viewer_repair_summaries, diagnostics, root)

    diagnostics = _dedupe(diagnostics)
    status = _status(diagnostics)
    failed_commands = sum(1 for _, payload in command_logs if _safe_int(payload.get("returncode")) != 0)
    return FailureDiagnosticsReport(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        status=status,
        command_log_count=len(command_logs),
        failed_command_count=failed_commands,
        training_summary_count=len(training_summaries),
        query_report_count=len(query_reports),
        diagnostics=diagnostics,
    )


def write_failure_diagnostics(
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> FailureDiagnosticsReport:
    """Build and write JSON plus Markdown failure diagnostics."""

    root = Path(run_dir)
    report = build_failure_diagnostics(root)
    report.to_json(output or root / "failure_diagnostics.json")
    report.to_markdown(markdown_output or root / "failure_diagnostics.md")
    return report


def _diagnose_pipeline_steps(
    pipeline: dict[str, Any],
    diagnostics: list[FailureDiagnostic],
) -> None:
    if not pipeline:
        diagnostics.append(
            FailureDiagnostic(
                severity="blocker",
                category="missing_pipeline_summary",
                message="pipeline_summary.json is missing or unreadable.",
                recommendation="Run scripts/run_scene_pipeline.py before diagnosing a run.",
                artifact="pipeline_summary.json",
            )
        )
        return
    for step in pipeline.get("steps") or []:
        if not isinstance(step, dict) or step.get("status") != "failed":
            continue
        name = str(step.get("name") or "unknown")
        diagnostics.append(
            FailureDiagnostic(
                severity="blocker",
                category="failed_pipeline_step",
                message=f"Pipeline step failed: {name}.",
                recommendation=str(step.get("error") or "Inspect the step command log and rerun."),
                artifact="pipeline_summary.json",
                source=name,
            )
        )


def _diagnose_environment(root: Path, diagnostics: list[FailureDiagnostic]) -> None:
    environment = _read_json(root / "environment_report.json")
    failures = [str(item) for item in environment.get("strict_failures") or []]
    for failure in failures:
        diagnostics.append(
            FailureDiagnostic(
                severity="blocker",
                category="environment_strict_failure",
                message=f"Required environment check failed: {failure}.",
                recommendation="Run the environment checker with upstream and GPU checks, then follow the reported hints.",
                artifact="environment_report.json",
                command="python scripts/check_env.py --check-upstream --require-gpu --verbose",
                source=failure,
            )
        )


def _diagnose_preflight(root: Path, diagnostics: list[FailureDiagnostic]) -> None:
    preflight = _read_json(root / "preflight_report.json")
    status = str(preflight.get("status") or "")
    if status != "blocked":
        return
    failed = [
        str(check.get("name") or check.get("category") or "unknown")
        for check in preflight.get("checks") or []
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    diagnostics.append(
        FailureDiagnostic(
            severity="blocker",
            category="preflight_blocked",
            message="Real-run preflight reported blocker-level checks.",
            recommendation="Fix failed preflight checks before launching training.",
            artifact="preflight_report.md",
            command="python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --require-gpu",
            source=", ".join(failed[:6]),
        )
    )


def _diagnose_command_logs(
    command_logs: list[tuple[Path, dict[str, Any]]],
    diagnostics: list[FailureDiagnostic],
    root: Path,
) -> None:
    for path, payload in command_logs:
        artifact = _relative_path(path, root)
        text = "\n".join(
            [
                str(payload.get("stdout") or ""),
                str(payload.get("stderr") or ""),
                str(payload.get("error") or ""),
            ]
        )
        lowered = text.lower()
        for pattern in LOG_PATTERNS:
            if re.search(pattern.pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
                diagnostics.append(
                    FailureDiagnostic(
                        severity=pattern.severity,
                        category=pattern.category,
                        message=pattern.message,
                        recommendation=pattern.recommendation,
                        artifact=artifact,
                        command=pattern.command,
                        source=path.name,
                    )
                )
        if _safe_int(payload.get("returncode")) != 0:
            diagnostics.append(
                FailureDiagnostic(
                    severity="blocker",
                    category="command_failed",
                    message=f"Command log reports non-zero exit code {_safe_int(payload.get('returncode'))}.",
                    recommendation="Open the command log, fix the failing dependency or input, and rerun the step.",
                    artifact=artifact,
                    source=path.name,
                )
            )


def _diagnose_training_summaries(
    summaries: list[tuple[Path, dict[str, Any]]],
    diagnostics: list[FailureDiagnostic],
    root: Path,
) -> None:
    for path, payload in summaries:
        artifact = _relative_path(path, root)
        dry_run = bool(payload.get("dry_run"))
        config_path = str(payload.get("config_path") or "").strip()
        if payload.get("success") is False:
            diagnostics.append(
                FailureDiagnostic(
                    severity="blocker",
                    category="training_summary_failed",
                    message=f"Training summary reports success=false for {path.name}.",
                    recommendation="Inspect the matching command log and rerun training after fixing the root cause.",
                    artifact=artifact,
                    source=path.name,
                )
            )
        if not dry_run and not config_path:
            diagnostics.append(
                FailureDiagnostic(
                    severity="blocker",
                    category="missing_trained_config",
                    message=f"No trained config_path was recorded for {path.name}.",
                    recommendation="Confirm ns-train completed and that config.yml is discoverable under the output directory.",
                    artifact=artifact,
                    source=path.name,
                )
            )
        if config_path and not _path_exists_from_run(config_path, root):
            diagnostics.append(
                FailureDiagnostic(
                    severity="warning",
                    category="recorded_config_not_found",
                    message=f"Recorded config_path for {path.name} was not found on this machine.",
                    recommendation="If the run was moved from another machine, copy the Nerfstudio output directory or update the config path.",
                    artifact=artifact,
                    source=path.name,
                )
            )


def _diagnose_query_reports(
    reports: list[tuple[Path, dict[str, Any]]],
    diagnostics: list[FailureDiagnostic],
    root: Path,
) -> None:
    fallback_count = 0
    warning_count = 0
    for path, payload in reports:
        artifact = _relative_path(path, root)
        rendered = payload.get("rendered_images") if isinstance(payload.get("rendered_images"), list) else []
        if any(isinstance(item, dict) and item.get("kind") == "viewer_fallback" for item in rendered):
            fallback_count += 1
        warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
        if warnings:
            warning_count += len(warnings)
        for warning in warnings:
            warning_text = str(warning).lower()
            if "automated lerf rendering failed" in warning_text or "viewer fallback" in warning_text:
                fallback_count += 1
                diagnostics.append(
                    FailureDiagnostic(
                        severity="warning",
                        category="lerf_render_fallback",
                        message="A query result used the interactive LERF viewer fallback.",
                        recommendation=(
                            "Repair the scene query report with manually saved viewer outputs, or import "
                            "single-query outputs before annotation/evaluation."
                        ),
                        artifact=artifact,
                        command=(
                            "python scripts/repair_scene_query_from_viewer.py "
                            "--report results/pipeline_runs/desk_scene/queries/mug/scene_query_report.json "
                            "--viewer-root results/manual_viewer"
                        ),
                        source=path.name,
                    )
                )
    if fallback_count and not any(item.category == "lerf_render_fallback" for item in diagnostics):
        diagnostics.append(
            FailureDiagnostic(
                severity="warning",
                category="lerf_render_fallback",
                message=f"{fallback_count} query artifacts used LERF viewer fallback outputs.",
                recommendation="Use repaired/imported viewer outputs or update the automated renderer for the installed upstream revision.",
                artifact="queries/",
            )
        )
    if warning_count and not fallback_count:
        diagnostics.append(
            FailureDiagnostic(
                severity="warning",
                category="query_warnings",
                message=f"Query reports include {warning_count} warning messages.",
                recommendation="Inspect scene_query_report.md and query_result.json before reporting semantic query quality.",
                artifact="queries/",
            )
        )


def _diagnose_viewer_repair_summaries(
    summaries: list[tuple[Path, dict[str, Any]]],
    diagnostics: list[FailureDiagnostic],
    root: Path,
) -> None:
    for path, payload in summaries:
        artifact = _relative_path(path, root)
        missing_required = [str(item) for item in payload.get("missing_required_queries") or []]
        if payload.get("ok") is False or missing_required:
            diagnostics.append(
                FailureDiagnostic(
                    severity="blocker",
                    category="viewer_repair_incomplete",
                    message="A scene-query viewer repair did not cover all required queries.",
                    recommendation="Add missing manual viewer output directories or rerun repair without --require-all.",
                    artifact=artifact,
                    command=(
                        "python scripts/repair_scene_query_from_viewer.py "
                        "--report results/pipeline_runs/desk_scene/queries/mug/scene_query_report.json "
                        "--viewer-root results/manual_viewer --require-all"
                    ),
                    source=path.name,
                )
            )
            continue
        missing_dirs = [str(item) for item in payload.get("missing_viewer_dirs") or []]
        if missing_dirs:
            diagnostics.append(
                FailureDiagnostic(
                    severity="warning",
                    category="viewer_repair_partial",
                    message=f"A scene-query viewer repair kept {len(missing_dirs)} query result(s) unchanged.",
                    recommendation="Review viewer_repair_summary.json before using the repaired report as evidence.",
                    artifact=artifact,
                    source=path.name,
                )
            )


def _command_logs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, _read_json(path)) for path in sorted((root / "logs").glob("*.json"))]


def _training_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, _read_json(path))
        for path in sorted((root / "training").glob("*_train_summary.json"))
    ]


def _query_reports(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, _read_json(path)) for path in sorted((root / "queries").rglob("query_result.json"))]


def _viewer_repair_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, _read_json(path))
        for path in sorted((root / "queries").rglob("viewer_repair_summary.json"))
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _path_exists_from_run(raw_path: str, root: Path) -> bool:
    path = Path(raw_path)
    if path.exists():
        return True
    if not path.is_absolute() and (root / path).exists():
        return True
    return False


def _status(diagnostics: list[FailureDiagnostic]) -> DiagnosticStatus:
    if any(item.severity == "blocker" for item in diagnostics):
        return "blocked"
    if any(item.severity == "warning" for item in diagnostics):
        return "needs_attention"
    return "clear"


def _dedupe(items: list[FailureDiagnostic]) -> list[FailureDiagnostic]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[FailureDiagnostic] = []
    for item in sorted(items, key=_sort_key):
        key = (item.category, item.artifact, item.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _sort_key(item: FailureDiagnostic) -> tuple[int, str, str]:
    order = {"blocker": 0, "warning": 1, "info": 2}
    return order[item.severity], item.category, item.artifact


def _diagnostic_lines(items: list[FailureDiagnostic]) -> list[str]:
    if not items:
        return ["- None. No known failure signatures were detected."]
    lines: list[str] = []
    for item in items:
        lines.append(f"- [{item.severity}] {item.category}: {item.message}")
        lines.append(f"  Recommendation: {item.recommendation}")
        if item.command:
            lines.append(f"  Command: `{item.command}`")
        if item.artifact:
            lines.append(f"  Artifact: `{item.artifact}`")
    return lines


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name
