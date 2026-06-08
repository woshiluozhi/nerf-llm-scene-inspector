"""Real-run preflight checks for captured scenes and upstream tool readiness."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.scene_validation import SceneDataInspection, inspect_processed_scene
from nerf_llm_scene_inspector.utils.env_check import (
    CheckItem,
    EnvReport,
    check_command,
    check_cuda,
    check_import,
    check_ns_train_methods,
)
from nerf_llm_scene_inspector.utils.paths import utc_timestamp


PreflightStatus = Literal["pass", "warn", "fail"]
PreflightLevel = Literal["ready", "needs_attention", "blocked"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
LERF_VARIANTS = {"lerf", "lerf-lite", "lerf-big"}


@dataclass
class PreflightCheck:
    """One preflight finding that can be shown in reports or CI output."""

    name: str
    status: PreflightStatus
    category: str
    detail: str = ""
    recommendation: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class PreflightReport:
    """Portable report that summarizes whether a real scene run is ready."""

    status: PreflightLevel
    ready_for_real_run: bool
    scene_name: str
    backend: str
    variant: str
    input_type: str
    input_path: str = ""
    data_path: str = ""
    config_path: str = ""
    dry_run: bool = False
    environment: dict[str, Any] = field(default_factory=dict)
    scene_inspection: dict[str, Any] | None = None
    checks: list[PreflightCheck] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    @property
    def fail_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready_for_real_run": self.ready_for_real_run,
            "scene_name": self.scene_name,
            "backend": self.backend,
            "variant": self.variant,
            "input_type": self.input_type,
            "input_path": self.input_path,
            "data_path": self.data_path,
            "config_path": self.config_path,
            "dry_run": self.dry_run,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "environment": self.environment,
            "scene_inspection": self.scene_inspection,
            "checks": [check.to_dict() for check in self.checks],
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
            "# Real-Run Preflight Report",
            "",
            f"- Status: {self.status}",
            f"- Ready for real run: {self.ready_for_real_run}",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Backend: {self.backend}",
            f"- Variant: {self.variant}",
            f"- Input type: {self.input_type}",
            f"- Input path: {self.input_path or 'not provided'}",
            f"- Processed data: {self.data_path or 'not provided'}",
            f"- Config path: {self.config_path or 'not provided'}",
            f"- Dry run mode: {self.dry_run}",
            f"- Failed checks: {self.fail_count}",
            f"- Warning checks: {self.warn_count}",
            "",
            "## Checks",
            "",
            *_check_lines(self.checks),
        ]
        if self.scene_inspection:
            lines.extend(
                [
                    "",
                    "## Scene Inspection Summary",
                    "",
                    f"- Ready for training: {self.scene_inspection.get('ready_for_training')}",
                    f"- Quality score: {self.scene_inspection.get('quality_score')}",
                    f"- Frame count: {self.scene_inspection.get('frame_count')}",
                    f"- Missing images: {self.scene_inspection.get('missing_image_count')}",
                    f"- Pose coverage: {self.scene_inspection.get('pose_coverage_score')}",
                ]
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_real_run_preflight(
    *,
    input_path: str | Path | None = None,
    input_type: str = "video",
    data_path: str | Path | None = None,
    config_path: str | Path | None = None,
    scene_name: str = "",
    backend: str = "lerf",
    variant: str = "lerf-lite",
    min_frames: int = 50,
    max_missing_image_ratio: float = 0.0,
    min_pose_extent: float = 0.05,
    require_gpu: bool = False,
    check_upstream: bool = True,
    dry_run: bool = False,
) -> PreflightReport:
    """Build a CPU-safe readiness report before running expensive training."""

    normalized_type = input_type.lower()
    checks: list[PreflightCheck] = []
    checks.extend(_backend_checks(backend=backend, variant=variant))
    checks.extend(_raw_input_checks(input_path, normalized_type, min_frames=min_frames, dry_run=dry_run))
    env_report = _build_targeted_env_report(
        backend=backend,
        variant=variant,
        input_type=normalized_type,
        require_gpu=require_gpu,
        check_upstream=check_upstream,
    )
    checks.extend(_env_checks(env_report))

    scene_inspection: SceneDataInspection | None = None
    if data_path:
        scene_inspection = inspect_processed_scene(
            data_path,
            min_frames=1 if dry_run else min_frames,
            max_missing_image_ratio=max_missing_image_ratio,
            min_pose_extent=min_pose_extent,
        )
        checks.append(_scene_check(scene_inspection))
    else:
        checks.append(
            PreflightCheck(
                name="processed_scene",
                status="warn",
                category="scene_data",
                detail="No processed scene directory was provided.",
                recommendation=(
                    "Run scripts/prepare_data.py first, or pass --data to preflight an existing "
                    "Nerfstudio scene."
                ),
            )
        )

    checks.append(_config_check(config_path))
    level = _overall_level(checks)
    return PreflightReport(
        status=level,
        ready_for_real_run=level == "ready",
        scene_name=scene_name,
        backend=backend,
        variant=variant,
        input_type=normalized_type,
        input_path=str(input_path) if input_path else "",
        data_path=str(data_path) if data_path else "",
        config_path=str(config_path) if config_path else "",
        dry_run=dry_run,
        environment=env_report.to_dict(),
        scene_inspection=scene_inspection.to_dict() if scene_inspection else None,
        checks=checks,
    )


def _build_targeted_env_report(
    *,
    backend: str,
    variant: str,
    input_type: str,
    require_gpu: bool,
    check_upstream: bool,
) -> EnvReport:
    checks: list[CheckItem] = [
        CheckItem(
            name="python>=3.10",
            ok=sys.version_info >= (3, 10),
            category="runtime",
            detail=sys.version.split()[0],
            required=True,
            hint="Use Python 3.10 or newer.",
        ),
        check_import("nerf_llm_scene_inspector", required=True),
        check_import("numpy", required=True),
        check_import("PIL", required=True),
        check_import("yaml", required=True),
    ]

    checks.append(check_cuda(require_gpu=require_gpu))

    if check_upstream:
        command_requirements = ["ns-process-data", "ns-train", "colmap"]
        if input_type == "video":
            command_requirements.append("ffmpeg")
        for command in command_requirements:
            checks.append(check_command(command, required=True))
        checks.append(check_command("ns-viewer", required=False))

        methods = ["nerfacto"]
        if backend == "lerf":
            methods.append(variant)
        elif backend == "opennerf":
            methods.append("opennerf")
        checks.extend(check_ns_train_methods(methods, required=True))

    strict_failures = [check.name for check in checks if check.required and not check.ok]
    return EnvReport(
        ok=not strict_failures,
        python_version=sys.version.split()[0],
        platform=sys.platform,
        checks=checks,
        strict_failures=strict_failures,
    )


def _backend_checks(*, backend: str, variant: str) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if backend not in {"lerf", "opennerf"}:
        checks.append(
            PreflightCheck(
                name="backend",
                status="fail",
                category="configuration",
                detail=f"Unsupported backend: {backend}",
                recommendation="Use --backend lerf or --backend opennerf.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="backend",
                status="pass",
                category="configuration",
                detail=backend,
            )
        )
    if backend == "lerf" and variant not in LERF_VARIANTS:
        checks.append(
            PreflightCheck(
                name="variant",
                status="fail",
                category="configuration",
                detail=f"Unsupported LERF variant: {variant}",
                recommendation="Use lerf, lerf-lite, or lerf-big.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="variant",
                status="pass",
                category="configuration",
                detail=variant,
            )
        )
    return checks


def _raw_input_checks(
    input_path: str | Path | None,
    input_type: str,
    *,
    min_frames: int,
    dry_run: bool,
) -> list[PreflightCheck]:
    if not input_path:
        return [
            PreflightCheck(
                name="raw_input",
                status="warn",
                category="input",
                detail="No raw input path was provided.",
                recommendation="Pass --input path/to/video_or_images for capture-level checks.",
            )
        ]
    path = Path(input_path)
    if not path.exists():
        return [
            PreflightCheck(
                name="raw_input",
                status="warn" if dry_run else "fail",
                category="input",
                detail=f"Input path does not exist: {path}",
                recommendation="Check the path before running ns-process-data.",
                artifact=str(path),
            )
        ]
    if input_type == "video":
        return [_video_input_check(path, dry_run=dry_run)]
    if input_type == "images":
        return [_image_input_check(path, min_frames=min_frames, dry_run=dry_run)]
    return [
        PreflightCheck(
            name="input_type",
            status="fail",
            category="input",
            detail=f"Unsupported input type: {input_type}",
            recommendation="Use --type video or --type images.",
        )
    ]


def _video_input_check(path: Path, *, dry_run: bool) -> PreflightCheck:
    if not path.is_file():
        return PreflightCheck(
            name="video_input",
            status="warn" if dry_run else "fail",
            category="input",
            detail=f"Video input is not a file: {path}",
            recommendation="Pass a video file such as .mp4 or .mov.",
            artifact=str(path),
        )
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return PreflightCheck(
            name="video_input",
            status="warn",
            category="input",
            detail=f"Unexpected video suffix: {path.suffix or '<none>'}",
            recommendation="Common tested suffixes are .mp4, .mov, .m4v, .avi, and .mkv.",
            artifact=str(path),
        )
    if path.stat().st_size <= 0:
        return PreflightCheck(
            name="video_input",
            status="warn" if dry_run else "fail",
            category="input",
            detail="Video file is empty.",
            recommendation="Use a non-empty phone capture with slow motion and high overlap.",
            artifact=str(path),
        )
    return PreflightCheck(
        name="video_input",
        status="pass",
        category="input",
        detail=f"{path.name}, {path.stat().st_size} bytes",
        artifact=str(path),
    )


def _image_input_check(path: Path, *, min_frames: int, dry_run: bool) -> PreflightCheck:
    if not path.is_dir():
        return PreflightCheck(
            name="image_input",
            status="warn" if dry_run else "fail",
            category="input",
            detail=f"Image input is not a directory: {path}",
            recommendation="Pass a directory containing overlapping scene images.",
            artifact=str(path),
        )
    images = [item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES]
    if len(images) < min_frames:
        return PreflightCheck(
            name="image_input",
            status="warn" if dry_run else "fail",
            category="input",
            detail=f"Found {len(images)} image files; recommended minimum is {min_frames}.",
            recommendation="Capture more overlapping views before running ns-process-data.",
            artifact=str(path),
        )
    return PreflightCheck(
        name="image_input",
        status="pass",
        category="input",
        detail=f"Found {len(images)} image files.",
        artifact=str(path),
    )


def _env_checks(report: EnvReport) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for item in report.checks:
        if item.ok:
            status: PreflightStatus = "pass"
        elif item.required:
            status = "fail"
        else:
            status = "warn"
        checks.append(
            PreflightCheck(
                name=item.name,
                status=status,
                category=f"environment:{item.category}",
                detail=item.detail,
                recommendation=item.hint,
            )
        )
    return checks


def _scene_check(inspection: SceneDataInspection) -> PreflightCheck:
    if inspection.ready_for_training:
        return PreflightCheck(
            name="processed_scene",
            status="pass",
            category="scene_data",
            detail=f"{inspection.frame_count} frames, quality_score={inspection.quality_score}",
            artifact=inspection.data_path,
        )
    return PreflightCheck(
        name="processed_scene",
        status="fail",
        category="scene_data",
        detail="Processed scene is not ready for training.",
        recommendation=" ".join(inspection.recommendations[:2]),
        artifact=inspection.data_path,
    )


def _config_check(config_path: str | Path | None) -> PreflightCheck:
    if not config_path:
        return PreflightCheck(
            name="model_config",
            status="pass",
            category="training",
            detail="No trained config path was provided.",
            recommendation="This is expected before language-field training; pass --config for query-only runs.",
        )
    path = Path(config_path)
    if not path.exists():
        return PreflightCheck(
            name="model_config",
            status="fail",
            category="training",
            detail=f"Config path does not exist: {path}",
            recommendation="Use the config.yml path reported by train_language_field.py.",
            artifact=str(path),
        )
    return PreflightCheck(
        name="model_config",
        status="pass",
        category="training",
        detail=str(path),
        artifact=str(path),
    )


def _overall_level(checks: list[PreflightCheck]) -> PreflightLevel:
    if any(check.status == "fail" for check in checks):
        return "blocked"
    if any(check.status == "warn" for check in checks):
        return "needs_attention"
    return "ready"


def _check_lines(checks: list[PreflightCheck]) -> list[str]:
    if not checks:
        return ["- None."]
    lines: list[str] = []
    for check in checks:
        lines.append(f"- [{check.status.upper()}] {check.category}/{check.name}: {check.detail}")
        if check.recommendation:
            lines.append(f"  Recommendation: {check.recommendation}")
    return lines
