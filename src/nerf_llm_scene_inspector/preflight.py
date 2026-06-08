"""Real-run preflight checks for captured scenes and upstream tool readiness."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from nerf_llm_scene_inspector.capture_manifest import validate_capture_manifest
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
    capture_manifest_path: str = ""
    data_path: str = ""
    config_path: str = ""
    dry_run: bool = False
    environment: dict[str, Any] = field(default_factory=dict)
    capture_manifest_validation: dict[str, Any] | None = None
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
            "capture_manifest_path": self.capture_manifest_path,
            "data_path": self.data_path,
            "config_path": self.config_path,
            "dry_run": self.dry_run,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "environment": self.environment,
            "capture_manifest_validation": self.capture_manifest_validation,
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
            f"- Capture manifest: {self.capture_manifest_path or 'not provided'}",
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
        if self.capture_manifest_validation:
            lines.extend(
                [
                    "",
                    "## Capture Manifest Summary",
                    "",
                    f"- Status: {self.capture_manifest_validation.get('status')}",
                    f"- Failed checks: {self.capture_manifest_validation.get('fail_count')}",
                    f"- Warning checks: {self.capture_manifest_validation.get('warn_count')}",
                ]
            )
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
    capture_manifest_path: str | Path | None = None,
    data_path: str | Path | None = None,
    config_path: str | Path | None = None,
    scene_name: str = "",
    backend: str = "lerf",
    variant: str = "lerf-lite",
    min_frames: int = 50,
    min_image_width: int = 640,
    min_image_height: int = 480,
    min_video_seconds: float = 5.0,
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
    checks.extend(
        _raw_input_checks(
            input_path,
            normalized_type,
            min_frames=min_frames,
            min_image_width=min_image_width,
            min_image_height=min_image_height,
            min_video_seconds=min_video_seconds,
            dry_run=dry_run,
        )
    )
    capture_validation = None
    if capture_manifest_path:
        capture_validation = validate_capture_manifest(
            capture_manifest_path,
            min_images=min_frames,
            require_privacy_review=not dry_run,
        )
        checks.append(_capture_manifest_check(capture_validation))
    else:
        checks.append(
            PreflightCheck(
                name="capture_manifest",
                status="warn",
                category="capture",
                detail="No capture manifest was provided.",
                recommendation=(
                    "Run scripts/create_capture_manifest.py or use --capture-manifest so capture "
                    "conditions and privacy review are reproducible."
                ),
            )
        )
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
        capture_manifest_path=str(capture_manifest_path) if capture_manifest_path else "",
        data_path=str(data_path) if data_path else "",
        config_path=str(config_path) if config_path else "",
        dry_run=dry_run,
        environment=env_report.to_dict(),
        capture_manifest_validation=capture_validation.to_dict() if capture_validation else None,
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
        if input_type == "video":
            checks.append(check_command("ffprobe", required=False))
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
    min_image_width: int,
    min_image_height: int,
    min_video_seconds: float,
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
        return _video_input_checks(
            path,
            min_frames=min_frames,
            min_video_seconds=min_video_seconds,
            dry_run=dry_run,
        )
    if input_type == "images":
        return _image_input_checks(
            path,
            min_frames=min_frames,
            min_image_width=min_image_width,
            min_image_height=min_image_height,
            dry_run=dry_run,
        )
    return [
        PreflightCheck(
            name="input_type",
            status="fail",
            category="input",
            detail=f"Unsupported input type: {input_type}",
            recommendation="Use --type video or --type images.",
        )
    ]


def _video_input_checks(
    path: Path,
    *,
    min_frames: int,
    min_video_seconds: float,
    dry_run: bool,
) -> list[PreflightCheck]:
    if not path.is_file():
        return [
            PreflightCheck(
                name="video_input",
                status="warn" if dry_run else "fail",
                category="input",
                detail=f"Video input is not a file: {path}",
                recommendation="Pass a video file such as .mp4 or .mov.",
                artifact=str(path),
            )
        ]
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return [
            PreflightCheck(
                name="video_input",
                status="warn",
                category="input",
                detail=f"Unexpected video suffix: {path.suffix or '<none>'}",
                recommendation="Common tested suffixes are .mp4, .mov, .m4v, .avi, and .mkv.",
                artifact=str(path),
            )
        ]
    if path.stat().st_size <= 0:
        return [
            PreflightCheck(
                name="video_input",
                status="warn" if dry_run else "fail",
                category="input",
                detail="Video file is empty.",
                recommendation="Use a non-empty phone capture with slow motion and high overlap.",
                artifact=str(path),
            )
        ]
    return [
        PreflightCheck(
            name="video_input",
            status="pass",
            category="input",
            detail=f"{path.name}, {path.stat().st_size} bytes",
            artifact=str(path),
        ),
        _video_metadata_check(
            path,
            min_frames=min_frames,
            min_video_seconds=min_video_seconds,
            dry_run=dry_run,
        ),
    ]


def _image_input_checks(
    path: Path,
    *,
    min_frames: int,
    min_image_width: int,
    min_image_height: int,
    dry_run: bool,
) -> list[PreflightCheck]:
    if not path.is_dir():
        return [
            PreflightCheck(
                name="image_input",
                status="warn" if dry_run else "fail",
                category="input",
                detail=f"Image input is not a directory: {path}",
                recommendation="Pass a directory containing overlapping scene images.",
                artifact=str(path),
            )
        ]
    images = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    checks: list[PreflightCheck] = []
    if len(images) < min_frames:
        checks.append(
            PreflightCheck(
                name="image_count",
                status="warn" if dry_run else "fail",
                category="input",
                detail=f"Found {len(images)} image files; recommended minimum is {min_frames}.",
                recommendation="Capture more overlapping views before running ns-process-data.",
                artifact=str(path),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="image_count",
                status="pass",
                category="input",
                detail=f"Found {len(images)} image files.",
                artifact=str(path),
            )
        )
    if not images:
        return checks

    image_profile = _inspect_image_files(images)
    bad_images = image_profile["bad_images"]
    dimensions = image_profile["dimensions"]
    if bad_images:
        checks.append(
            PreflightCheck(
                name="image_decode",
                status="warn" if dry_run else "fail",
                category="input",
                detail=f"{len(bad_images)} image files could not be decoded.",
                recommendation="Remove corrupt files or re-export the capture before running COLMAP.",
                artifact=", ".join(bad_images[:3]),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="image_decode",
                status="pass",
                category="input",
                detail=f"Decoded {len(dimensions)} image files.",
                artifact=str(path),
            )
        )
    if dimensions:
        checks.append(
            _image_dimensions_check(
                dimensions,
                min_image_width=min_image_width,
                min_image_height=min_image_height,
            )
        )
    return checks


def _video_metadata_check(
    path: Path,
    *,
    min_frames: int,
    min_video_seconds: float,
    dry_run: bool,
) -> PreflightCheck:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return PreflightCheck(
            name="video_metadata",
            status="warn",
            category="input",
            detail="ffprobe is unavailable, so video duration, frame rate, and resolution were not checked.",
            recommendation="Install FFmpeg/ffprobe before relying on a phone video for a real run.",
            artifact=str(path),
        )
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PreflightCheck(
            name="video_metadata",
            status="warn",
            category="input",
            detail=f"ffprobe could not inspect the video: {exc}",
            recommendation="Verify the video opens locally and rerun preflight with FFmpeg installed.",
            artifact=str(path),
        )
    if proc.returncode != 0:
        return PreflightCheck(
            name="video_metadata",
            status="warn" if dry_run else "fail",
            category="input",
            detail=f"ffprobe failed: {(proc.stderr or proc.stdout).strip()[:200]}",
            recommendation="Re-export the video as a standard H.264/H.265 .mp4 or use an image sequence.",
            artifact=str(path),
        )

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return PreflightCheck(
            name="video_metadata",
            status="warn",
            category="input",
            detail=f"ffprobe returned invalid JSON: {exc}",
            recommendation="Rerun preflight after updating FFmpeg.",
            artifact=str(path),
        )
    streams = payload.get("streams") or []
    stream = streams[0] if streams and isinstance(streams[0], dict) else {}
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    width = _safe_int(stream.get("width"))
    height = _safe_int(stream.get("height"))
    duration = _safe_float(stream.get("duration")) or _safe_float(fmt.get("duration"))
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    frame_count = _safe_int(stream.get("nb_frames"))
    if frame_count is None and duration is not None and fps is not None:
        frame_count = int(duration * fps)

    problems: list[str] = []
    status: PreflightStatus = "pass"
    if width is None or height is None:
        problems.append("missing resolution metadata")
        status = "warn"
    if duration is None:
        problems.append("missing duration metadata")
        status = "warn"
    elif duration < min_video_seconds:
        problems.append(f"duration {duration:.1f}s is shorter than recommended {min_video_seconds:.1f}s")
        status = "warn"
    if frame_count is None:
        problems.append("missing frame-count metadata")
        status = "warn"
    elif frame_count < min_frames:
        problems.append(f"estimated frame count {frame_count} is below recommended {min_frames}")
        status = "warn" if dry_run else "fail"
    detail_parts = [
        f"resolution={width or '?'}x{height or '?'}",
        f"duration={duration:.2f}s" if duration is not None else "duration=?",
        f"fps={fps:.2f}" if fps is not None else "fps=?",
        f"frames={frame_count}" if frame_count is not None else "frames=?",
    ]
    if problems:
        detail_parts.append("issues=" + "; ".join(problems))
    return PreflightCheck(
        name="video_metadata",
        status=status,
        category="input",
        detail=", ".join(detail_parts),
        recommendation=(
            ""
            if status == "pass"
            else "Use a longer, sharp phone video with slow motion, high overlap, and enough parallax."
        ),
        artifact=str(path),
    )


def _inspect_image_files(images: list[Path]) -> dict[str, Any]:
    dimensions: list[tuple[int, int]] = []
    bad_images: list[str] = []
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                dimensions.append((int(image.width), int(image.height)))
                image.verify()
        except (OSError, UnidentifiedImageError):
            bad_images.append(str(image_path))
    return {"dimensions": dimensions, "bad_images": bad_images}


def _image_dimensions_check(
    dimensions: list[tuple[int, int]],
    *,
    min_image_width: int,
    min_image_height: int,
) -> PreflightCheck:
    widths = [width for width, _ in dimensions]
    heights = [height for _, height in dimensions]
    min_width = min(widths)
    min_height = min(heights)
    unique_dims = sorted(set(dimensions))
    problems: list[str] = []
    if min_width < min_image_width or min_height < min_image_height:
        problems.append(
            f"minimum resolution {min_width}x{min_height} is below recommended "
            f"{min_image_width}x{min_image_height}"
        )
    if len(unique_dims) > 1:
        problems.append(f"{len(unique_dims)} different image resolutions found")
    return PreflightCheck(
        name="image_dimensions",
        status="warn" if problems else "pass",
        category="input",
        detail=(
            f"min={min_width}x{min_height}, unique_resolutions={len(unique_dims)}, "
            f"checked={len(dimensions)}"
            + (", issues=" + "; ".join(problems) if problems else "")
        ),
        recommendation=(
            ""
            if not problems
            else "Use a consistent high-resolution image export, ideally original phone frames."
        ),
    )


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        num = _safe_float(numerator)
        den = _safe_float(denominator)
        if num is None or den in (None, 0):
            return None
        return num / den
    return _safe_float(text)


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


def _capture_manifest_check(validation: Any) -> PreflightCheck:
    status = str(validation.status)
    if status == "ready":
        check_status: PreflightStatus = "pass"
    elif status == "blocked":
        check_status = "fail"
    else:
        check_status = "warn"
    return PreflightCheck(
        name="capture_manifest",
        status=check_status,
        category="capture",
        detail=f"status={status}, warnings={validation.warn_count}, failures={validation.fail_count}",
        recommendation=(
            ""
            if check_status == "pass"
            else "Open capture_manifest_validation.md and fill missing capture/privacy fields."
        ),
        artifact=validation.manifest_path,
    )


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
