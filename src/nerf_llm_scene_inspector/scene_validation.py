"""Processed Nerfstudio scene validation utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


@dataclass
class SceneDataInspection:
    """Portable quality report for a processed Nerfstudio scene directory."""

    data_path: str
    transforms_path: str
    success: bool
    ready_for_training: bool
    frame_count: int = 0
    image_count: int = 0
    missing_image_count: int = 0
    valid_transform_count: int = 0
    invalid_transform_count: int = 0
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None
    quality_score: float = 0.0
    pose_coverage_score: float = 0.0
    camera_position_extent: list[float] = field(default_factory=list)
    camera_path_length: float = 0.0
    median_camera_step: float = 0.0
    duplicate_pose_count: int = 0
    missing_images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status = "ready" if self.ready_for_training else "needs attention"
        lines = [
            "# Scene Data Inspection",
            "",
            f"- Status: {status}",
            f"- Quality score: {self.quality_score:.2f}",
            f"- Frames: {self.frame_count}",
            f"- Existing images: {self.image_count}",
            f"- Missing images: {self.missing_image_count}",
            f"- Valid poses: {self.valid_transform_count}",
            f"- Invalid poses: {self.invalid_transform_count}",
            f"- Camera model: {self.camera_model or 'unknown'}",
            f"- Resolution: {self.width or 'unknown'} x {self.height or 'unknown'}",
            f"- Pose coverage score: {self.pose_coverage_score:.2f}",
            f"- Camera position extent: {_format_vector(self.camera_position_extent)}",
            f"- Camera path length: {self.camera_path_length:.4f}",
            f"- Median camera step: {self.median_camera_step:.4f}",
            f"- Duplicate adjacent poses: {self.duplicate_pose_count}",
            "",
            "## Warnings",
            "",
            *_markdown_list(self.warnings),
            "",
            "## Recommendations",
            "",
            *_markdown_list(self.recommendations),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def inspect_processed_scene(
    data_path: str | Path,
    *,
    min_frames: int = 20,
    max_missing_image_ratio: float = 0.0,
    min_pose_extent: float = 0.05,
) -> SceneDataInspection:
    """Inspect a processed Nerfstudio scene directory without requiring GPU."""

    scene_dir = Path(data_path)
    transforms_path = scene_dir / "transforms.json"
    warnings: list[str] = []
    recommendations: list[str] = []
    if not transforms_path.exists():
        return SceneDataInspection(
            data_path=str(scene_dir),
            transforms_path=str(transforms_path),
            success=False,
            ready_for_training=False,
            warnings=[f"Missing transforms.json at {transforms_path}."],
            recommendations=["Run scripts/prepare_data.py before training."],
        )

    try:
        raw = json.loads(transforms_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return SceneDataInspection(
            data_path=str(scene_dir),
            transforms_path=str(transforms_path),
            success=False,
            ready_for_training=False,
            warnings=[f"Could not parse transforms.json: {exc}"],
            recommendations=["Rerun ns-process-data and inspect COLMAP logs."],
        )

    frames = raw.get("frames") or []
    if not isinstance(frames, list):
        frames = []
        warnings.append("transforms.json field 'frames' is not a list.")

    missing_images: list[str] = []
    existing_images = 0
    valid_transforms = 0
    invalid_transforms = 0
    camera_centers: list[list[float]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            warnings.append(f"Frame {index} is not a mapping.")
            invalid_transforms += 1
            continue
        image_path = _resolve_frame_path(scene_dir, frame.get("file_path"))
        if image_path is None or not image_path.exists():
            missing_images.append(str(frame.get("file_path", f"<frame {index}>")))
        else:
            existing_images += 1
        transform = frame.get("transform_matrix")
        if _is_valid_transform(transform):
            valid_transforms += 1
            camera_centers.append(_camera_center(transform))
        else:
            invalid_transforms += 1

    frame_count = len(frames)
    missing_ratio = missing_images and len(missing_images) / max(frame_count, 1) or 0.0
    pose_stats = _pose_diagnostics(camera_centers, min_pose_extent=min_pose_extent)
    if frame_count == 0:
        warnings.append("No frames were found in transforms.json.")
        recommendations.append("Capture or process a scene with overlapping images.")
    if frame_count and frame_count < min_frames:
        warnings.append(f"Only {frame_count} frames found; recommended minimum is {min_frames}.")
        recommendations.append("Capture a longer sequence with slow motion and high overlap.")
    if missing_images:
        warnings.append(f"{len(missing_images)} frame images referenced by transforms.json are missing.")
        recommendations.append("Check image paths under the processed scene directory.")
    if invalid_transforms:
        warnings.append(f"{invalid_transforms} frames have invalid 4x4 camera transforms.")
        recommendations.append("Rerun COLMAP/Nerfstudio processing and inspect pose estimation logs.")
    if valid_transforms and pose_stats["pose_coverage_score"] < 1.0:
        warnings.append(
            "Camera translation extent is "
            f"{pose_stats['max_extent']:.4f}; recommended minimum is {min_pose_extent:.4f}."
        )
        recommendations.append(
            "Capture from multiple viewpoints around the scene instead of rotating in place."
        )
    duplicate_pose_limit = max(1, int(0.2 * max(valid_transforms - 1, 1)))
    if pose_stats["duplicate_pose_count"] >= duplicate_pose_limit:
        warnings.append(
            f"{pose_stats['duplicate_pose_count']} adjacent camera poses are effectively duplicated."
        )
        recommendations.append("Use slower, smoother motion with enough parallax for COLMAP tracking.")

    ready = (
        frame_count >= min_frames
        and missing_ratio <= max_missing_image_ratio
        and invalid_transforms == 0
        and pose_stats["pose_coverage_score"] >= 1.0
        and frame_count > 0
    )
    score = _quality_score(
        frame_count=frame_count,
        min_frames=min_frames,
        missing_ratio=missing_ratio,
        valid_transforms=valid_transforms,
        pose_coverage_score=pose_stats["pose_coverage_score"],
    )
    return SceneDataInspection(
        data_path=str(scene_dir),
        transforms_path=str(transforms_path),
        success=True,
        ready_for_training=ready,
        frame_count=frame_count,
        image_count=existing_images,
        missing_image_count=len(missing_images),
        valid_transform_count=valid_transforms,
        invalid_transform_count=invalid_transforms,
        camera_model=_optional_str(raw.get("camera_model")),
        width=_optional_int(raw.get("w")),
        height=_optional_int(raw.get("h")),
        quality_score=score,
        pose_coverage_score=pose_stats["pose_coverage_score"],
        camera_position_extent=pose_stats["camera_position_extent"],
        camera_path_length=pose_stats["camera_path_length"],
        median_camera_step=pose_stats["median_camera_step"],
        duplicate_pose_count=pose_stats["duplicate_pose_count"],
        missing_images=missing_images[:25],
        warnings=warnings,
        recommendations=recommendations or ["Scene data passed basic structural checks."],
    )


def _resolve_frame_path(scene_dir: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    frame_path = Path(raw_path)
    if frame_path.is_absolute():
        return frame_path
    return scene_dir / frame_path


def _is_valid_transform(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return False
        for item in row:
            if not isinstance(item, (int, float)) or not math.isfinite(item):
                return False
    return True


def _camera_center(transform: object) -> list[float]:
    matrix = transform if isinstance(transform, list) else []
    return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]


def _pose_diagnostics(
    camera_centers: list[list[float]],
    *,
    min_pose_extent: float,
) -> dict[str, Any]:
    if not camera_centers:
        return {
            "camera_position_extent": [],
            "camera_path_length": 0.0,
            "median_camera_step": 0.0,
            "duplicate_pose_count": 0,
            "pose_coverage_score": 0.0,
            "max_extent": 0.0,
        }
    extents = [
        round(max(center[axis] for center in camera_centers) - min(center[axis] for center in camera_centers), 6)
        for axis in range(3)
    ]
    steps = [
        _distance(camera_centers[index - 1], camera_centers[index])
        for index in range(1, len(camera_centers))
    ]
    path_length = round(sum(steps), 6)
    median_step = round(_median(steps), 6) if steps else 0.0
    duplicate_pose_count = sum(1 for step in steps if step < 1e-6)
    max_extent = max(extents) if extents else 0.0
    pose_coverage_score = min(max_extent / max(min_pose_extent, 1e-9), 1.0)
    return {
        "camera_position_extent": extents,
        "camera_path_length": path_length,
        "median_camera_step": median_step,
        "duplicate_pose_count": duplicate_pose_count,
        "pose_coverage_score": round(pose_coverage_score, 4),
        "max_extent": round(max_extent, 6),
    }


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _quality_score(
    *,
    frame_count: int,
    min_frames: int,
    missing_ratio: float,
    valid_transforms: int,
    pose_coverage_score: float,
) -> float:
    if frame_count <= 0:
        return 0.0
    frame_score = min(frame_count / max(min_frames, 1), 1.0)
    image_score = max(0.0, 1.0 - missing_ratio)
    pose_score = valid_transforms / frame_count
    return round(0.3 * frame_score + 0.25 * image_score + 0.25 * pose_score + 0.2 * pose_coverage_score, 4)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _format_vector(values: list[float]) -> str:
    if not values:
        return "unknown"
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"
