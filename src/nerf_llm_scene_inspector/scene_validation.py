"""Processed Nerfstudio scene validation utilities."""

from __future__ import annotations

import json
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
        if _is_valid_transform(frame.get("transform_matrix")):
            valid_transforms += 1
        else:
            invalid_transforms += 1

    frame_count = len(frames)
    missing_ratio = missing_images and len(missing_images) / max(frame_count, 1) or 0.0
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

    ready = (
        frame_count >= min_frames
        and missing_ratio <= max_missing_image_ratio
        and invalid_transforms == 0
        and frame_count > 0
    )
    score = _quality_score(
        frame_count=frame_count,
        min_frames=min_frames,
        missing_ratio=missing_ratio,
        valid_transforms=valid_transforms,
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
            if not isinstance(item, (int, float)):
                return False
    return True


def _quality_score(
    *,
    frame_count: int,
    min_frames: int,
    missing_ratio: float,
    valid_transforms: int,
) -> float:
    if frame_count <= 0:
        return 0.0
    frame_score = min(frame_count / max(min_frames, 1), 1.0)
    image_score = max(0.0, 1.0 - missing_ratio)
    pose_score = valid_transforms / frame_count
    return round(0.4 * frame_score + 0.3 * image_score + 0.3 * pose_score, 4)


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
