import json
from pathlib import Path

from PIL import Image

from nerf_llm_scene_inspector.scene_validation import inspect_processed_scene


def test_inspect_processed_scene_ready(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    image_dir = scene / "images"
    image_dir.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (32, 32), (index * 30, 20, 40)).save(
            image_dir / f"frame_{index:05d}.png"
        )
    _write_transforms(scene, frame_count=3)

    report = inspect_processed_scene(scene, min_frames=3)

    assert report.ready_for_training is True
    assert report.frame_count == 3
    assert report.image_count == 3
    assert report.invalid_transform_count == 0
    assert report.quality_score == 1.0
    assert report.pose_coverage_score == 1.0
    assert report.camera_position_extent[0] > 0.05


def test_inspect_processed_scene_reports_missing_images_and_bad_pose(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    _write_transforms(scene, frame_count=2, invalid_last=True)

    report = inspect_processed_scene(scene, min_frames=2)

    assert report.ready_for_training is False
    assert report.missing_image_count == 2
    assert report.invalid_transform_count == 1
    assert any("missing" in warning.lower() for warning in report.warnings)


def test_inspect_processed_scene_flags_static_camera_poses(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    image_dir = scene / "images"
    image_dir.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (32, 32), (index * 30, 20, 40)).save(
            image_dir / f"frame_{index:05d}.png"
        )
    _write_transforms(scene, frame_count=3, static_pose=True)

    report = inspect_processed_scene(scene, min_frames=3)

    assert report.ready_for_training is False
    assert report.pose_coverage_score == 0.0
    assert report.duplicate_pose_count == 2
    assert any("translation extent" in warning for warning in report.warnings)


def _write_transforms(
    scene: Path,
    *,
    frame_count: int,
    invalid_last: bool = False,
    static_pose: bool = False,
) -> None:
    frames = []
    for index in range(frame_count):
        x_position = 0.0 if static_pose else round(index * 0.08, 4)
        matrix = [
            [1.0, 0.0, 0.0, x_position],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append(
            {
                "file_path": f"images/frame_{index:05d}.png",
                "transform_matrix": [[1.0, 0.0]] if invalid_last and index == frame_count - 1 else matrix,
            }
        )
    (scene / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "OPENCV",
                "w": 32,
                "h": 32,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
