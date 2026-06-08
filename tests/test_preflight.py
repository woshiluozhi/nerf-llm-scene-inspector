import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from nerf_llm_scene_inspector import preflight
from nerf_llm_scene_inspector.capture_manifest import build_capture_manifest, write_capture_manifest_bundle
from nerf_llm_scene_inspector.preflight import build_real_run_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_real_run_preflight_ready_with_processed_scene(tmp_path: Path) -> None:
    images = tmp_path / "raw_images"
    scene = tmp_path / "processed_scene"
    config = tmp_path / "config.yml"
    _write_images(images, count=3)
    _write_processed_scene(scene, frame_count=3)
    config.write_text("method_name: lerf-lite\n", encoding="utf-8")
    manifest = build_capture_manifest(
        input_path=images,
        input_type="images",
        scene_name="unit_scene",
        capture_device="phone",
        lighting="bright indoor",
        static_scene=True,
        high_overlap=True,
        privacy_reviewed=True,
    )
    manifest_json, *_ = write_capture_manifest_bundle(manifest, tmp_path / "manifest", min_images=3)

    report = build_real_run_preflight(
        input_path=images,
        input_type="images",
        capture_manifest_path=manifest_json,
        data_path=scene,
        config_path=config,
        scene_name="unit_scene",
        min_frames=3,
        check_upstream=False,
    )

    assert report.status == "ready"
    assert report.ready_for_real_run is True
    assert report.fail_count == 0
    assert report.scene_inspection is not None
    assert report.scene_inspection["ready_for_training"] is True
    assert any(check.name == "image_dimensions" and check.status == "pass" for check in report.checks)


def test_real_run_preflight_flags_bad_or_low_resolution_images(tmp_path: Path) -> None:
    images = tmp_path / "raw_images"
    _write_images(images, count=2, size=(320, 240))
    (images / "broken.png").write_bytes(b"not an image")

    report = build_real_run_preflight(
        input_path=images,
        input_type="images",
        scene_name="bad_images",
        min_frames=2,
        check_upstream=False,
    )

    assert report.status == "blocked"
    assert any(check.name == "image_decode" and check.status == "fail" for check in report.checks)
    assert any(check.name == "image_dimensions" and check.status == "warn" for check in report.checks)


def test_video_preflight_warns_when_ffprobe_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"not a real video, but non-empty")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None if name == "ffprobe" else "tool")

    report = build_real_run_preflight(
        input_path=video,
        input_type="video",
        scene_name="video_scene",
        check_upstream=False,
        dry_run=True,
    )

    assert report.status == "needs_attention"
    metadata = next(check for check in report.checks if check.name == "video_metadata")
    assert metadata.status == "warn"
    assert "ffprobe" in metadata.recommendation


def test_real_run_preflight_blocks_missing_real_input(tmp_path: Path) -> None:
    report = build_real_run_preflight(
        input_path=tmp_path / "missing.mp4",
        input_type="video",
        scene_name="missing_scene",
        check_upstream=False,
    )

    assert report.status == "blocked"
    assert any(check.name == "raw_input" and check.status == "fail" for check in report.checks)


def test_real_run_preflight_cli_writes_reports(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "preflight_real_run.py"),
            "--input",
            str(tmp_path / "missing.mp4"),
            "--no-check-upstream",
            "--dry-run",
            "--allow-warnings",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads((output / "preflight_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "needs_attention"
    assert (output / "preflight_report.md").exists()


def _write_images(root: Path, *, count: int, size: tuple[int, int] = (640, 480)) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", size, (index * 40, 20, 30)).save(root / f"image_{index:03d}.png")


def _write_processed_scene(root: Path, *, frame_count: int) -> None:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    for index in range(frame_count):
        Image.new("RGB", (32, 32), (index * 40, 20, 30)).save(
            image_dir / f"frame_{index:05d}.png"
        )
    frames = []
    for index in range(frame_count):
        x_position = round(index * 0.08, 4)
        frames.append(
            {
                "file_path": f"images/frame_{index:05d}.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, x_position],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (root / "transforms.json").write_text(
        json.dumps({"camera_model": "OPENCV", "w": 32, "h": 32, "frames": frames}),
        encoding="utf-8",
    )
