import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from nerf_llm_scene_inspector.capture_manifest import (
    build_capture_manifest,
    validate_capture_manifest,
    write_capture_manifest_bundle,
)
from nerf_llm_scene_inspector.preflight import build_real_run_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_capture_manifest_ready_for_complete_image_capture(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_images(images, count=4)

    manifest = build_capture_manifest(
        input_path=images,
        input_type="images",
        scene_name="desk_scene",
        capture_device="iPhone",
        scene_type="desk",
        lighting="bright indoor diffuse",
        camera_motion="slow orbit",
        static_scene=True,
        high_overlap=True,
        privacy_reviewed=True,
    )
    validation = validate_capture_manifest(manifest, min_images=4, require_privacy_review=True)

    assert manifest.approximate_frame_count == 4
    assert validation.status == "ready"
    assert validation.ok is True


def test_capture_manifest_validation_warns_on_scaffold(tmp_path: Path) -> None:
    manifest = build_capture_manifest(
        input_path=tmp_path / "missing",
        input_type="images",
        scene_name="desk_scene",
    )
    validation = validate_capture_manifest(manifest, min_images=3)

    assert validation.status == "needs_review"
    assert validation.warn_count >= 1
    assert any(check.name == "privacy_review" for check in validation.checks)


def test_create_capture_manifest_cli_writes_bundle(tmp_path: Path) -> None:
    images = tmp_path / "images"
    output = tmp_path / "manifest"
    _write_images(images, count=3)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_capture_manifest.py"),
            "--input",
            str(images),
            "--type",
            "images",
            "--scene-name",
            "desk_scene",
            "--capture-device",
            "phone",
            "--lighting",
            "bright indoor",
            "--static-scene",
            "--high-overlap",
            "--privacy-reviewed",
            "--min-images",
            "3",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "capture_manifest.json").exists()
    assert (output / "capture_manifest.md").exists()
    payload = json.loads((output / "capture_manifest_validation.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ready"


def test_preflight_includes_capture_manifest_validation(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _write_images(images, count=3)
    manifest = build_capture_manifest(
        input_path=images,
        input_type="images",
        scene_name="desk_scene",
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
        scene_name="desk_scene",
        min_frames=3,
        check_upstream=False,
        dry_run=True,
    )

    assert report.capture_manifest_validation is not None
    assert any(check.name == "capture_manifest" and check.status == "pass" for check in report.checks)


def _write_images(root: Path, *, count: int) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (32, 32), (index * 20, 40, 60)).save(root / f"image_{index:03d}.png")
