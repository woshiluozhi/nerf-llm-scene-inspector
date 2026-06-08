"""Data processing pipeline wrappers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from nerf_llm_scene_inspector.utils.paths import utc_timestamp
from nerf_llm_scene_inspector.utils.shell import require_executable, run_command


NERFSTUDIO_INSTALL_HINT = """Install Nerfstudio:
python -m pip install nerfstudio
ns-install-cli
ns-process-data --help"""

FFMPEG_HINT = "Install FFmpeg, for example: conda install -c conda-forge ffmpeg"
COLMAP_HINT = "Install COLMAP, for example: conda install -c conda-forge colmap"


def prepare_data(
    input_path: str | Path,
    output_path: str | Path,
    data_type: str,
    *,
    dry_run: bool = False,
    command_log_path: str | Path | None = None,
) -> dict[str, object]:
    """Run Nerfstudio ns-process-data and write metadata."""

    if data_type not in {"video", "images"}:
        raise ValueError("--type must be either 'video' or 'images'")

    input_resolved = Path(input_path).expanduser()
    output_resolved = Path(output_path).expanduser()
    output_resolved.mkdir(parents=True, exist_ok=True)
    command = [
        "ns-process-data",
        data_type,
        "--data",
        str(input_resolved),
        "--output-dir",
        str(output_resolved),
    ]
    metadata: dict[str, object] = {
        "input_path": str(input_resolved),
        "output_path": str(output_resolved),
        "timestamp": utc_timestamp(),
        "command": command,
        "success": False,
        "dry_run": dry_run,
        "diagnostics": [],
    }

    try:
        if dry_run:
            _write_mock_transforms(output_resolved)
            result = run_command(command, dry_run=True, log_path=command_log_path)
        else:
            _validate_processing_prerequisites(input_resolved, data_type)
            result = run_command(command, check=False, log_path=command_log_path)
            if not result.ok:
                raise RuntimeError(
                    f"ns-process-data failed with exit code {result.returncode}\n"
                    f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
                )
        metadata["command_result"] = result.to_dict()
        transforms_path = output_resolved / "transforms.json"
        if not transforms_path.exists():
            raise RuntimeError(
                f"Expected {transforms_path} after processing, but it was not found. "
                "Check COLMAP reconstruction logs and input image quality."
            )
        metadata["success"] = True
        metadata["transforms_path"] = str(transforms_path)
    except Exception as exc:
        metadata["success"] = False
        metadata["error"] = str(exc)
        raise
    finally:
        metadata_path = output_resolved / "scene_inspector_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _validate_processing_prerequisites(input_path: Path, data_type: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    require_executable("ns-process-data", NERFSTUDIO_INSTALL_HINT)
    if data_type == "video" and shutil.which("ffmpeg") is None:
        raise RuntimeError(f"FFmpeg was not found on PATH.\n\n{FFMPEG_HINT}")
    if shutil.which("colmap") is None:
        raise RuntimeError(f"COLMAP was not found on PATH.\n\n{COLMAP_HINT}")


def _write_mock_transforms(output_path: Path) -> None:
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    _write_mock_images(image_dir, count=8)
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": 500.0,
        "fl_y": 500.0,
        "cx": 256.0,
        "cy": 256.0,
        "w": 512,
        "h": 512,
        "frames": [_mock_frame(index) for index in range(8)],
    }
    (output_path / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")


def _mock_frame(index: int) -> dict[str, object]:
    return {
        "file_path": f"images/frame_{index:05d}.png",
        "transform_matrix": [
            [1.0, 0.0, 0.0, round((index - 4) * 0.03, 4)],
            [0.0, 1.0, 0.0, round((index % 3 - 1) * 0.02, 4)],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _write_mock_images(image_dir: Path, *, count: int) -> None:
    from PIL import Image, ImageDraw

    for index in range(count):
        image = Image.new("RGB", (512, 512), (34 + index * 8, 48, 62))
        draw = ImageDraw.Draw(image)
        x0 = 96 + index * 18
        y0 = 160 + (index % 3) * 22
        draw.rectangle((x0, y0, x0 + 150, y0 + 120), fill=(180, 206, 220), outline=(245, 245, 245))
        draw.ellipse((280 - index * 8, 225, 370 - index * 8, 315), fill=(214, 102, 85))
        image.save(image_dir / f"frame_{index:05d}.png")
