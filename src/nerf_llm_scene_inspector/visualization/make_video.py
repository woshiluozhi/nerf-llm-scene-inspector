"""Create GIF or MP4 montages from query artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_gif(image_paths: list[str | Path], output_path: str | Path, duration_ms: int = 900) -> Path:
    """Create an animated GIF from image paths."""

    if not image_paths:
        raise ValueError("No images provided for GIF generation.")
    frames = [Image.open(path).convert("RGB") for path in image_paths]
    first, rest = frames[0], frames[1:]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first.save(out, save_all=True, append_images=rest, duration=duration_ms, loop=0)
    return out


def make_mp4_or_gif(image_paths: list[str | Path], output_path: str | Path, fps: int = 2) -> Path:
    """Create MP4 with imageio when available, otherwise create a GIF."""

    out = Path(output_path)
    if out.suffix.lower() == ".gif":
        return make_gif(image_paths, out, duration_ms=max(100, int(1000 / fps)))
    try:
        import imageio.v2 as imageio  # type: ignore
    except ModuleNotFoundError:
        return make_gif(image_paths, out.with_suffix(".gif"), duration_ms=max(100, int(1000 / fps)))
    frames = [imageio.imread(path) for path in image_paths]
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    return out
