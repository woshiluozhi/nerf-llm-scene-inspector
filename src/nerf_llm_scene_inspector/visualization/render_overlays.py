"""Create side-by-side semantic query overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def heatmap_from_scores(scores: np.ndarray) -> Image.Image:
    """Convert a normalized score map to a simple RGB heatmap."""

    clipped = np.clip(scores, 0.0, 1.0)
    red = np.clip(4 * clipped - 1.5, 0, 1)
    green = np.clip(4 * np.minimum(clipped, 1 - clipped), 0, 1)
    blue = np.clip(1.5 - 4 * clipped, 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def blend_heatmap(rgb: Image.Image, heatmap: Image.Image, alpha: float = 0.45) -> Image.Image:
    """Blend an RGB render and heatmap."""

    base = rgb.convert("RGB")
    hm = heatmap.convert("RGB").resize(base.size)
    return Image.blend(base, hm, alpha=alpha)


def create_side_by_side_overlay(
    rgb_path: str | Path,
    heatmap_path: str | Path,
    output_path: str | Path,
    *,
    query: str,
    caption: str | None = None,
) -> Path:
    """Create RGB, heatmap, and blended overlay panels in one image."""

    rgb = Image.open(rgb_path).convert("RGB")
    heatmap = Image.open(heatmap_path).convert("RGB").resize(rgb.size)
    overlay = blend_heatmap(rgb, heatmap)
    width, height = rgb.size
    title_height = 44
    canvas = Image.new("RGB", (width * 3, height + title_height), color=(245, 245, 245))
    canvas.paste(rgb, (0, title_height))
    canvas.paste(heatmap, (width, title_height))
    canvas.paste(overlay, (width * 2, title_height))
    draw = ImageDraw.Draw(canvas)
    font = _default_font()
    label = caption or f"Query: {query}"
    draw.text((12, 12), "RGB", fill=(20, 20, 20), font=font)
    draw.text((width + 12, 12), "Relevancy", fill=(20, 20, 20), font=font)
    draw.text((width * 2 + 12, 12), label[:80], fill=(20, 20, 20), font=font)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def create_mock_rgb_and_heatmap(
    output_dir: str | Path,
    *,
    query: str,
    view_id: str = "view_0000",
    width: int = 512,
    height: int = 384,
) -> tuple[Path, Path, Path]:
    """Create deterministic mock RGB, heatmap, and overlay artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:height, 0:width]
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[..., 0] = np.clip(70 + x / width * 100, 0, 255)
    rgb_array[..., 1] = np.clip(90 + y / height * 90, 0, 255)
    rgb_array[..., 2] = 150
    view_offset = sum(ord(ch) for ch in view_id) % 17
    cx = width * (0.28 + ((len(query) + view_offset) % 6) * 0.08)
    cy = height * (0.30 + ((len(query) + view_offset) % 4) * 0.10)
    sigma = min(width, height) * 0.16
    scores = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma)))
    rgb = Image.fromarray(rgb_array, mode="RGB")
    heatmap = heatmap_from_scores(scores)
    rgb_path = directory / f"{view_id}_rgb.png"
    heatmap_path = directory / f"{view_id}_relevancy.png"
    overlay_path = directory / f"{view_id}_overlay.png"
    rgb.save(rgb_path)
    heatmap.save(heatmap_path)
    create_side_by_side_overlay(
        rgb_path,
        heatmap_path,
        overlay_path,
        query=query,
        caption=f"Dry-run query: {query} | {view_id}",
    )
    return rgb_path, heatmap_path, overlay_path


def _default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()
