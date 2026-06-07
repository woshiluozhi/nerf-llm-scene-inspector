"""Utilities for extracting simple regions from rendered relevancy maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from nerf_llm_scene_inspector.backends.base import BoundingRegion


def extract_bbox_from_heatmap(
    heatmap_path: str | Path,
    *,
    label: str,
    threshold_quantile: float = 0.9,
    source_view: str | None = None,
) -> BoundingRegion | None:
    """Extract a rough 2D bounding box from a heatmap image."""

    path = Path(heatmap_path)
    if not path.exists():
        return None
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    if array.size == 0:
        return None
    threshold = float(np.quantile(array, threshold_quantile))
    mask = array >= threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    score = float(array[mask].mean() / 255.0)
    return BoundingRegion(
        label=label,
        score=score,
        coordinate_frame="image",
        bbox_2d=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        source_view=source_view or path.name,
        notes="Extracted from rendered heatmap using an image-space threshold.",
    )


def best_render_score(heatmap_path: str | Path) -> float | None:
    """Return the max normalized heatmap score."""

    path = Path(heatmap_path)
    if not path.exists():
        return None
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    if array.size == 0:
        return None
    return float(array.max() / 255.0)
