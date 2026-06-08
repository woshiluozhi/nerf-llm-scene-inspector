"""Import manually saved Nerfstudio/LERF viewer outputs into QueryResult artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult, RenderedView
from nerf_llm_scene_inspector.querying.relevancy_extractor import (
    best_render_score,
    extract_bbox_from_heatmap,
)
from nerf_llm_scene_inspector.utils.paths import utc_timestamp
from nerf_llm_scene_inspector.visualization.render_overlays import create_side_by_side_overlay

ViewArtifactKind = Literal["rgb", "relevancy", "overlay", "composited", "artifact"]


@dataclass
class ViewerImportSummary:
    """Summary of a manual viewer import."""

    query: str
    input_dir: str
    output_dir: str
    query_result_path: str
    imported_files: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path


@dataclass
class _ViewGroup:
    view_id: str
    files: dict[ViewArtifactKind, Path] = field(default_factory=dict)


def import_viewer_outputs(
    *,
    query: str,
    config_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    backend_name: str = "lerf",
    threshold_quantile: float = 0.9,
    create_missing_overlays: bool = True,
) -> tuple[QueryResult, ViewerImportSummary]:
    """Create a QueryResult from files manually saved from the Nerfstudio viewer."""

    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Viewer output directory does not exist: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    imported_files: list[str] = []
    generated_files: list[str] = []
    rendered_views: list[RenderedView] = []
    regions: list[BoundingRegion] = []
    scores: list[float] = []

    groups = _collect_view_groups(source_dir)
    if not groups:
        warnings.append("No supported viewer image files were found.")

    for group in groups:
        copied = {
            kind: _copy_artifact(path, destination_dir, imported_files)
            for kind, path in group.files.items()
        }
        rendered_views.extend(_rendered_views(query, group.view_id, copied))

        heatmap_path = copied.get("relevancy")
        if heatmap_path is not None:
            region = extract_bbox_from_heatmap(
                heatmap_path,
                label=query,
                threshold_quantile=threshold_quantile,
                source_view=group.view_id,
            )
            if region is not None:
                regions.append(region)
            score = best_render_score(heatmap_path)
            if score is not None:
                scores.append(score)

        rgb_path = copied.get("rgb")
        overlay_path = copied.get("overlay")
        if (
            create_missing_overlays
            and rgb_path is not None
            and heatmap_path is not None
            and overlay_path is None
        ):
            overlay_path = destination_dir / f"{group.view_id}_overlay.png"
            create_side_by_side_overlay(rgb_path, heatmap_path, overlay_path, query=query)
            generated_files.append(str(overlay_path))
            rendered_views.append(
                _view_from_file(overlay_path, kind="overlay", query=query, view_id=group.view_id)
            )

        if heatmap_path is None:
            warnings.append(f"No relevancy heatmap found for {group.view_id}; bbox extraction skipped.")
        if rgb_path is None:
            warnings.append(f"No RGB image found for {group.view_id}; overlay generation may be unavailable.")

    result = QueryResult(
        query=query,
        backend_name=backend_name,
        config_path=str(config_path),
        rendered_images=rendered_views,
        bounding_regions=regions,
        confidence=max(scores) if scores else None,
        warnings=warnings,
        provenance={
            "timestamp": utc_timestamp(),
            "manual_viewer_import": True,
            "input_dir": str(source_dir),
            "output_dir": str(destination_dir),
            "threshold_quantile": threshold_quantile,
            "notes": [
                "Imported from manually saved Nerfstudio/LERF viewer outputs.",
                "2D boxes are image-space estimates from relevancy heatmaps.",
            ],
        },
    )
    query_result_path = result.to_json(destination_dir / "query_result.json")
    summary = ViewerImportSummary(
        query=query,
        input_dir=str(source_dir),
        output_dir=str(destination_dir),
        query_result_path=str(query_result_path),
        imported_files=imported_files,
        generated_files=generated_files,
        warnings=warnings,
    )
    summary.to_json(destination_dir / "viewer_import_summary.json")
    return result, summary


def _collect_view_groups(source_dir: Path) -> list[_ViewGroup]:
    groups: dict[str, _ViewGroup] = {}
    for path in sorted(source_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        kind = _artifact_kind(path)
        if kind == "artifact":
            continue
        view_id = _view_id(path, kind)
        groups.setdefault(view_id, _ViewGroup(view_id=view_id)).files[kind] = path
    return [groups[key] for key in sorted(groups)]


def _artifact_kind(path: Path) -> ViewArtifactKind:
    stem = path.stem.lower()
    if stem.endswith("_rgb") or stem.endswith("_color"):
        return "rgb"
    if stem.endswith("_relevancy") or stem.endswith("_relevancy_0") or stem.endswith("_heatmap"):
        return "relevancy"
    if stem.endswith("_overlay"):
        return "overlay"
    if stem.endswith("_composited") or stem.endswith("_composited_0"):
        return "composited"
    return "artifact"


def _view_id(path: Path, kind: ViewArtifactKind) -> str:
    stem = path.stem
    suffixes = {
        "rgb": ["_rgb", "_color"],
        "relevancy": ["_relevancy_0", "_relevancy", "_heatmap"],
        "overlay": ["_overlay"],
        "composited": ["_composited_0", "_composited"],
        "artifact": [],
    }[kind]
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _copy_artifact(path: Path, destination_dir: Path, imported_files: list[str]) -> Path:
    destination = destination_dir / path.name
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    imported_files.append(str(destination))
    return destination


def _rendered_views(
    query: str,
    view_id: str,
    files: dict[ViewArtifactKind, Path],
) -> list[RenderedView]:
    ordered: list[RenderedView] = []
    for kind in ("rgb", "relevancy", "composited", "overlay"):
        path = files.get(kind)
        if path is not None:
            ordered.append(_view_from_file(path, kind=kind, query=query, view_id=view_id))
    return ordered


def _view_from_file(path: Path, *, kind: str, query: str, view_id: str) -> RenderedView:
    width = None
    height = None
    try:
        width, height = Image.open(path).size
    except OSError:
        pass
    return RenderedView(
        path=str(path),
        kind=kind,
        query=query,
        caption=f"Imported viewer {kind} for '{query}'",
        camera_id=view_id,
        width=width,
        height=height,
        score=best_render_score(path) if kind == "relevancy" else None,
    )
