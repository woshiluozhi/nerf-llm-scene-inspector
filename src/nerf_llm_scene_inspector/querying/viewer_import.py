"""Import manually saved Nerfstudio/LERF viewer outputs into QueryResult artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from nerf_llm_scene_inspector.backends.base import (
    BoundingRegion,
    QueryResult,
    RenderedView,
    SceneQueryReport,
)
from nerf_llm_scene_inspector.querying.answer_synthesis import synthesize_scene_answer
from nerf_llm_scene_inspector.querying.relevancy_extractor import (
    best_render_score,
    extract_bbox_from_heatmap,
)
from nerf_llm_scene_inspector.querying.spatial_reasoning import aggregate_multi_query_results
from nerf_llm_scene_inspector.utils.paths import slugify
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
class SceneQueryRepairSummary:
    """Summary for repairing a scene query report with manual viewer outputs."""

    report_path: str
    viewer_root: str
    output_report_path: str
    markdown_report_path: str
    repaired_queries: list[str] = field(default_factory=list)
    kept_queries: list[str] = field(default_factory=list)
    missing_viewer_dirs: list[str] = field(default_factory=list)
    missing_required_queries: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required_queries

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload

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


def repair_scene_query_report_from_viewer_outputs(
    *,
    report_path: str | Path,
    viewer_root: str | Path,
    output_report_path: str | Path | None = None,
    markdown_report_path: str | Path | None = None,
    threshold_quantile: float = 0.9,
    create_missing_overlays: bool = True,
    require_all: bool = False,
) -> tuple[SceneQueryReport, SceneQueryRepairSummary]:
    """Replace query results in a scene report with manually saved viewer outputs."""

    source_report = Path(report_path)
    root = Path(viewer_root)
    if not source_report.exists():
        raise FileNotFoundError(f"Scene query report does not exist: {source_report}")
    if not root.exists():
        raise FileNotFoundError(f"Viewer output root does not exist: {root}")

    raw = json.loads(source_report.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Scene query report must be a JSON object.")

    report_dir = source_report.parent
    output_json = Path(output_report_path) if output_report_path else source_report
    output_md = Path(markdown_report_path) if markdown_report_path else output_json.with_suffix(".md")
    results = [QueryResult.from_dict(item) for item in raw.get("query_results") or [] if isinstance(item, dict)]
    repaired: list[QueryResult] = []
    repaired_queries: list[str] = []
    kept_queries: list[str] = []
    missing_dirs: list[str] = []
    missing_required: list[str] = []
    generated_files: list[str] = []
    warnings: list[str] = []
    used_slugs: dict[str, int] = {}

    for index, result in enumerate(results):
        query_dir = report_dir / _unique_query_slug(result.query, used_slugs)
        manual_dir = _viewer_dir_for_query(root, result.query, index)
        if manual_dir is None:
            kept_queries.append(result.query)
            missing_dirs.append(result.query)
            if require_all:
                missing_required.append(result.query)
            elif _has_viewer_fallback(result):
                warnings.append(
                    f"No manual viewer output directory found for fallback query '{result.query}'."
                )
            repaired.append(result)
            continue

        imported, import_summary = import_viewer_outputs(
            query=result.query,
            config_path=result.config_path,
            input_dir=manual_dir,
            output_dir=query_dir,
            backend_name=result.backend_name,
            threshold_quantile=threshold_quantile,
            create_missing_overlays=create_missing_overlays,
        )
        imported.provenance["repaired_scene_query_report"] = str(source_report)
        imported.to_json(query_dir / "query_result.json")
        repaired.append(imported)
        repaired_queries.append(result.query)
        generated_files.extend(import_summary.generated_files)
        warnings.extend(import_summary.warnings)

    aggregate = aggregate_multi_query_results(repaired)
    plan = dict(raw.get("plan") or {})
    answer = synthesize_scene_answer(
        task=str(raw.get("task") or ""),
        plan=plan,
        results=repaired,
    )
    report_warnings = _dedupe(_plan_warnings(plan) + aggregate.warnings + warnings)
    repaired_report = SceneQueryReport(
        scene_name=str(raw.get("scene_name") or "unknown"),
        task=str(raw.get("task") or ""),
        plan=plan,
        query_results=repaired,
        answer=answer.answer,
        answer_summary=answer.to_dict(),
        warnings=report_warnings,
    )
    repaired_report.to_json(output_json)
    repaired_report.to_markdown(output_md)
    summary = SceneQueryRepairSummary(
        report_path=str(source_report),
        viewer_root=str(root),
        output_report_path=str(output_json),
        markdown_report_path=str(output_md),
        repaired_queries=repaired_queries,
        kept_queries=kept_queries,
        missing_viewer_dirs=missing_dirs,
        missing_required_queries=missing_required,
        generated_files=generated_files,
        warnings=report_warnings,
    )
    summary.to_json(output_json.with_name("viewer_repair_summary.json"))
    return repaired_report, summary


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


def _viewer_dir_for_query(root: Path, query: str, index: int) -> Path | None:
    slug = slugify(query)
    candidates = [
        root / slug,
        root / query,
        root / f"{index:02d}_{slug}",
        root / f"{index + 1:02d}_{slug}",
    ]
    for candidate in candidates:
        if candidate.is_dir() and _collect_view_groups(candidate):
            return candidate
    return None


def _unique_query_slug(query: str, used_slugs: dict[str, int]) -> str:
    base = slugify(query)
    count = used_slugs.get(base, 0) + 1
    used_slugs[base] = count
    return base if count == 1 else f"{base}_{count}"


def _has_viewer_fallback(result: QueryResult) -> bool:
    if any(view.kind == "viewer_fallback" for view in result.rendered_images):
        return True
    return any("viewer fallback" in warning.lower() for warning in result.warnings)


def _plan_warnings(plan: dict[str, object]) -> list[str]:
    warnings = plan.get("warnings")
    return [str(item) for item in warnings] if isinstance(warnings, list) else []


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
