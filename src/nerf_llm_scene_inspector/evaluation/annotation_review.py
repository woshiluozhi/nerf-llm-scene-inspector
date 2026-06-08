"""Visual review artifacts for manual query annotations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from nerf_llm_scene_inspector.backends.base import QueryResult, RenderedView
from nerf_llm_scene_inspector.evaluation.annotation_schema import QueryAnnotation, load_annotations
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp


@dataclass
class AnnotationReviewItem:
    """Review status for one manual query annotation."""

    query: str
    status: str
    target_description: str = ""
    source_view: str = ""
    source_image: str = ""
    review_image: str = ""
    bbox_2d: tuple[float, float, float, float] | None = None
    image_width: int | None = None
    image_height: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bbox_2d"] = list(self.bbox_2d) if self.bbox_2d else None
        return payload


@dataclass
class AnnotationReviewReport:
    """Portable report for annotation visual QA."""

    scene_name: str
    annotations_path: str
    results_dir: str
    output_dir: str
    ok: bool
    total_annotations: int
    reviewed_annotations: int
    bbox_annotations: int
    warning_count: int
    contact_sheet: str = ""
    items: list[AnnotationReviewItem] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_name": self.scene_name,
            "annotations_path": self.annotations_path,
            "results_dir": self.results_dir,
            "output_dir": self.output_dir,
            "ok": self.ok,
            "total_annotations": self.total_annotations,
            "reviewed_annotations": self.reviewed_annotations,
            "bbox_annotations": self.bbox_annotations,
            "warning_count": self.warning_count,
            "contact_sheet": self.contact_sheet,
            "items": [item.to_dict() for item in self.items],
            "timestamp": self.timestamp,
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Annotation Review: {self.scene_name}",
            "",
            "This report visualizes manual `bbox_2d` annotations against query render artifacts.",
            "It is a QA aid for portfolio metrics, not an automatic ground-truthing tool.",
            "",
            f"- OK: {self.ok}",
            f"- Total annotations: {self.total_annotations}",
            f"- BBox annotations: {self.bbox_annotations}",
            f"- Reviewed annotations: {self.reviewed_annotations}",
            f"- Warnings: {self.warning_count}",
        ]
        if self.contact_sheet:
            lines.extend(["", f"![Annotation contact sheet]({self.contact_sheet})"])
        lines.extend(
            [
                "",
                "## Items",
                "",
                "| Query | Status | View | Review Image | Warnings |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in self.items:
            review_link = f"[open]({item.review_image})" if item.review_image else "n/a"
            warnings = "; ".join(item.warnings).replace("|", "/") if item.warnings else ""
            lines.append(
                f"| {item.query} | {item.status} | {item.source_view or 'n/a'} | "
                f"{review_link} | {warnings} |"
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


@dataclass
class _LoadedResult:
    result: QueryResult
    path: Path


def build_annotation_review(
    *,
    annotations_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    max_sheet_columns: int = 2,
) -> AnnotationReviewReport:
    """Build visual QA artifacts for manual annotations."""

    annotations = load_annotations(annotations_path)
    results_root = Path(results_dir)
    output_root = Path(output_dir)
    images_dir = output_root / "annotation_review_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    result_by_query = _load_query_results(results_root)

    items = [
        _review_annotation(
            annotation,
            result_by_query.get(annotation.query),
            results_root=results_root,
            output_root=output_root,
            images_dir=images_dir,
        )
        for annotation in annotations.queries
    ]
    contact_sheet = _write_contact_sheet(items, output_root, max_columns=max_sheet_columns)
    warning_count = sum(len(item.warnings) for item in items)
    reviewed_count = sum(1 for item in items if item.review_image)
    bbox_count = sum(1 for item in items if item.bbox_2d is not None)
    ok = all(item.status in {"ready", "qualitative_only"} for item in items)
    return AnnotationReviewReport(
        scene_name=annotations.scene_name,
        annotations_path=_display_path(Path(annotations_path), output_root),
        results_dir=_display_path(results_root, output_root),
        output_dir=".",
        ok=ok,
        total_annotations=len(items),
        reviewed_annotations=reviewed_count,
        bbox_annotations=bbox_count,
        warning_count=warning_count,
        contact_sheet=contact_sheet,
        items=items,
    )


def _review_annotation(
    annotation: QueryAnnotation,
    loaded: _LoadedResult | None,
    *,
    results_root: Path,
    output_root: Path,
    images_dir: Path,
) -> AnnotationReviewItem:
    warnings: list[str] = []
    if loaded is None:
        return AnnotationReviewItem(
            query=annotation.query,
            status="missing_result",
            target_description=annotation.target_description,
            bbox_2d=annotation.bbox_2d,
            warnings=["No query_result.json was found for this annotation."],
        )
    if annotation.bbox_2d is None:
        return AnnotationReviewItem(
            query=annotation.query,
            status="qualitative_only",
            target_description=annotation.target_description,
            warnings=["Annotation has no bbox_2d; kept as qualitative-only."],
        )

    selected = _select_rendered_view(annotation, loaded.result.rendered_images)
    if selected is None:
        return AnnotationReviewItem(
            query=annotation.query,
            status="missing_image",
            target_description=annotation.target_description,
            bbox_2d=annotation.bbox_2d,
            warnings=["Query result has no rendered image suitable for annotation review."],
        )
    view, matched_preferred = selected
    if annotation.acceptable_views and not matched_preferred:
        warnings.append(
            "Preferred acceptable_views were not found in rendered images; drew bbox on fallback view."
        )
    source_path = _resolve_rendered_path(view.path, loaded.path.parent, results_root)
    if source_path is None:
        return AnnotationReviewItem(
            query=annotation.query,
            status="missing_image",
            target_description=annotation.target_description,
            source_view=view.camera_id or "",
            source_image=view.path,
            bbox_2d=annotation.bbox_2d,
            warnings=[f"Rendered image path does not exist: {view.path}"],
        )
    try:
        review_path, image_size, bbox_warnings = _draw_review_image(
            source_path,
            annotation,
            images_dir / f"{slugify(annotation.query)}_review.png",
        )
    except OSError as exc:
        return AnnotationReviewItem(
            query=annotation.query,
            status="image_unreadable",
            target_description=annotation.target_description,
            source_view=view.camera_id or "",
            source_image=_display_path(source_path, output_root),
            bbox_2d=annotation.bbox_2d,
            warnings=[f"Could not read image: {exc}"],
        )
    warnings.extend(bbox_warnings)
    status = "ready"
    if bbox_warnings:
        status = "bbox_out_of_bounds"
    elif annotation.acceptable_views and not matched_preferred:
        status = "view_fallback"
    return AnnotationReviewItem(
        query=annotation.query,
        status=status,
        target_description=annotation.target_description,
        source_view=view.camera_id or Path(view.path).stem,
        source_image=_display_path(source_path, output_root),
        review_image=_display_path(review_path, output_root),
        bbox_2d=annotation.bbox_2d,
        image_width=image_size[0],
        image_height=image_size[1],
        warnings=warnings,
    )


def _load_query_results(results_dir: Path) -> dict[str, _LoadedResult]:
    loaded: dict[str, _LoadedResult] = {}
    if not results_dir.exists():
        return loaded
    for path in sorted(results_dir.rglob("query_result.json")):
        try:
            result = QueryResult.from_json(path)
        except Exception:
            continue
        loaded.setdefault(result.query, _LoadedResult(result=result, path=path))
    return loaded


def _select_rendered_view(
    annotation: QueryAnnotation,
    rendered_images: list[RenderedView],
) -> tuple[RenderedView, bool] | None:
    if not rendered_images:
        return None
    preferred = {_normalize_view_id(view_id) for view_id in annotation.acceptable_views}
    ordered = sorted(rendered_images, key=_view_sort_key)
    if preferred:
        for view in ordered:
            if _rendered_view_ids(view) & preferred:
                return view, True
    return ordered[0], not preferred


def _view_sort_key(view: RenderedView) -> tuple[int, str]:
    kind_order = {"rgb": 0, "relevancy": 1, "composited": 2, "overlay": 3}
    return kind_order.get(view.kind, 9), view.path


def _rendered_view_ids(view: RenderedView) -> set[str]:
    values = {view.camera_id or "", Path(view.path).name, Path(view.path).stem}
    return {_normalize_view_id(value) for value in values if value}


def _normalize_view_id(value: str) -> str:
    path = Path(value.strip().replace("\\", "/"))
    return path.stem.lower() if path.suffix else path.name.lower()


def _resolve_rendered_path(raw_path: str, result_dir: Path, results_root: Path) -> Path | None:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [result_dir / path, results_root / path, path]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _draw_review_image(
    source_path: Path,
    annotation: QueryAnnotation,
    output_path: Path,
) -> tuple[Path, tuple[int, int], list[str]]:
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    warnings: list[str] = []
    bbox = annotation.bbox_2d
    if bbox is None:
        raise ValueError("Annotation has no bbox_2d.")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        warnings.append("bbox_2d has non-positive width or height.")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        warnings.append(f"bbox_2d extends outside image bounds {width}x{height}.")
    clamped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    draw_box = (
        min(clamped[0], clamped[2]),
        min(clamped[1], clamped[3]),
        max(clamped[0], clamped[2]),
        max(clamped[1], clamped[3]),
    )
    draw = ImageDraw.Draw(image)
    color = (255, 92, 0) if warnings else (255, 0, 0)
    line_width = max(3, round(min(width, height) * 0.008))
    draw.rectangle(draw_box, outline=color, width=line_width)
    label = annotation.query
    if annotation.target_description:
        label = f"{annotation.query}: {annotation.target_description}"
    _draw_label(draw, label, draw_box[0], draw_box[1], color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path, (width, height), warnings


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: float,
    y: float,
    color: tuple[int, int, int],
) -> None:
    font = ImageFont.load_default()
    text = label[:80]
    left = int(max(0, x))
    top = int(max(0, y - 16))
    try:
        bbox = draw.textbbox((left, top), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
    except Exception:
        pass
    draw.text((left, top), text, fill=color, font=font)


def _write_contact_sheet(
    items: list[AnnotationReviewItem],
    output_root: Path,
    *,
    max_columns: int,
) -> str:
    image_paths = [output_root / item.review_image for item in items if item.review_image]
    if not image_paths:
        return ""
    columns = max(1, min(max_columns, len(image_paths)))
    rows = math.ceil(len(image_paths) / columns)
    thumb_width = 420
    caption_height = 52
    thumbs: list[tuple[Image.Image, str]] = []
    for item, path in zip([item for item in items if item.review_image], image_paths, strict=False):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, 280))
        tile = Image.new("RGB", (thumb_width, image.height + caption_height), "white")
        tile.paste(image, ((thumb_width - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        caption = _short_text(f"{item.query} | {item.status}", 58)
        draw.text((8, image.height + 8), caption, fill=(20, 20, 20), font=ImageFont.load_default())
        if item.warnings:
            warning = _short_text(item.warnings[0], 58)
            draw.text((8, image.height + 26), warning, fill=(150, 80, 0), font=ImageFont.load_default())
        thumbs.append((tile, item.query))
    tile_height = max(tile.height for tile, _ in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * tile_height), (245, 247, 250))
    for index, (tile, _) in enumerate(thumbs):
        col = index % columns
        row = index // columns
        sheet.paste(tile, (col * thumb_width, row * tile_height))
    output_path = output_root / "annotation_review_contact_sheet.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return _display_path(output_path, output_root)


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _short_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
