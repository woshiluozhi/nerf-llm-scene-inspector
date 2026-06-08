"""Offline HTML workbench for human-in-the-loop bbox annotation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from nerf_llm_scene_inspector.backends.base import QueryResult, RenderedView
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp


@dataclass
class AnnotationWorkbenchItem:
    """One query row shown in the annotation workbench."""

    query: str
    target_description: str = ""
    acceptable_views: list[str] = field(default_factory=list)
    bbox_2d: list[float] | None = None
    notes: str = ""
    candidate_views: list[str] = field(default_factory=list)
    candidate_bbox_2d_suggestions: list[dict[str, Any]] = field(default_factory=list)
    image_path: str = ""
    source_image: str = ""
    source_view: str = ""
    image_width: int | None = None
    image_height: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AnnotationWorkbench:
    """Generated annotation workbench artifact manifest."""

    scene_name: str
    annotations_path: str
    results_dir: str
    output_dir: str
    html_path: str
    seed_annotations_path: str
    manifest_path: str
    generated_at: str
    item_count: int
    image_count: int
    missing_image_count: int
    items: list[AnnotationWorkbenchItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_name": self.scene_name,
            "annotations_path": self.annotations_path,
            "results_dir": self.results_dir,
            "output_dir": self.output_dir,
            "html_path": self.html_path,
            "seed_annotations_path": self.seed_annotations_path,
            "manifest_path": self.manifest_path,
            "generated_at": self.generated_at,
            "item_count": self.item_count,
            "image_count": self.image_count,
            "missing_image_count": self.missing_image_count,
            "items": [item.to_dict() for item in self.items],
            "warnings": list(self.warnings),
        }


def build_annotation_workbench(
    *,
    annotations_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    title: str = "NeRF-LLM Annotation Workbench",
) -> AnnotationWorkbench:
    """Create a standalone HTML workbench for editing bbox annotations."""

    annotations = _read_annotations(Path(annotations_path))
    results_root = Path(results_dir)
    output_root = Path(output_dir)
    assets_dir = output_root / "assets"
    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    result_by_query = _load_query_results(results_root)
    scene_name = str(annotations.get("scene_name") or "scene")
    items = [
        _build_item(
            raw=item,
            result=result_by_query.get(str(item.get("query") or "")),
            results_root=results_root,
            output_root=output_root,
            assets_dir=assets_dir,
        )
        for item in annotations.get("queries") or []
        if isinstance(item, dict) and str(item.get("query") or "").strip()
    ]
    seed_payload = _seed_annotations(scene_name, items)
    seed_path = output_root / "annotation_seed.json"
    html_path = output_root / "annotation_workbench.html"
    manifest_path = output_root / "annotation_workbench_manifest.json"

    seed_path.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(title=title, scene_name=scene_name, items=items), encoding="utf-8")

    warnings = [warning for item in items for warning in item.warnings]
    workbench = AnnotationWorkbench(
        scene_name=scene_name,
        annotations_path=_display_path(Path(annotations_path), output_root),
        results_dir=_display_path(results_root, output_root),
        output_dir=".",
        html_path=_display_path(html_path, output_root),
        seed_annotations_path=_display_path(seed_path, output_root),
        manifest_path=_display_path(manifest_path, output_root),
        generated_at=utc_timestamp(),
        item_count=len(items),
        image_count=sum(1 for item in items if item.image_path),
        missing_image_count=sum(1 for item in items if not item.image_path),
        items=items,
        warnings=warnings,
    )
    manifest_path.write_text(json.dumps(workbench.to_dict(), indent=2), encoding="utf-8")
    return workbench


def _build_item(
    *,
    raw: dict[str, Any],
    result: QueryResult | None,
    results_root: Path,
    output_root: Path,
    assets_dir: Path,
) -> AnnotationWorkbenchItem:
    query = str(raw.get("query") or "").strip()
    candidate_boxes = _candidate_boxes(raw)
    initial_bbox = _initial_bbox(raw, candidate_boxes)
    warnings: list[str] = []
    selected_view = _select_rendered_view(raw, result)
    image_path = ""
    source_image = ""
    source_view = ""
    width: int | None = None
    height: int | None = None
    if selected_view is None:
        warnings.append("No rendered image found for this query; annotate after importing viewer outputs.")
    else:
        source = _resolve_rendered_path(selected_view.path, results_root, result_dir=results_root)
        if source is None:
            warnings.append(f"Rendered image path does not exist: {selected_view.path}")
        else:
            copied = _copy_asset(source, assets_dir, prefix=slugify(query))
            image_path = _display_path(copied, output_root)
            source_image = _display_path(source, output_root)
            source_view = selected_view.camera_id or Path(selected_view.path).stem
            try:
                with Image.open(copied) as image:
                    width, height = image.size
            except OSError as exc:
                warnings.append(f"Could not read copied image dimensions: {exc}")
    return AnnotationWorkbenchItem(
        query=query,
        target_description=str(raw.get("target_description") or ""),
        acceptable_views=[str(item) for item in raw.get("acceptable_views") or []],
        bbox_2d=initial_bbox,
        notes=str(raw.get("notes") or ""),
        candidate_views=[str(item) for item in raw.get("candidate_views") or []],
        candidate_bbox_2d_suggestions=candidate_boxes,
        image_path=image_path,
        source_image=source_image,
        source_view=source_view,
        image_width=width,
        image_height=height,
        warnings=warnings,
    )


def _read_annotations(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Annotation workbench input must be a JSON object.")
    queries = raw.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Annotation workbench input must contain a queries list.")
    return raw


def _load_query_results(results_dir: Path) -> dict[str, QueryResult]:
    if not results_dir.exists():
        return {}
    results: dict[str, QueryResult] = {}
    for path in sorted(results_dir.rglob("query_result.json")):
        try:
            result = QueryResult.from_json(path)
        except Exception:
            continue
        results.setdefault(result.query, result)
    return results


def _candidate_boxes(raw: dict[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for item in raw.get("candidate_bbox_2d_suggestions") or []:
        if not isinstance(item, dict):
            continue
        bbox = _bbox_list(item.get("bbox_2d"))
        if bbox is None:
            continue
        boxes.append(
            {
                "source_view": str(item.get("source_view") or ""),
                "bbox_2d": bbox,
                "score": item.get("score"),
                "notes": str(item.get("notes") or ""),
            }
        )
    return boxes


def _initial_bbox(raw: dict[str, Any], candidate_boxes: list[dict[str, Any]]) -> list[float] | None:
    manual = _bbox_list(raw.get("bbox_2d"))
    if manual is not None:
        return manual
    if candidate_boxes:
        bbox = candidate_boxes[0].get("bbox_2d")
        return list(bbox) if isinstance(bbox, list) else None
    return None


def _select_rendered_view(raw: dict[str, Any], result: QueryResult | None) -> RenderedView | None:
    if result is None or not result.rendered_images:
        return None
    preferred = {
        _normalize_view_id(str(value))
        for value in [
            *(raw.get("acceptable_views") or []),
            *(raw.get("candidate_views") or []),
            *[
                item.get("source_view")
                for item in raw.get("candidate_bbox_2d_suggestions") or []
                if isinstance(item, dict)
            ],
        ]
        if value
    }
    ordered = sorted(result.rendered_images, key=_view_sort_key)
    if preferred:
        for view in ordered:
            if _rendered_view_ids(view) & preferred:
                return view
    return ordered[0]


def _view_sort_key(view: RenderedView) -> tuple[int, str]:
    kind_order = {"rgb": 0, "overlay": 1, "composited": 2, "relevancy": 3}
    return kind_order.get(view.kind, 9), view.path


def _rendered_view_ids(view: RenderedView) -> set[str]:
    values = {view.camera_id or "", Path(view.path).name, Path(view.path).stem}
    return {_normalize_view_id(value) for value in values if value}


def _normalize_view_id(value: str) -> str:
    path = Path(value.strip().replace("\\", "/"))
    return path.stem.lower() if path.suffix else path.name.lower()


def _resolve_rendered_path(raw_path: str, results_root: Path, *, result_dir: Path) -> Path | None:
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [result_dir / raw, results_root / raw, raw]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(results_root.rglob(Path(raw_path).name)):
        if candidate.is_file():
            return candidate
    return None


def _copy_asset(source: Path, assets_dir: Path, *, prefix: str) -> Path:
    safe_name = f"{prefix}_{source.name}" if prefix else source.name
    destination = assets_dir / safe_name
    if destination.resolve() != source.resolve():
        shutil.copy2(source, destination)
    return destination


def _seed_annotations(scene_name: str, items: list[AnnotationWorkbenchItem]) -> dict[str, Any]:
    return {
        "scene_name": scene_name,
        "created_at": utc_timestamp(),
        "instructions": [
            "Generated from the offline annotation workbench.",
            "Inspect each bbox_2d before using quantitative metrics.",
        ],
        "queries": [
            {
                "query": item.query,
                "target_description": item.target_description,
                "acceptable_views": item.acceptable_views or ([item.source_view] if item.source_view else []),
                "bbox_2d": item.bbox_2d,
                "notes": item.notes,
            }
            for item in items
        ],
    }


def _render_html(*, title: str, scene_name: str, items: list[AnnotationWorkbenchItem]) -> str:
    data = {
        "scene_name": scene_name,
        "generated_at": utc_timestamp(),
        "items": [item.to_dict() for item in items],
    }
    json_data = json.dumps(data, indent=2).replace("</", "<\\/")
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  <title>{_escape_text(title)}</title>",
            "  <style>",
            _style_block(),
            "  </style>",
            "</head>",
            "<body>",
            '  <main class="app">',
            '    <aside class="sidebar">',
            f"      <h1>{_escape_text(scene_name)}</h1>",
            '      <div id="query-list" class="query-list"></div>',
            '      <button id="download-json" type="button">Download JSON</button>',
            "    </aside>",
            '    <section class="workspace">',
            '      <header class="toolbar">',
            '        <div><span class="eyebrow">Query</span><h2 id="query-title"></h2></div>',
            '        <div class="actions">',
            '          <button id="use-suggestion" type="button">Use Suggestion</button>',
            '          <button id="clear-box" type="button">Clear Box</button>',
            "        </div>",
            "      </header>",
            '      <section class="content">',
            '        <div class="canvas-wrap">',
            '          <img id="scene-image" alt="">',
            '          <canvas id="bbox-canvas"></canvas>',
            "        </div>",
            '        <form class="panel">',
            '          <label>Target<input id="target-description" type="text"></label>',
            '          <label>View<input id="acceptable-view" type="text"></label>',
            '          <label>BBox<input id="bbox-field" type="text"></label>',
            '          <label>Notes<textarea id="notes-field" rows="4"></textarea></label>',
            '          <div id="warnings" class="warnings"></div>',
            "        </form>",
            "      </section>",
            "    </section>",
            "  </main>",
            f'  <script type="application/json" id="workbench-data">{json_data}</script>',
            "  <script>",
            _script_block(),
            "  </script>",
            "</body>",
            "</html>",
        ]
    )


def _bbox_list(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _escape_text(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _style_block() -> str:
    return """
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5e6877;
      --line: #d8dee8;
      --accent: #0f766e;
      --warning: #8a4b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    .app { display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--line); background: var(--panel); padding: 18px; }
    h1 { margin: 0 0 16px; font-size: 1.25rem; overflow-wrap: anywhere; }
    h2 { margin: 0; font-size: 1.15rem; overflow-wrap: anywhere; }
    .eyebrow { display: block; color: var(--accent); font-size: 0.75rem; font-weight: 750; text-transform: uppercase; }
    .query-list { display: grid; gap: 8px; margin-bottom: 16px; }
    .query-button {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfe;
      padding: 10px;
      color: var(--ink);
      cursor: pointer;
    }
    .query-button.active { border-color: var(--accent); background: #e9f6f3; }
    button {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      color: var(--ink);
      padding: 9px 12px;
      font-weight: 650;
      cursor: pointer;
    }
    #download-json { width: 100%; background: var(--accent); color: white; border-color: var(--accent); }
    .workspace { min-width: 0; padding: 20px; }
    .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .content { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 16px; align-items: start; }
    .canvas-wrap { position: relative; background: #e9edf3; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; min-height: 320px; }
    #scene-image { display: block; max-width: 100%; height: auto; }
    #bbox-canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; display: grid; gap: 12px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 0.82rem; font-weight: 700; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      font: inherit;
      color: var(--ink);
      background: #fbfcfe;
    }
    .warnings { color: var(--warning); font-size: 0.88rem; display: grid; gap: 6px; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .content { grid-template-columns: 1fr; }
      .toolbar { align-items: flex-start; flex-direction: column; }
    }
    """.strip()


def _script_block() -> str:
    return r"""
    const state = JSON.parse(document.getElementById("workbench-data").textContent);
    let index = 0;
    let dragging = false;
    let start = null;
    const list = document.getElementById("query-list");
    const img = document.getElementById("scene-image");
    const canvas = document.getElementById("bbox-canvas");
    const ctx = canvas.getContext("2d");
    const title = document.getElementById("query-title");
    const target = document.getElementById("target-description");
    const view = document.getElementById("acceptable-view");
    const bboxField = document.getElementById("bbox-field");
    const notes = document.getElementById("notes-field");
    const warnings = document.getElementById("warnings");

    function item() { return state.items[index]; }
    function renderList() {
      list.innerHTML = "";
      state.items.forEach((entry, i) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "query-button" + (i === index ? " active" : "");
        button.textContent = entry.query + (entry.bbox_2d ? " | bbox" : "");
        button.onclick = () => { saveFields(); index = i; render(); };
        list.appendChild(button);
      });
    }
    function render() {
      renderList();
      const entry = item();
      title.textContent = entry.query;
      target.value = entry.target_description || "";
      view.value = (entry.acceptable_views && entry.acceptable_views[0]) || entry.source_view || "";
      notes.value = entry.notes || "";
      bboxField.value = entry.bbox_2d ? entry.bbox_2d.map(v => Math.round(v * 100) / 100).join(", ") : "";
      warnings.innerHTML = (entry.warnings || []).map(text => "<div>" + escapeHtml(text) + "</div>").join("");
      img.src = entry.image_path || "";
      img.alt = entry.query;
      img.onload = resizeCanvas;
      if (!entry.image_path) resizeCanvas();
    }
    function saveFields() {
      const entry = item();
      entry.target_description = target.value.trim();
      entry.acceptable_views = view.value.trim() ? [view.value.trim()] : [];
      entry.notes = notes.value.trim();
      entry.bbox_2d = parseBBox(bboxField.value);
    }
    function resizeCanvas() {
      const rect = img.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width));
      canvas.height = Math.max(1, Math.round(rect.height));
      draw();
    }
    function draw(tempBox) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const box = tempBox || item().bbox_2d;
      if (!box || !img.naturalWidth || !img.naturalHeight) return;
      const sx = canvas.width / img.naturalWidth;
      const sy = canvas.height / img.naturalHeight;
      ctx.strokeStyle = "#ff3b00";
      ctx.lineWidth = 3;
      ctx.strokeRect(box[0] * sx, box[1] * sy, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy);
    }
    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * img.naturalWidth / Math.max(1, rect.width);
      const y = (event.clientY - rect.top) * img.naturalHeight / Math.max(1, rect.height);
      return [Math.max(0, Math.min(img.naturalWidth, x)), Math.max(0, Math.min(img.naturalHeight, y))];
    }
    canvas.addEventListener("mousedown", event => {
      if (!img.naturalWidth) return;
      dragging = true;
      start = canvasPoint(event);
    });
    canvas.addEventListener("mousemove", event => {
      if (!dragging || !start) return;
      const point = canvasPoint(event);
      draw(normalizeBox([start[0], start[1], point[0], point[1]]));
    });
    window.addEventListener("mouseup", event => {
      if (!dragging || !start) return;
      dragging = false;
      const point = canvasPoint(event);
      item().bbox_2d = normalizeBox([start[0], start[1], point[0], point[1]]);
      bboxField.value = item().bbox_2d.map(v => Math.round(v * 100) / 100).join(", ");
      draw();
      renderList();
    });
    document.getElementById("use-suggestion").onclick = () => {
      const suggestion = (item().candidate_bbox_2d_suggestions || [])[0];
      if (suggestion && suggestion.bbox_2d) {
        item().bbox_2d = suggestion.bbox_2d.map(Number);
        bboxField.value = item().bbox_2d.join(", ");
        draw();
        renderList();
      }
    };
    document.getElementById("clear-box").onclick = () => {
      item().bbox_2d = null;
      bboxField.value = "";
      draw();
      renderList();
    };
    bboxField.addEventListener("change", () => {
      item().bbox_2d = parseBBox(bboxField.value);
      draw();
      renderList();
    });
    document.getElementById("download-json").onclick = () => {
      saveFields();
      const payload = {
        scene_name: state.scene_name,
        created_at: new Date().toISOString(),
        queries: state.items.map(entry => ({
          query: entry.query,
          target_description: entry.target_description || "",
          acceptable_views: entry.acceptable_views || [],
          bbox_2d: entry.bbox_2d || null,
          notes: entry.notes || ""
        }))
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "annotations_filled.json";
      link.click();
      URL.revokeObjectURL(url);
    };
    function parseBBox(value) {
      const parts = value.split(",").map(v => Number(v.trim())).filter(v => Number.isFinite(v));
      return parts.length === 4 ? normalizeBox(parts) : null;
    }
    function normalizeBox(box) {
      const x1 = Math.min(box[0], box[2]);
      const y1 = Math.min(box[1], box[3]);
      const x2 = Math.max(box[0], box[2]);
      const y2 = Math.max(box[1], box[3]);
      return [x1, y1, x2, y2];
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    window.addEventListener("resize", resizeCanvas);
    render();
    """.strip()
