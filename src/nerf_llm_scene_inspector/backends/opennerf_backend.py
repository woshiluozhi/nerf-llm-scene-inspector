"""OpenNeRF secondary backend adapter."""

from __future__ import annotations

import json
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import QueryResult, RenderedView, SemanticFieldBackend
from nerf_llm_scene_inspector.backends.nerfstudio_backend import NerfstudioConfigMixin
from nerf_llm_scene_inspector.querying.relevancy_extractor import (
    best_render_score,
    extract_bbox_from_heatmap,
)
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp
from nerf_llm_scene_inspector.visualization.render_overlays import create_mock_rgb_and_heatmap


OPENNERF_INSTALL_INSTRUCTIONS = """Optional OpenNeRF setup:

git clone https://github.com/opennerf/opennerf
cd opennerf
python -m pip install -e .
ns-install-cli
ns-train -h

Confirm that the expected OpenNeRF method appears in ns-train help."""


class OpenNeRFBackend(NerfstudioConfigMixin, SemanticFieldBackend):
    """Secondary adapter for OpenNeRF-style semantic fields."""

    backend_name = "opennerf"

    def __init__(
        self,
        dry_run: bool = False,
        *,
        num_views: int = 1,
        save_manual_template: bool = False,
        strict_backend: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.num_views = max(1, int(num_views))
        self.save_manual_template = save_manual_template
        self.strict_backend = strict_backend

    def query_text(self, query: str, output_dir: str, top_k: int = 5) -> QueryResult:
        if not self.config_path:
            raise RuntimeError("Call load(config_path) before query_text().")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.warnings = []
        rendered = self.render_relevancy(query, str(output))
        heatmaps = [view for view in rendered if view.kind == "relevancy"]
        regions = [
            region
            for view in heatmaps
            if (region := extract_bbox_from_heatmap(view.path, label=query, source_view=view.camera_id))
            is not None
        ][:top_k]
        scores = [score for view in heatmaps if (score := best_render_score(view.path)) is not None]
        result = QueryResult(
            query=query,
            backend_name=self.backend_name,
            config_path=self.config_path,
            rendered_images=rendered,
            bounding_regions=regions,
            confidence=max(scores) if scores else None,
            warnings=list(self.warnings),
            provenance={
                "timestamp": utc_timestamp(),
                "commands": [self.viewer_command()],
                "model_config_path": self.config_path,
                "num_views": self.num_views,
                "notes": [
                    "OpenNeRF adapter is secondary.",
                    "Real-mode automated rendering is checkout-specific; viewer import/repair keeps outputs structured.",
                ],
            },
        )
        self.export_query_artifacts(result, str(output))
        return result

    def render_relevancy(self, query: str, output_dir: str) -> list[RenderedView]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return self._render_mock(query, output)

        try:
            views = self._render_with_opennerf_hook(query, output)
            if views:
                return views
        except Exception as exc:  # pragma: no cover - depends on upstream OpenNeRF installs
            if self.strict_backend:
                raise RuntimeError(
                    "Automated OpenNeRF rendering failed in --strict-backend mode. "
                    "OpenNeRF rendering hooks are checkout-specific; verify the installed revision "
                    f"or use viewer import/repair. Original error: {exc}"
                ) from exc
            self.warnings.append(
                "OpenNeRF automated rendering is not available for this checkout; "
                f"wrote viewer fallback instructions. Reason: {exc}"
            )
        return self._write_viewer_fallback(query, output)

    def _render_mock(self, query: str, output: Path) -> list[RenderedView]:
        views: list[RenderedView] = []
        for index in range(self.num_views):
            view_id = f"view_{index:04d}"
            rgb_path, heatmap_path, overlay_path = create_mock_rgb_and_heatmap(
                output,
                query=query,
                view_id=view_id,
            )
            views.extend(
                [
                    RenderedView(str(rgb_path), "rgb", query, "Dry-run RGB render", view_id),
                    RenderedView(
                        str(heatmap_path),
                        "relevancy",
                        query,
                        "Dry-run OpenNeRF heatmap",
                        view_id,
                        score=1.0,
                    ),
                    RenderedView(str(overlay_path), "overlay", query, "Dry-run OpenNeRF overlay", view_id),
                ]
            )
        return views

    def _write_viewer_fallback(self, query: str, output: Path) -> list[RenderedView]:
        self.warnings.append(
            "Use the OpenNeRF viewer and repair the scene query report from manually saved outputs."
        )
        fallback = output / "opennerf_viewer_workflow.md"
        template_path = (
            write_opennerf_query_template(query, self.config_path or "", output)
            if self.save_manual_template
            else None
        )
        template_lines = (
            [f"4. Fill or edit this JSON template if needed: `{template_path.name}`."]
            if template_path is not None
            else ["4. Use `scripts/import_viewer_outputs.py` for a single query directory if needed."]
        )
        fallback.write_text(
            "\n".join(
                [
                    "# OpenNeRF Query Fallback",
                    "",
                    "Automated OpenNeRF rendering is not standardized across repository revisions.",
                    "",
                    "Run:",
                    "",
                    f"```bash\n{self.viewer_command_text()}\n```",
                    "",
                    "Then in the installed OpenNeRF or Nerfstudio viewer:",
                    "",
                    f"1. Enter or select the text prompt: `{query}`",
                    f"2. Save RGB and semantic/relevancy outputs into: `{output}`",
                    "3. Name screenshots like `view_0000_rgb.png`, `view_0000_relevancy.png`,",
                    "   and `view_0000_overlay.png` when possible.",
                    *template_lines,
                    "5. Repair the full scene query report after saving all prompt folders:",
                    "",
                    "```bash",
                    "python scripts/repair_scene_query_from_viewer.py \\",
                    "  --report results/pipeline_runs/desk_scene/queries/mug/scene_query_report.json \\",
                    "  --viewer-root results/manual_viewer",
                    "```",
                    "",
                    "For a single query directory, use scripts/import_viewer_outputs.py.",
                ]
            ),
            encoding="utf-8",
        )
        views = [RenderedView(str(fallback), "viewer_fallback", query, "OpenNeRF viewer workflow")]
        if template_path is not None:
            views.append(RenderedView(str(template_path), "manual_template", query, "Manual OpenNeRF QueryResult template"))
        return views

    def _render_with_opennerf_hook(self, query: str, output: Path) -> list[RenderedView]:
        """Placeholder for checkout-specific OpenNeRF render hooks.

        OpenNeRF has changed across public revisions and does not expose one stable
        query-rendering CLI in the way this project can depend on in CPU-only CI.
        Integrators can monkeypatch or subclass this hook for a specific checkout.
        """

        raise RuntimeError(
            "No stable OpenNeRF automated render hook is configured for this checkout."
        )

    def export_query_artifacts(self, result: QueryResult, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result.to_json(output / "query_result.json")


def write_opennerf_query_template(query: str, config_path: str, output_dir: str | Path) -> Path:
    """Write a structured template for manual OpenNeRF viewer outputs."""

    path = Path(output_dir) / f"{slugify(query)}_opennerf_manual_report_template.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "backend_name": "opennerf",
        "config_path": config_path,
        "rendered_images": [],
        "candidate_points": [],
        "bounding_regions": [],
        "confidence": None,
        "warnings": ["Manual OpenNeRF viewer output template."],
        "provenance": {"timestamp": utc_timestamp(), "manual": True},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
