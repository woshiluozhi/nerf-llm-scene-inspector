"""OpenNeRF secondary backend adapter."""

from __future__ import annotations

from pathlib import Path

from nerf_llm_scene_inspector.backends.base import QueryResult, RenderedView, SemanticFieldBackend
from nerf_llm_scene_inspector.backends.nerfstudio_backend import NerfstudioConfigMixin
from nerf_llm_scene_inspector.querying.relevancy_extractor import (
    best_render_score,
    extract_bbox_from_heatmap,
)
from nerf_llm_scene_inspector.utils.paths import utc_timestamp
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
                "notes": ["OpenNeRF adapter is secondary and may need checkout-specific render hooks."],
            },
        )
        self.export_query_artifacts(result, str(output))
        return result

    def render_relevancy(self, query: str, output_dir: str) -> list[RenderedView]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            rgb_path, heatmap_path, overlay_path = create_mock_rgb_and_heatmap(output, query=query)
            return [
                RenderedView(str(rgb_path), "rgb", query, "Dry-run RGB render", "view_0000"),
                RenderedView(str(heatmap_path), "relevancy", query, "Dry-run OpenNeRF heatmap", "view_0000"),
                RenderedView(str(overlay_path), "overlay", query, "Dry-run OpenNeRF overlay", "view_0000"),
            ]
        self.warnings.append(
            "OpenNeRF real-mode rendering is checkout-specific. Use the OpenNeRF viewer or add a "
            "project-specific render hook for this repository revision."
        )
        fallback = output / "opennerf_viewer_workflow.md"
        fallback.write_text(
            "\n".join(
                [
                    "# OpenNeRF Query Fallback",
                    "",
                    f"Run `{self.viewer_command_text()}` and use the installed OpenNeRF query UI.",
                    "Save rendered semantic outputs into this directory, then rerun evaluation.",
                ]
            ),
            encoding="utf-8",
        )
        return [RenderedView(str(fallback), "viewer_fallback", query, "OpenNeRF viewer workflow")]

    def export_query_artifacts(self, result: QueryResult, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result.to_json(output / "query_result.json")
