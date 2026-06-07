"""LERF backend adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nerf_llm_scene_inspector.backends.base import (
    BoundingRegion,
    QueryResult,
    RenderedView,
    SemanticFieldBackend,
)
from nerf_llm_scene_inspector.backends.nerfstudio_backend import NerfstudioConfigMixin
from nerf_llm_scene_inspector.querying.relevancy_extractor import (
    best_render_score,
    extract_bbox_from_heatmap,
)
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp
from nerf_llm_scene_inspector.visualization.render_overlays import (
    create_mock_rgb_and_heatmap,
    create_side_by_side_overlay,
)


LERF_INSTALL_INSTRUCTIONS = """Install LERF in the same environment as Nerfstudio:

git clone https://github.com/kerrj/lerf
cd lerf
python -m pip install -e .
ns-install-cli
ns-train -h

The ns-train help output should include lerf, lerf-lite, and lerf-big."""


class LERFBackend(NerfstudioConfigMixin, SemanticFieldBackend):
    """Practical adapter for LERF semantic relevancy rendering."""

    backend_name = "lerf"

    def query_text(self, query: str, output_dir: str, top_k: int = 5) -> QueryResult:
        if not self.config_path:
            raise RuntimeError("Call load(config_path) before query_text().")
        query_dir = Path(output_dir)
        query_dir.mkdir(parents=True, exist_ok=True)
        self.warnings = []
        rendered = self.render_relevancy(query, str(query_dir))

        heatmap_views = [view for view in rendered if view.kind == "relevancy"]
        regions: list[BoundingRegion] = []
        scores: list[float] = []
        for view in heatmap_views:
            region = extract_bbox_from_heatmap(view.path, label=query, source_view=view.camera_id)
            if region is not None:
                regions.append(region)
            score = best_render_score(view.path)
            if score is not None:
                scores.append(score)
        confidence = max(scores) if scores else None
        result = QueryResult(
            query=query,
            backend_name=self.backend_name,
            config_path=self.config_path,
            rendered_images=rendered,
            bounding_regions=regions[:top_k],
            confidence=confidence,
            warnings=list(self.warnings),
            provenance={
                "timestamp": utc_timestamp(),
                "commands": [self.viewer_command()],
                "model_config_path": self.config_path,
                "notes": [
                    "LERF positive prompts are set through image_encoder.set_positives.",
                    "Expected LERF render outputs include relevancy_0 and composited_0.",
                ],
            },
        )
        self.export_query_artifacts(result, str(query_dir))
        return result

    def render_relevancy(self, query: str, output_dir: str) -> list[RenderedView]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return self._render_mock(query, output)

        try:
            views = self._render_with_lerf_internal_api(query, output)
            if views:
                return views
        except Exception as exc:  # pragma: no cover - depends on upstream installs
            self.warnings.append(
                "Automated LERF rendering failed; wrote viewer fallback instructions. "
                f"Reason: {exc}"
            )
        return self._write_viewer_fallback(query, output)

    def export_query_artifacts(self, result: QueryResult, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result.to_json(output / "query_result.json")

    def _render_mock(self, query: str, output: Path) -> list[RenderedView]:
        rgb_path, heatmap_path, overlay_path = create_mock_rgb_and_heatmap(output, query=query)
        return [
            RenderedView(
                path=str(rgb_path),
                kind="rgb",
                query=query,
                caption=f"Dry-run RGB render for '{query}'",
                camera_id="view_0000",
                width=512,
                height=384,
            ),
            RenderedView(
                path=str(heatmap_path),
                kind="relevancy",
                query=query,
                caption=f"Dry-run LERF relevancy for '{query}'",
                camera_id="view_0000",
                width=512,
                height=384,
                score=1.0,
            ),
            RenderedView(
                path=str(overlay_path),
                kind="overlay",
                query=query,
                caption=f"Dry-run overlay for '{query}'",
                camera_id="view_0000",
                width=1536,
                height=428,
            ),
        ]

    def _write_viewer_fallback(self, query: str, output: Path) -> list[RenderedView]:
        fallback = output / "interactive_viewer_workflow.md"
        fallback.write_text(
            "\n".join(
                [
                    "# LERF Interactive Query Fallback",
                    "",
                    "Automated rendering was not available for this installed LERF/Nerfstudio version.",
                    "",
                    "Run:",
                    "",
                    f"```bash\n{self.viewer_command_text()}\n```",
                    "",
                    "Then in the Nerfstudio viewer:",
                    "",
                    f"1. Enter the text prompt: `{query}`",
                    "2. Select `relevancy_0` or `composited_0` as the output type.",
                    "3. Save screenshots or camera-path renders into this output directory.",
                    "",
                    "This fallback is expected for some upstream revisions because LERF documents",
                    "viewer prompt entry and notes that command-line prompt rendering is not a",
                    "stable upstream feature.",
                ]
            ),
            encoding="utf-8",
        )
        return [
            RenderedView(
                path=str(fallback),
                kind="viewer_fallback",
                query=query,
                caption="Interactive LERF viewer workflow",
            )
        ]

    def _render_with_lerf_internal_api(self, query: str, output: Path) -> list[RenderedView]:
        """Best-effort internal LERF renderer.

        This path follows current LERF behavior: the model owns an image_encoder
        with set_positives, and evaluation outputs include rgb, relevancy_0, and
        composited_0. Nerfstudio internals can change, so callers handle failure.
        """

        if not self.config_path:
            raise RuntimeError("Config path not loaded.")

        import torch  # type: ignore
        from nerfstudio.utils.eval_utils import eval_setup  # type: ignore

        loaded = eval_setup(Path(self.config_path), test_mode="inference")
        pipeline = _extract_pipeline(loaded)
        model = getattr(pipeline, "model", None)
        if model is None:
            raise RuntimeError("Could not find model on loaded Nerfstudio pipeline.")
        model = getattr(model, "module", model)
        image_encoder = getattr(model, "image_encoder", None)
        if image_encoder is None or not hasattr(image_encoder, "set_positives"):
            raise RuntimeError(
                "Loaded model does not expose image_encoder.set_positives; "
                "this does not look like a compatible LERF model."
            )
        image_encoder.set_positives([query])
        if hasattr(model, "eval"):
            model.eval()

        camera = _first_eval_camera(pipeline)
        if hasattr(camera, "to"):
            device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
            camera = camera.to(device)

        with torch.no_grad():
            if hasattr(model, "get_outputs_for_camera"):
                outputs = model.get_outputs_for_camera(camera)
            else:
                ray_bundle = camera.generate_rays(camera_indices=0)
                outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle)

        rgb_path = output / "view_0000_rgb.png"
        relevancy_path = output / "view_0000_relevancy.png"
        composited_path = output / "view_0000_composited.png"
        overlay_path = output / "view_0000_overlay.png"

        _save_tensor_image(outputs["rgb"], rgb_path)
        if "relevancy_0" not in outputs:
            raise RuntimeError("LERF outputs did not contain relevancy_0.")
        _save_tensor_image(outputs["relevancy_0"], relevancy_path)
        if "composited_0" in outputs:
            _save_tensor_image(outputs["composited_0"], composited_path)
        else:
            create_side_by_side_overlay(rgb_path, relevancy_path, composited_path, query=query)
        create_side_by_side_overlay(rgb_path, relevancy_path, overlay_path, query=query)

        width, height = Image.open(rgb_path).size
        overlay_width, overlay_height = Image.open(overlay_path).size
        return [
            RenderedView(str(rgb_path), "rgb", query, "RGB render", "view_0000", width, height),
            RenderedView(
                str(relevancy_path),
                "relevancy",
                query,
                "LERF relevancy_0 render",
                "view_0000",
                width,
                height,
            ),
            RenderedView(
                str(composited_path),
                "composited",
                query,
                "LERF composited_0 render",
                "view_0000",
                width,
                height,
            ),
            RenderedView(
                str(overlay_path),
                "overlay",
                query,
                "RGB plus relevancy overlay",
                "view_0000",
                overlay_width,
                overlay_height,
            ),
        ]


def _extract_pipeline(eval_setup_result: Any) -> Any:
    if hasattr(eval_setup_result, "model") or hasattr(eval_setup_result, "datamanager"):
        return eval_setup_result
    if isinstance(eval_setup_result, tuple):
        for item in eval_setup_result:
            if hasattr(item, "model") and hasattr(item, "datamanager"):
                return item
    raise RuntimeError("Could not extract Nerfstudio pipeline from eval_setup result.")


def _first_eval_camera(pipeline: Any) -> Any:
    datamanager = getattr(pipeline, "datamanager", None)
    if datamanager is None:
        raise RuntimeError("Loaded pipeline does not expose datamanager.")
    dataset = getattr(datamanager, "eval_dataset", None) or getattr(datamanager, "train_dataset", None)
    cameras = getattr(dataset, "cameras", None) if dataset is not None else None
    if cameras is None:
        loader = getattr(datamanager, "fixed_indices_eval_dataloader", None)
        if loader is not None and hasattr(loader, "cameras"):
            cameras = loader.cameras
    if cameras is None:
        raise RuntimeError("Could not find eval cameras in the loaded pipeline.")
    try:
        return cameras[0:1]
    except Exception:
        return cameras[0]


def _save_tensor_image(tensor: Any, path: Path) -> None:
    array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    array = np.squeeze(array)
    if array.ndim == 1:
        side = int(np.sqrt(array.shape[0]))
        array = array[: side * side].reshape(side, side)
    if array.ndim == 2:
        array = np.clip(array, 0.0, 1.0)
        image = Image.fromarray((array * 255).astype(np.uint8), mode="L").convert("RGB")
    elif array.ndim == 3:
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = np.clip(array[..., :3], 0.0, 1.0)
        image = Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported tensor image shape: {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_query_report_template(query: str, config_path: str, output_dir: str | Path) -> Path:
    """Write a structured report template for manual LERF viewer outputs."""

    path = Path(output_dir) / f"{slugify(query)}_manual_report_template.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "backend_name": "lerf",
        "config_path": config_path,
        "rendered_images": [],
        "candidate_points": [],
        "bounding_regions": [],
        "confidence": None,
        "warnings": ["Manual viewer output template."],
        "provenance": {"timestamp": utc_timestamp(), "manual": True},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
