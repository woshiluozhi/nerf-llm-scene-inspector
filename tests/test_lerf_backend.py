from pathlib import Path

import pytest

from nerf_llm_scene_inspector.backends.base import QueryResult
from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend


def test_lerf_dry_run_multiview_outputs(tmp_path: Path) -> None:
    backend = LERFBackend(dry_run=True, num_views=3)
    backend.load(str(tmp_path / "config.yml"))
    result = backend.query_text("mug", str(tmp_path / "query"), top_k=5)

    assert len([view for view in result.rendered_images if view.kind == "rgb"]) == 3
    assert len([view for view in result.rendered_images if view.kind == "relevancy"]) == 3
    assert len(result.bounding_regions) == 3
    assert (tmp_path / "query" / "query_result.json").exists()
    loaded = QueryResult.from_json(tmp_path / "query" / "query_result.json")
    assert loaded.provenance["num_views"] == 3


def test_lerf_fallback_artifacts(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("method_name: lerf\n", encoding="utf-8")
    backend = LERFBackend(dry_run=False, save_manual_template=True)
    backend.load(str(config))

    def fail_render(_query: str, _output: Path):
        raise RuntimeError("mock upstream failure")

    monkeypatch.setattr(backend, "_render_with_lerf_internal_api", fail_render)
    result = backend.query_text("mug", str(tmp_path / "query"))

    assert any(view.kind == "viewer_fallback" for view in result.rendered_images)
    assert any(view.kind == "manual_template" for view in result.rendered_images)
    assert result.warnings
    assert (tmp_path / "query" / "interactive_viewer_workflow.md").exists()


def test_lerf_strict_backend_raises(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("method_name: lerf\n", encoding="utf-8")
    backend = LERFBackend(dry_run=False, strict_backend=True)
    backend.load(str(config))

    def fail_render(_query: str, _output: Path):
        raise RuntimeError("mock upstream failure")

    monkeypatch.setattr(backend, "_render_with_lerf_internal_api", fail_render)
    with pytest.raises(RuntimeError, match="strict-backend"):
        backend.query_text("mug", str(tmp_path / "query"))
