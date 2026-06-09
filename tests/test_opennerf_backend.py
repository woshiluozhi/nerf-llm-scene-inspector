from pathlib import Path

import pytest

from nerf_llm_scene_inspector.backends.base import QueryResult
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend


def test_opennerf_dry_run_multiview_outputs(tmp_path: Path) -> None:
    backend = OpenNeRFBackend(dry_run=True, num_views=2)
    backend.load(str(tmp_path / "config.yml"))
    result = backend.query_text("mug", str(tmp_path / "query"), top_k=5)

    assert len([view for view in result.rendered_images if view.kind == "rgb"]) == 2
    assert len([view for view in result.rendered_images if view.kind == "relevancy"]) == 2
    assert len(result.bounding_regions) == 2
    assert result.provenance["num_views"] == 2
    loaded = QueryResult.from_json(tmp_path / "query" / "query_result.json")
    assert loaded.backend_name == "opennerf"
    assert loaded.rendered_images[0].camera_id == "view_0000"


def test_opennerf_fallback_writes_repair_workflow_and_template(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("method_name: opennerf\n", encoding="utf-8")
    backend = OpenNeRFBackend(dry_run=False, save_manual_template=True)
    backend.load(str(config))

    result = backend.query_text("mug", str(tmp_path / "query"))

    assert any(view.kind == "viewer_fallback" for view in result.rendered_images)
    assert any(view.kind == "manual_template" for view in result.rendered_images)
    assert result.warnings
    workflow = (tmp_path / "query" / "opennerf_viewer_workflow.md").read_text(encoding="utf-8")
    assert "repair_scene_query_from_viewer.py" in workflow
    assert (tmp_path / "query" / "mug_opennerf_manual_report_template.json").exists()


def test_opennerf_fallback_respects_manual_template_flag(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("method_name: opennerf\n", encoding="utf-8")
    backend = OpenNeRFBackend(dry_run=False, save_manual_template=False)
    backend.load(str(config))

    result = backend.query_text("mug", str(tmp_path / "query"))

    assert any(view.kind == "viewer_fallback" for view in result.rendered_images)
    assert not any(view.kind == "manual_template" for view in result.rendered_images)
    assert not (tmp_path / "query" / "mug_opennerf_manual_report_template.json").exists()


def test_opennerf_strict_backend_raises(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text("method_name: opennerf\n", encoding="utf-8")
    backend = OpenNeRFBackend(dry_run=False, strict_backend=True)
    backend.load(str(config))

    with pytest.raises(RuntimeError, match="Automated OpenNeRF rendering failed in --strict-backend mode"):
        backend.query_text("mug", str(tmp_path / "query"))
