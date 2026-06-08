import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from nerf_llm_scene_inspector.backends.base import QueryResult
from nerf_llm_scene_inspector.backends.base import RenderedView, SceneQueryReport
from nerf_llm_scene_inspector.querying.viewer_import import import_viewer_outputs
from nerf_llm_scene_inspector.querying.viewer_import import repair_scene_query_report_from_viewer_outputs
from nerf_llm_scene_inspector.visualization.render_overlays import create_mock_rgb_and_heatmap


ROOT = Path(__file__).resolve().parents[1]


def test_import_viewer_outputs_writes_query_result(tmp_path: Path) -> None:
    source = tmp_path / "viewer"
    output = tmp_path / "query"
    create_mock_rgb_and_heatmap(source, query="mug", view_id="view_0003")
    (tmp_path / "config.yml").write_text("method_name: lerf\n", encoding="utf-8")

    result, summary = import_viewer_outputs(
        query="mug",
        config_path=tmp_path / "config.yml",
        input_dir=source,
        output_dir=output,
    )

    assert result.query == "mug"
    assert result.confidence is not None
    assert len(result.bounding_regions) == 1
    assert result.bounding_regions[0].source_view == "view_0003"
    assert (output / "query_result.json").exists()
    assert (output / "viewer_import_summary.json").exists()
    assert summary.imported_files
    loaded = QueryResult.from_json(output / "query_result.json")
    assert loaded.rendered_images[0].camera_id == "view_0003"


def test_import_viewer_outputs_generates_missing_overlay(tmp_path: Path) -> None:
    source = tmp_path / "viewer"
    source.mkdir()
    rgb_path, heatmap_path, overlay_path = create_mock_rgb_and_heatmap(
        source,
        query="mug",
        view_id="view_0001",
    )
    overlay_path.unlink()

    result, summary = import_viewer_outputs(
        query="mug",
        config_path="config.yml",
        input_dir=source,
        output_dir=tmp_path / "query",
    )

    assert any(view.kind == "overlay" for view in result.rendered_images)
    assert summary.generated_files
    assert Path(summary.generated_files[0]).name == "view_0001_overlay.png"
    assert rgb_path.exists()
    assert heatmap_path.exists()


def test_import_viewer_outputs_warns_without_heatmap(tmp_path: Path) -> None:
    source = tmp_path / "viewer"
    source.mkdir()
    Image.new("RGB", (64, 64), color=(40, 80, 120)).save(source / "view_0000_rgb.png")

    result, _summary = import_viewer_outputs(
        query="mug",
        config_path="config.yml",
        input_dir=source,
        output_dir=tmp_path / "query",
    )

    assert result.bounding_regions == []
    assert any("No relevancy heatmap" in warning for warning in result.warnings)


def test_import_viewer_outputs_cli(tmp_path: Path) -> None:
    source = tmp_path / "viewer"
    output = tmp_path / "query"
    create_mock_rgb_and_heatmap(source, query="mug", view_id="view_0000")
    config = tmp_path / "config.yml"
    config.write_text("method_name: lerf\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_viewer_outputs.py"),
            "--query",
            "mug",
            "--config",
            str(config),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "query_result.json").read_text(encoding="utf-8"))
    assert payload["query"] == "mug"
    assert payload["provenance"]["manual_viewer_import"] is True


def test_repair_scene_query_report_from_viewer_outputs(tmp_path: Path) -> None:
    report_path = _write_scene_query_report(tmp_path / "scene_query_report.json")
    viewer_root = tmp_path / "manual_viewer"
    create_mock_rgb_and_heatmap(viewer_root / "mug", query="mug", view_id="view_0002")

    report, summary = repair_scene_query_report_from_viewer_outputs(
        report_path=report_path,
        viewer_root=viewer_root,
    )

    assert summary.ok
    assert summary.repaired_queries == ["mug"]
    assert summary.kept_queries == ["bottle"]
    assert (tmp_path / "mug" / "query_result.json").exists()
    assert (tmp_path / "viewer_repair_summary.json").exists()
    assert (tmp_path / "scene_query_report.md").exists()
    repaired_result = report.query_results[0]
    assert repaired_result.query == "mug"
    assert repaired_result.bounding_regions
    assert repaired_result.provenance["manual_viewer_import"] is True
    assert not any(view.kind == "viewer_fallback" for view in repaired_result.rendered_images)
    assert report.answer_summary["support_level"] == "2d_relevancy_fallback"


def test_repair_scene_query_report_cli_requires_all(tmp_path: Path) -> None:
    report_path = _write_scene_query_report(tmp_path / "scene_query_report.json")
    viewer_root = tmp_path / "manual_viewer"
    create_mock_rgb_and_heatmap(viewer_root / "mug", query="mug", view_id="view_0002")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "repair_scene_query_from_viewer.py"),
            "--report",
            str(report_path),
            "--viewer-root",
            str(viewer_root),
            "--require-all",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    summary = json.loads((tmp_path / "viewer_repair_summary.json").read_text(encoding="utf-8"))
    assert summary["repaired_queries"] == ["mug"]
    assert summary["missing_required_queries"] == ["bottle"]


def test_repair_scene_query_report_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "repair_scene_query_from_viewer.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--viewer-root" in result.stdout


def _write_scene_query_report(path: Path) -> Path:
    report = SceneQueryReport(
        scene_name="desk_scene",
        task="Find objects that can hold water.",
        plan={
            "planner_name": "local_rules",
            "final_answer_template": "Likely relevant scene regions are {items}.",
            "primary_visual_queries": ["mug", "bottle"],
        },
        query_results=[
            QueryResult(
                query="mug",
                backend_name="lerf",
                config_path="config.yml",
                rendered_images=[
                    RenderedView(
                        path="interactive_viewer_workflow.md",
                        kind="viewer_fallback",
                        query="mug",
                    )
                ],
                warnings=["Automated LERF rendering failed; wrote viewer fallback instructions."],
            ),
            QueryResult(
                query="bottle",
                backend_name="lerf",
                config_path="config.yml",
                confidence=0.2,
            ),
        ],
        answer="Previous fallback answer.",
        warnings=["Automated LERF rendering failed; wrote viewer fallback instructions."],
    )
    return report.to_json(path)
