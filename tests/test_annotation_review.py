import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from nerf_llm_scene_inspector.backends.base import QueryResult, RenderedView
from nerf_llm_scene_inspector.evaluation.annotation_review import build_annotation_review


ROOT = Path(__file__).resolve().parents[1]


def test_build_annotation_review_draws_bbox_and_contact_sheet(tmp_path: Path) -> None:
    annotations = _write_annotations(
        tmp_path / "annotations.json",
        acceptable_views=["view_0000"],
        bbox=[10, 12, 64, 70],
    )
    results = tmp_path / "results"
    _write_query_result(results / "mug", image_name="view_0000_rgb.png", camera_id="view_0000")

    report = build_annotation_review(
        annotations_path=annotations,
        results_dir=results,
        output_dir=tmp_path / "review",
    )

    assert report.ok is True
    assert report.reviewed_annotations == 1
    assert report.items[0].status == "ready"
    assert (tmp_path / "review" / report.items[0].review_image).exists()
    assert (tmp_path / "review" / report.contact_sheet).exists()


def test_review_annotations_cli_allows_view_fallback_warning(tmp_path: Path) -> None:
    annotations = _write_annotations(
        tmp_path / "annotations.json",
        acceptable_views=["view_9999"],
        bbox=[10, 12, 64, 70],
    )
    results = tmp_path / "results"
    _write_query_result(results / "mug", image_name="view_0000_rgb.png", camera_id="view_0000")
    output = tmp_path / "review"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "review_annotations.py"),
            "--annotations",
            str(annotations),
            "--results",
            str(results),
            "--output",
            str(output),
            "--allow-warnings",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "annotation_review.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["items"][0]["status"] == "view_fallback"
    assert "fallback view" in payload["items"][0]["warnings"][0]
    assert (output / "annotation_review.md").exists()


def _write_annotations(path: Path, *, acceptable_views: list[str], bbox: list[int]) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "white mug",
                        "acceptable_views": acceptable_views,
                        "bbox_2d": bbox,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_query_result(query_dir: Path, *, image_name: str, camera_id: str) -> None:
    query_dir.mkdir(parents=True)
    Image.new("RGB", (96, 80), (230, 235, 240)).save(query_dir / image_name)
    QueryResult(
        query="mug",
        backend_name="dry-run",
        config_path="config.yml",
        rendered_images=[
            RenderedView(
                path=image_name,
                kind="rgb",
                query="mug",
                camera_id=camera_id,
                width=96,
                height=80,
            )
        ],
    ).to_json(query_dir / "query_result.json")
