import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult, RenderedView
from nerf_llm_scene_inspector.evaluation.annotation_workbench import build_annotation_workbench


ROOT = Path(__file__).resolve().parents[1]


def test_build_annotation_workbench_copies_images_and_prefills_bbox(tmp_path: Path) -> None:
    annotations = _write_template(tmp_path / "annotations.json")
    results = tmp_path / "results"
    _write_query_result(results / "mug")
    output = tmp_path / "workbench"

    workbench = build_annotation_workbench(
        annotations_path=annotations,
        results_dir=results,
        output_dir=output,
    )

    assert workbench.scene_name == "desk_scene"
    assert workbench.item_count == 1
    assert workbench.image_count == 1
    item = workbench.items[0]
    assert item.query == "mug"
    assert item.bbox_2d == [10.0, 12.0, 64.0, 70.0]
    assert item.image_width == 96
    assert item.image_height == 80
    assert (output / item.image_path).exists()
    assert (output / "annotation_workbench.html").exists()
    assert (output / "annotation_workbench_manifest.json").exists()
    assert (output / "annotation_seed.json").exists()
    assert "Download JSON" in (output / "annotation_workbench.html").read_text(encoding="utf-8")
    seed = json.loads((output / "annotation_seed.json").read_text(encoding="utf-8"))
    assert seed["queries"][0]["bbox_2d"] == [10.0, 12.0, 64.0, 70.0]


def test_create_annotation_workbench_cli(tmp_path: Path) -> None:
    annotations = _write_template(tmp_path / "annotations.json")
    results = tmp_path / "results"
    _write_query_result(results / "mug")
    output = tmp_path / "workbench"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_annotation_workbench.py"),
            "--annotations",
            str(annotations),
            "--results",
            str(results),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "annotation_workbench_manifest.json").read_text(encoding="utf-8"))
    assert manifest["item_count"] == 1
    assert manifest["image_count"] == 1


def _write_template(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "white mug",
                        "acceptable_views": ["view_0000"],
                        "bbox_2d": None,
                        "notes": "candidate",
                        "candidate_views": ["view_0000"],
                        "candidate_bbox_2d_suggestions": [
                            {
                                "source_view": "view_0000",
                                "bbox_2d": [10, 12, 64, 70],
                                "score": 0.8,
                                "notes": "suggested",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_query_result(query_dir: Path) -> None:
    query_dir.mkdir(parents=True)
    Image.new("RGB", (96, 80), (230, 235, 240)).save(query_dir / "view_0000_rgb.png")
    QueryResult(
        query="mug",
        backend_name="dry-run",
        config_path="config.yml",
        rendered_images=[
            RenderedView(
                path="view_0000_rgb.png",
                kind="rgb",
                query="mug",
                camera_id="view_0000",
                width=96,
                height=80,
            )
        ],
        bounding_regions=[
            BoundingRegion(
                label="mug",
                score=0.8,
                bbox_2d=(10.0, 12.0, 64.0, 70.0),
                source_view="view_0000",
            )
        ],
    ).to_json(query_dir / "query_result.json")
