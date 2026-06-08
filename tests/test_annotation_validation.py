import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult, RenderedView
from nerf_llm_scene_inspector.evaluation.annotation_validation import validate_annotations


ROOT = Path(__file__).resolve().parents[1]


def test_validate_annotations_reports_warnings_without_failing(tmp_path: Path) -> None:
    queries = tmp_path / "queries.yaml"
    annotations = tmp_path / "annotations.json"
    results = tmp_path / "results"
    queries.write_text("scene_name: desk_scene\nqueries:\n  - mug\n  - laptop\n", encoding="utf-8")
    annotations.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "white mug",
                        "acceptable_views": ["view_0000"],
                        "bbox_2d": [0, 0, 20, 20],
                    },
                    {"query": "extra", "target_description": "extra label", "bbox_2d": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_result(results / "mug" / "query_result.json")

    report = validate_annotations(annotations, queries_path=queries, results_dir=results)

    assert report.ok is True
    assert report.missing_annotations == ["laptop"]
    assert report.extra_annotations == ["extra"]
    assert report.missing_result_queries == ["extra"]
    assert report.bbox_annotation_count == 1
    assert report.warnings


def test_validate_annotations_fails_duplicate_or_invalid_bbox(tmp_path: Path) -> None:
    annotations = tmp_path / "bad_annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {"query": "mug", "bbox_2d": [20, 20, 10, 30]},
                    {"query": "mug", "bbox_2d": [0, 0, 10, 10]},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_annotations(annotations)

    assert report.ok is False
    assert report.duplicate_annotations == ["mug"]
    assert report.invalid_bboxes[0]["query"] == "mug"


def test_validate_annotations_cli_writes_report(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "validation.json"
    annotations.write_text(
        json.dumps({"scene_name": "desk_scene", "queries": [{"query": "mug", "bbox_2d": [0, 0, 10, 10]}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_annotations.py"),
            "--annotations",
            str(annotations),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_validate_annotations_reports_non_object_json(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    report = validate_annotations(annotations)

    assert report.ok is False
    assert report.errors == ["Annotation JSON must be an object with scene_name and queries fields."]


def _write_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    QueryResult(
        query="mug",
        backend_name="dry-run",
        config_path="config.yml",
        rendered_images=[
            RenderedView(
                path="view_0000_overlay.png",
                kind="overlay",
                query="mug",
                camera_id="view_0000",
            )
        ],
        bounding_regions=[
            BoundingRegion(label="mug", bbox_2d=(0, 0, 20, 20), source_view="view_0000")
        ],
    ).to_json(path)
