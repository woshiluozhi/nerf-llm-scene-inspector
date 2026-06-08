import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult, RenderedView
from nerf_llm_scene_inspector.evaluation.annotation_merge import merge_workbench_annotations


ROOT = Path(__file__).resolve().parents[1]


def test_merge_workbench_annotations_outputs_clean_schema(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "annotation_template.json")
    filled = _write_filled(tmp_path / "annotations_filled.json")
    output = tmp_path / "annotations_merged.json"
    report_path = tmp_path / "merge_report.json"

    report = merge_workbench_annotations(
        template_path=template,
        filled_path=filled,
        output_path=output,
        report_path=report_path,
    )

    assert report.ok is True
    assert report.updated_count == 1
    assert report.bbox_annotation_count == 1
    assert report.missing_filled_queries == ["laptop"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scene_name"] == "desk_scene"
    assert payload["queries"][0] == {
        "query": "mug",
        "target_description": "white mug on desk",
        "acceptable_views": ["view_0000"],
        "bbox_2d": [8.0, 9.0, 70.0, 75.0],
        "notes": "verified in workbench",
    }
    assert "candidate_bbox_2d_suggestions" not in payload["queries"][0]
    assert (tmp_path / "merge_report.json").exists()


def test_merge_annotation_workbench_cli_with_validation(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "annotation_template.json")
    filled = _write_filled(tmp_path / "annotations_filled.json")
    queries = tmp_path / "queries.yaml"
    queries.write_text("scene_name: desk_scene\nqueries:\n  - mug\n  - laptop\n", encoding="utf-8")
    results = tmp_path / "results"
    _write_query_result(results / "mug", query="mug", view="view_0000")
    _write_query_result(results / "laptop", query="laptop", view="view_0001")
    output = tmp_path / "annotations_merged.json"
    report_path = tmp_path / "merge_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "merge_annotation_workbench.py"),
            "--template",
            str(template),
            "--filled",
            str(filled),
            "--output",
            str(output),
            "--report-output",
            str(report_path),
            "--queries",
            str(queries),
            "--results",
            str(results),
            "--overwrite",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["validation"]["ok"] is True
    assert report["validation"]["bbox_annotation_count"] == 1


def test_merge_workbench_annotations_reports_invalid_filled_bbox(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "annotation_template.json")
    filled = tmp_path / "bad_filled.json"
    filled.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "mug",
                        "acceptable_views": ["view_0000"],
                        "bbox_2d": [70, 9, 8, 75],
                        "notes": "bad box",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = merge_workbench_annotations(
        template_path=template,
        filled_path=filled,
        output_path=tmp_path / "merged.json",
    )

    assert report.ok is False
    assert report.invalid_bboxes[0]["query"] == "mug"
    assert "bbox_2d" in report.errors[0]


def _write_template(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "",
                        "acceptable_views": ["view_0000"],
                        "bbox_2d": None,
                        "notes": "candidate only",
                        "candidate_views": ["view_0000"],
                        "candidate_bbox_2d_suggestions": [
                            {
                                "source_view": "view_0000",
                                "bbox_2d": [10, 12, 64, 70],
                                "score": 0.8,
                                "notes": "suggested",
                            }
                        ],
                    },
                    {
                        "query": "laptop",
                        "target_description": "open laptop",
                        "acceptable_views": ["view_0001"],
                        "bbox_2d": None,
                        "notes": "qualitative",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_filled(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "white mug on desk",
                        "acceptable_views": ["view_0000"],
                        "bbox_2d": [8, 9, 70, 75],
                        "notes": "verified in workbench",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_query_result(query_dir: Path, *, query: str, view: str) -> None:
    query_dir.mkdir(parents=True)
    QueryResult(
        query=query,
        backend_name="dry-run",
        config_path="config.yml",
        rendered_images=[
            RenderedView(
                path=f"{view}_rgb.png",
                kind="rgb",
                query=query,
                camera_id=view,
            )
        ],
        bounding_regions=[
            BoundingRegion(
                label=query,
                score=0.8,
                bbox_2d=(1.0, 2.0, 20.0, 30.0),
                source_view=view,
            )
        ],
    ).to_json(query_dir / "query_result.json")
