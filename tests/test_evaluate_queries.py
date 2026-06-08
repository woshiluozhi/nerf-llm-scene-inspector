import csv
import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult


ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_queries_counts_only_bbox_annotated_rows(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.yaml"
    annotations_path = tmp_path / "annotations.json"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "evaluation"
    report_path = tmp_path / "project_report.md"

    queries_path.write_text(
        "scene_name: desk_scene\nqueries:\n  - mug\n  - container\n",
        encoding="utf-8",
    )
    annotations_path.write_text(
        json.dumps(
            {
                "scene_name": "desk_scene",
                "queries": [
                    {
                        "query": "mug",
                        "target_description": "white mug",
                        "acceptable_views": ["view_0000"],
                        "bbox_2d": [0, 0, 25, 25],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_result(
        results_dir / "mug" / "query_result.json",
        "mug",
        BoundingRegion(label="mug", score=0.9, bbox_2d=(0, 0, 20, 20), source_view="view_0000"),
    )
    _write_result(
        results_dir / "container" / "query_result.json",
        "container",
        BoundingRegion(label="container", score=0.5, bbox_2d=(40, 40, 80, 80), source_view="view_0000"),
    )
    _write_result(
        results_dir / "container_task" / "mug" / "query_result.json",
        "mug",
        BoundingRegion(label="mug", score=0.4, bbox_2d=(100, 100, 120, 120), source_view="view_0000"),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_queries.py"),
            "--queries",
            str(queries_path),
            "--annotations",
            str(annotations_path),
            "--results",
            str(results_dir),
            "--output",
            str(output_dir),
            "--report-output",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    validation = json.loads((output_dir / "annotation_validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True
    assert validation["missing_annotations"] == ["container"]
    summary = json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["num_result_queries"] == 3
    assert summary["num_unique_result_queries"] == 2
    assert summary["num_evaluated_queries"] == 1
    assert summary["num_qualitative_only_queries"] == 1
    assert summary["top_k_hit_rate"] == 1.0

    rows = list(csv.DictReader((output_dir / "eval_table.csv").open(encoding="utf-8")))
    status_by_query = {row["query"]: row["evaluation_status"] for row in rows}
    assert status_by_query == {"container": "unannotated", "mug": "evaluated"}
    assert "unannotated" in report_path.read_text(encoding="utf-8")


def _write_result(path: Path, query: str, region: BoundingRegion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    QueryResult(
        query=query,
        backend_name="dry-run",
        config_path="config.yml",
        bounding_regions=[region],
        confidence=region.score,
    ).to_json(path)
