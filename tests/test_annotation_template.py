import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_create_annotation_template_from_queries_and_results(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.yaml"
    queries_path.write_text(
        "\n".join(
            [
                "scene_name: test_scene",
                "queries:",
                "  - mug",
                "  - laptop",
                "tasks:",
                "  - Find containers.",
            ]
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "results" / "mug" / "query_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "query": "mug",
                "backend_name": "dry-run",
                "config_path": "config.yml",
                "rendered_images": [
                    {"path": "view_0000_rgb.png", "kind": "rgb", "query": "mug", "camera_id": "view_0000"}
                ],
                "candidate_points": [],
                "bounding_regions": [
                    {
                        "label": "mug",
                        "score": 0.8,
                        "coordinate_frame": "image",
                        "bbox_2d": [10, 20, 100, 120],
                        "source_view": "view_0000",
                    }
                ],
                "confidence": 0.8,
                "warnings": [],
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "annotations_template.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_annotation_template.py"),
            "--queries",
            str(queries_path),
            "--results",
            str(tmp_path / "results"),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["scene_name"] == "test_scene"
    assert [item["query"] for item in payload["queries"]] == ["mug", "laptop"]
    mug = payload["queries"][0]
    assert mug["bbox_2d"] is None
    assert mug["acceptable_views"] == ["view_0000"]
    assert mug["candidate_bbox_2d_suggestions"][0]["bbox_2d"] == [10.0, 20.0, 100.0, 120.0]
