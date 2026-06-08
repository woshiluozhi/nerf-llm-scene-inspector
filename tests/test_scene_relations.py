import csv
import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, Candidate3DPoint, QueryResult
from nerf_llm_scene_inspector.evaluation.scene_relations import analyze_scene_relations


ROOT = Path(__file__).resolve().parents[1]


def test_scene_relation_analysis_finds_2d_support(tmp_path: Path) -> None:
    results = tmp_path / "results"
    QueryResult(
        query="desk",
        backend_name="dry-run",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(
                label="desk",
                score=0.9,
                bbox_2d=(50, 220, 420, 360),
                source_view="view_0000",
            )
        ],
    ).to_json(results / "desk" / "query_result.json")
    QueryResult(
        query="mug",
        backend_name="dry-run",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(
                label="mug",
                score=0.8,
                bbox_2d=(110, 150, 170, 220),
                source_view="view_0000",
            )
        ],
    ).to_json(results / "mug" / "query_result.json")

    report = analyze_scene_relations(
        results_dir=results,
        output_dir=tmp_path / "relations",
        scene_name="unit_scene",
    )

    assert report.num_entities == 2
    assert any(
        relation.subject_label == "desk"
        and relation.object_label == "mug"
        and relation.relation == "likely_supports"
        for relation in report.relations
    )
    assert (tmp_path / "relations" / "scene_relations_summary.json").exists()
    assert (tmp_path / "relations" / "scene_relations_report.md").exists()
    rows = list(csv.DictReader((tmp_path / "relations" / "scene_relations_edges.csv").open()))
    assert any(row["relation"] == "likely_supports" for row in rows)


def test_scene_relation_analysis_finds_3d_on_top_relation(tmp_path: Path) -> None:
    results = tmp_path / "results"
    QueryResult(
        query="table",
        backend_name="dry-run",
        config_path="config.yml",
        candidate_points=[Candidate3DPoint(label="table", x=0.0, y=0.0, z=0.0, score=0.7)],
    ).to_json(results / "table" / "query_result.json")
    QueryResult(
        query="cup",
        backend_name="dry-run",
        config_path="config.yml",
        candidate_points=[Candidate3DPoint(label="cup", x=0.05, y=0.02, z=0.12, score=0.8)],
    ).to_json(results / "cup" / "query_result.json")

    report = analyze_scene_relations(results_dir=results, scene_name="unit_scene")

    assert any(relation.evidence_type == "3d" for relation in report.relations)
    assert any(relation.relation == "on_top_of_or_supported_by" for relation in report.relations)


def test_analyze_scene_relations_cli_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "relations"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_scene_relations.py"),
            "--results",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--scene-name",
            "dry_scene",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "scene_relations_summary.json").read_text(encoding="utf-8"))
    assert summary["scene_name"] == "dry_scene"
    assert summary["num_entities"] == 3
    assert summary["num_relations"] > 0
