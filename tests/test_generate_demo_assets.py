import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generate_demo_assets_planned_mode_writes_scene_reports(tmp_path: Path) -> None:
    queries = tmp_path / "queries.yaml"
    queries.write_text("scene_name: desk_scene\nqueries:\n  - object next to the mug\n", encoding="utf-8")
    output = tmp_path / "demo_assets"
    report = tmp_path / "project_report.md"
    card = tmp_path / "portfolio_card.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_demo_assets.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--queries",
            str(queries),
            "--output",
            str(output),
            "--report-output",
            str(report),
            "--portfolio-card-output",
            str(card),
            "--planner-mode",
            "planned",
            "--max-queries",
            "3",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "demo_summary.json").read_text(encoding="utf-8"))
    assert summary["planner_mode"] == "planned"
    assert summary["num_user_queries"] == 1
    assert summary["num_backend_results"] == 2
    scene_report = output / "object_next_to_the_mug" / "scene_query_report.json"
    assert summary["scene_reports"] == [str(scene_report)]
    payload = json.loads(scene_report.read_text(encoding="utf-8"))
    assert payload["plan"]["relation_anchors"][0]["anchor_query"] == "mug"
    query_result = json.loads(
        (output / "object_next_to_the_mug" / "mug" / "query_result.json").read_text(encoding="utf-8")
    )
    call = query_result["provenance"]["planner_backend_call"]
    assert call["purpose"] == "relation_anchor"
    assert call["metadata"]["relation"] == "near relation"
    assert "## Scene Relation Analysis" in report.read_text(encoding="utf-8")
    assert "- Demo planner mode: `planned`" in card.read_text(encoding="utf-8")


def test_generate_demo_assets_direct_mode_keeps_flat_query_result(tmp_path: Path) -> None:
    queries = tmp_path / "queries.yaml"
    queries.write_text("scene_name: desk_scene\nqueries:\n  - mug\n", encoding="utf-8")
    output = tmp_path / "demo_assets"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_demo_assets.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--queries",
            str(queries),
            "--output",
            str(output),
            "--report-output",
            str(tmp_path / "project_report.md"),
            "--portfolio-card-output",
            str(tmp_path / "portfolio_card.md"),
            "--planner-mode",
            "direct",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "demo_summary.json").read_text(encoding="utf-8"))
    assert summary["planner_mode"] == "direct"
    assert summary["num_user_queries"] == 1
    assert summary["num_backend_results"] == 1
    assert summary["scene_reports"] == []
    assert (output / "mug" / "query_result.json").exists()
    assert not (output / "mug" / "scene_query_report.json").exists()
