import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_query_scene_cli_writes_answer_summary_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "query"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--backend",
            "lerf",
            "--query",
            "mug",
            "--output",
            str(output),
            "--scene-name",
            "desk_scene",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "scene_query_report.json").read_text(encoding="utf-8"))
    assert report["scene_name"] == "desk_scene"
    assert report["answer_summary"]["support_level"] == "2d_relevancy_fallback"
    assert (output / "scene_query_report.md").exists()
    assert (output / "query_grid.png").exists()
    visual_summary = json.loads((output / "query_visual_summary.json").read_text(encoding="utf-8"))
    assert visual_summary["query_grid"] == "query_grid.png"
    assert visual_summary["expanded_queries"] == ["mug"]
    assert "Wrote markdown report" in result.stdout
    assert "Wrote query grid" in result.stdout


def test_query_scene_cli_expands_high_level_task_with_max_queries(tmp_path: Path) -> None:
    output = tmp_path / "query"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--backend",
            "lerf",
            "--query",
            "Find objects that can hold water.",
            "--output",
            str(output),
            "--scene-name",
            "desk_scene",
            "--dry-run",
            "--max-queries",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "scene_query_report.json").read_text(encoding="utf-8"))
    queries = [item["query"] for item in report["query_results"]]
    assert queries == ["cup", "mug"]
    assert (output / "cup").exists()
    assert (output / "mug").exists()
    visual_summary = json.loads((output / "query_visual_summary.json").read_text(encoding="utf-8"))
    assert visual_summary["expanded_queries"] == ["cup", "mug"]
    assert visual_summary["num_overlay_images"] == 2


def test_query_scene_cli_records_negative_query_purpose_without_positive_evidence(tmp_path: Path) -> None:
    output = tmp_path / "query"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--backend",
            "lerf",
            "--query",
            "Find objects that can hold water.",
            "--output",
            str(output),
            "--scene-name",
            "desk_scene",
            "--dry-run",
            "--include-negative-queries",
            "--max-queries",
            "8",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "scene_query_report.json").read_text(encoding="utf-8"))
    result_by_query = {item["query"]: item for item in report["query_results"]}
    assert result_by_query["flat screen"]["provenance"]["planner_backend_call"]["purpose"] == "negative"
    persisted = json.loads((output / "flat_screen" / "query_result.json").read_text(encoding="utf-8"))
    assert persisted["provenance"]["planner_backend_call"]["purpose"] == "negative"
    evidence_labels = [item["label"] for item in report["answer_summary"]["evidence"]]
    assert "flat screen" not in evidence_labels
    assert any(
        "Negative/disambiguation query results" in item
        for item in report["answer_summary"]["limitations"]
    )


def test_query_scene_cli_opennerf_dry_run_uses_num_views(tmp_path: Path) -> None:
    output = tmp_path / "query"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--backend",
            "opennerf",
            "--query",
            "mug",
            "--output",
            str(output),
            "--scene-name",
            "desk_scene",
            "--dry-run",
            "--num-views",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "scene_query_report.json").read_text(encoding="utf-8"))
    rendered = report["query_results"][0]["rendered_images"]
    assert len([item for item in rendered if item["kind"] == "relevancy"]) == 2
    assert report["query_results"][0]["provenance"]["num_views"] == 2


def test_query_scene_cli_can_skip_grid_and_write_montage(tmp_path: Path) -> None:
    output = tmp_path / "query"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--backend",
            "lerf",
            "--query",
            "mug",
            "--output",
            str(output),
            "--scene-name",
            "desk_scene",
            "--dry-run",
            "--no-query-grid",
            "--make-montage",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (output / "query_grid.png").exists()
    assert (output / "query_montage.gif").exists()
    visual_summary = json.loads((output / "query_visual_summary.json").read_text(encoding="utf-8"))
    assert visual_summary["query_grid"] is None
    assert visual_summary["query_montage"] == "query_montage.gif"
    assert "Wrote query montage" in result.stdout
