import csv
import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult
from nerf_llm_scene_inspector.evaluation.prompt_sensitivity import analyze_prompt_sensitivity


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_sensitivity_marks_stable_prompt_group(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    results = tmp_path / "results"
    output = tmp_path / "prompt_sensitivity"
    _write_result(results / "mug" / "query_result.json", "mug", (0, 0, 100, 100), 0.9)
    _write_result(results / "coffee_mug" / "query_result.json", "coffee mug", (5, 5, 105, 105), 0.85)

    report = analyze_prompt_sensitivity(
        suite_path=suite,
        results_dir=results,
        output_dir=output,
        min_box_consistency_iou=0.2,
    )

    assert report.stable_group_count == 1
    group = report.groups[0]
    assert group.stability_label == "stable"
    assert group.num_results == 2
    assert group.mean_pairwise_top1_iou is not None
    assert (output / "prompt_sensitivity_summary.json").exists()
    assert "# Prompt Sensitivity Report" in (output / "prompt_sensitivity_report.md").read_text(
        encoding="utf-8"
    )


def test_prompt_sensitivity_reports_missing_prompts(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    results = tmp_path / "results"
    _write_result(results / "mug" / "query_result.json", "mug", (0, 0, 100, 100), 0.9)

    report = analyze_prompt_sensitivity(suite_path=suite, results_dir=results)

    assert report.groups[0].stability_label == "insufficient_evidence"
    assert report.groups[0].missing_prompts == ["coffee mug"]
    assert report.groups[0].rows[1].status == "missing"


def test_prompt_sensitivity_prefers_direct_prompt_result(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    results = tmp_path / "results"
    direct_path = results / "mug" / "mug" / "query_result.json"
    _write_result(direct_path, "mug", (0, 0, 100, 100), 0.7)
    _write_result(results / "other_task" / "mug" / "query_result.json", "mug", (200, 200, 300, 300), 0.95)
    _write_result(
        results / "coffee_mug" / "coffee_mug" / "query_result.json",
        "coffee mug",
        (5, 5, 105, 105),
        0.8,
    )

    report = analyze_prompt_sensitivity(suite_path=suite, results_dir=results)

    mug_row = report.groups[0].rows[0]
    assert mug_row.prompt == "mug"
    assert mug_row.result_path == str(direct_path)
    assert mug_row.confidence == 0.7


def test_prompt_sensitivity_cli_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path)
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_prompt_sensitivity.py"),
            "--suite",
            str(suite),
            "--results",
            str(tmp_path / "missing_results"),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "prompt_sensitivity_summary.json").read_text(encoding="utf-8"))
    assert summary["warnings"] == ["Dry-run synthetic prompt sensitivity results were generated."]
    rows = list(csv.DictReader((output / "prompt_sensitivity_table.csv").open(encoding="utf-8")))
    assert [row["prompt"] for row in rows] == ["mug", "coffee mug"]
    assert (output / "prompt_sensitivity_report.md").exists()


def _write_suite(tmp_path: Path) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(
        "\n".join(
            [
                "scene_name: desk_scene",
                "groups:",
                "  - name: mug",
                "    description: Mug wording variants.",
                "    prompts:",
                "      - mug",
                "      - coffee mug",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_result(path: Path, query: str, bbox: tuple[float, float, float, float], score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    QueryResult(
        query=query,
        backend_name="dry-run",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(label=query, score=score, bbox_2d=bbox, source_view="view_0000")
        ],
        confidence=score,
    ).to_json(path)
