import csv
import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.experiment_matrix import run_experiment_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_collects_existing_runs(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: unit_matrix\n"
        "experiments:\n"
        "  - name: lerf_lite\n"
        "    scene_name: scene_lerf\n"
        "    backend: lerf\n"
        "    variant: lerf-lite\n"
        "  - name: opennerf\n"
        "    scene_name: scene_opennerf\n"
        "    backend: opennerf\n",
        encoding="utf-8",
    )
    _write_run(output / "pipeline_runs" / "scene_lerf", scene_name="scene_lerf", backend="lerf", score=90)
    _write_run(
        output / "pipeline_runs" / "scene_opennerf",
        scene_name="scene_opennerf",
        backend="opennerf",
        score=70,
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    assert report.total_experiments == 2
    assert report.successful_experiments == 2
    assert report.best_experiment is not None
    assert report.best_experiment["experiment_name"] == "lerf_lite"
    assert (output / "experiment_matrix_summary.json").exists()
    assert (output / "experiment_matrix_report.md").exists()
    rows = list(csv.DictReader((output / "experiment_matrix_table.csv").open(encoding="utf-8")))
    assert rows[0]["experiment_name"] == "lerf_lite"
    assert rows[0]["relation_edge_count"] == "4"


def test_run_experiment_matrix_cli_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "matrix_output"
    matrix.write_text(
        "matrix_name: cli_matrix\n"
        "base:\n"
        f"  input: {tmp_path.as_posix()}\n"
        "  type: images\n"
        f"  config: {config_path.as_posix()}\n"
        "  dry_run: true\n"
        "  skip_baseline: true\n"
        "  skip_language: true\n"
        "  skip_demo: true\n"
        "  skip_eval: true\n"
        "  analyze_relations: true\n"
        "experiments:\n"
        "  - name: cli_lerf\n"
        "    scene_name: cli_lerf\n"
        "    backend: lerf\n"
        "    queries:\n"
        "      - mug\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment_matrix.py"),
            "--config",
            str(matrix),
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
    summary = json.loads((output / "experiment_matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["matrix_name"] == "cli_matrix"
    assert summary["successful_experiments"] == 1
    assert (output / "pipeline_runs" / "cli_lerf" / "scene_relations" / "scene_relations_report.md").exists()


def _write_run(run_dir: Path, *, scene_name: str, backend: str, score: int) -> None:
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": scene_name,
            "success": True,
            "dry_run": True,
            "backend": backend,
            "queries": ["mug", "cup"],
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    )
    _write_json(run_dir / "evidence_scorecard.json", {"evidence_level": "dry_run_demo_ready", "score": score, "max_score": 100})
    _write_json(run_dir / "run_audit.json", {"status": "ready"})
    _write_json(run_dir / "quality_gate.json", {"status": "warn", "passed": True})
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {
            "num_evaluated_queries": 2,
            "top_k_hit_rate": 0.5,
            "mean_iou_2d": 0.25,
            "average_relevancy_score": 0.8,
        },
    )
    _write_json(
        run_dir / "prompt_sensitivity" / "prompt_sensitivity_summary.json",
        {"stable_group_count": 1, "num_groups": 2},
    )
    _write_json(
        run_dir / "scene_relations" / "scene_relations_summary.json",
        {"num_entities": 3, "num_relations": 4},
    )
    _write_json(run_dir / "run_recommendations.json", {"top_next_action": "review real outputs"})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
