import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.research_report import build_research_report, write_research_report


ROOT = Path(__file__).resolve().parents[1]


def test_build_research_report_from_run_artifacts(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")

    report = build_research_report(run_dir)

    assert report.scene_name == "desk_scene"
    assert report.backend == "lerf"
    assert report.key_results["evidence_level"] == "dry_run_demo_ready"
    assert report.key_results["scene_relation_edges"] == 4
    assert "Dry-run outputs are synthetic" in " ".join(report.limitations)

    written = write_research_report(run_dir)
    assert (run_dir / "research_report.md").exists()
    assert (run_dir / "research_report.json").exists()
    assert written.artifacts["scene_relations"] == "scene_relations/scene_relations_report.md"
    assert written.artifacts["run_result_card"] == "run_result_card.md"


def test_generate_research_report_cli(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    output = tmp_path / "report.md"
    json_output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_research_report.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--json-output",
            str(json_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "# NeRF-LLM Scene Inspector Research Report" in output.read_text(encoding="utf-8")
    assert json.loads(json_output.read_text(encoding="utf-8"))["scene_name"] == "desk_scene"


def _write_run(run_dir: Path) -> Path:
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": True,
            "backend": "lerf",
            "queries": ["mug", "cup"],
        },
    )
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "scene_name": "desk_scene",
            "backend": "lerf",
            "dry_run": True,
            "evidence_level": "dry_run_demo_ready",
            "score": 80,
            "max_score": 100,
            "summary": "Dry-run smoke demo.",
        },
    )
    _write_json(run_dir / "run_audit.json", {"status": "needs_review"})
    _write_json(run_dir / "quality_gate.json", {"status": "warn"})
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {"num_evaluated_queries": 1, "num_bbox_annotated_queries": 1, "top_k_hit_rate": 0.5},
    )
    _write_json(
        run_dir / "prompt_sensitivity" / "prompt_sensitivity_summary.json",
        {"stable_group_count": 1, "num_groups": 2},
    )
    _write_json(
        run_dir / "scene_relations" / "scene_relations_summary.json",
        {"num_entities": 3, "num_relations": 4},
    )
    _write_text(run_dir / "scene_relations" / "scene_relations_report.md", "# Relations\n")
    _write_text(run_dir / "run_result_card.md", "# Run Result Card\n")
    _write_json(
        run_dir / "run_recommendations.json",
        {"recommendations": [{"severity": "high", "action": "Run a real GPU experiment."}]},
    )
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
