import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.run_audit import audit_pipeline_run


ROOT = Path(__file__).resolve().parents[1]


def test_audit_pipeline_run_reports_ready_run(tmp_path: Path) -> None:
    run_dir = _write_complete_run(tmp_path)

    report = audit_pipeline_run(run_dir)

    assert report.status == "ready"
    assert report.score == 100
    assert report.query_report_count == 1
    assert report.evaluated_query_count == 1
    assert report.run_dir == "run"
    assert report.key_artifacts["command_logs"] == "logs/"


def test_audit_pipeline_run_blocks_missing_successful_artifact(tmp_path: Path) -> None:
    run_dir = _write_complete_run(tmp_path)
    (run_dir / "demo_assets" / "query_grid.png").unlink()

    report = audit_pipeline_run(run_dir)

    assert report.status == "blocked"
    assert any(finding.category == "missing_artifact" for finding in report.findings)


def test_audit_pipeline_run_blocks_missing_declared_command_log(tmp_path: Path) -> None:
    run_dir = _write_complete_run(tmp_path)
    (run_dir / "logs" / "prepare_data_command.json").unlink()

    report = audit_pipeline_run(run_dir)

    assert report.status == "blocked"
    assert any(finding.category == "command_logs" for finding in report.findings)


def test_audit_pipeline_run_warns_on_capture_manifest_review(tmp_path: Path) -> None:
    run_dir = _write_complete_run(tmp_path)
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "needs_review", "ok": False, "warn_count": 2, "fail_count": 0},
    )

    report = audit_pipeline_run(run_dir)

    assert report.status == "needs_review"
    assert any(finding.category == "capture_manifest" for finding in report.findings)


def test_audit_run_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_complete_run(tmp_path)
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_run.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready"
    assert "# Pipeline Run Audit" in markdown.read_text(encoding="utf-8")


def _write_complete_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    steps = [
        {"name": "preflight_real_run", "status": "success"},
        {"name": "check_environment", "status": "success"},
        {
            "name": "prepare_data",
            "status": "success",
            "outputs": {"command_log": str(tmp_path / "run" / "logs" / "prepare_data_command.json")},
        },
        {"name": "inspect_scene_data", "status": "success"},
        {"name": "train_baseline_nerf", "status": "success"},
        {"name": "train_language_field", "status": "success"},
        {"name": "query_scene", "status": "success"},
        {"name": "create_annotation_template", "status": "success"},
        {"name": "generate_demo_assets", "status": "success"},
        {"name": "evaluate_queries", "status": "success"},
    ]
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": False,
            "backend": "lerf",
            "queries": ["mug"],
            "steps": steps,
        },
    )
    _write_json(run_dir / "capture_manifest.json", {"scene_name": "desk_scene"})
    _write_text(run_dir / "capture_manifest.md", "# Capture\n")
    _write_json(run_dir / "capture_manifest_validation.json", {"status": "ready", "ok": True})
    _write_text(run_dir / "capture_manifest_validation.md", "# Capture Validation\n")
    _write_json(run_dir / "preflight_report.json", {"status": "ready", "ready_for_real_run": True})
    _write_text(run_dir / "preflight_report.md", "# Preflight\n")
    _write_json(run_dir / "environment_report.json", {"ok": True, "strict_failures": []})
    _write_json(run_dir / "logs" / "prepare_data_command.json", {"returncode": 0})
    _write_json(
        run_dir / "scene_data_inspection.json",
        {"ready_for_training": True, "quality_score": 0.95},
    )
    (run_dir / "queries.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "queries.yaml").write_text("scene_name: desk_scene\nqueries:\n  - mug\n", encoding="utf-8")
    _write_json(run_dir / "training" / "baseline_train_summary.json", {"success": True})
    _write_json(run_dir / "training" / "language_train_summary.json", {"success": True})
    _write_json(run_dir / "queries" / "mug" / "scene_query_report.json", {"query": "mug"})
    _write_json(run_dir / "annotation_template.json", {"queries": [{"query": "mug"}]})
    _write_json(
        run_dir / "evaluation" / "annotation_validation.json",
        {"ok": True, "warnings": []},
    )
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"num_evaluated_queries": 1})
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image")
    _write_text(run_dir / "project_report.md", "# Report\n")
    _write_text(run_dir / "portfolio_result_card.md", "# Card\n")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
