import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.failure_diagnostics import (
    build_failure_diagnostics,
    write_failure_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_failure_diagnostics_clear_run(tmp_path: Path) -> None:
    run_dir = _write_base_run(tmp_path)

    report = build_failure_diagnostics(run_dir)

    assert report.status == "clear"
    assert report.blocker_count == 0
    assert report.command_log_count == 1
    assert report.training_summary_count == 1


def test_failure_diagnostics_classifies_real_run_failures(tmp_path: Path) -> None:
    run_dir = _write_base_run(tmp_path)
    _write_json(
        run_dir / "logs" / "train_language_field_command.json",
        {
            "returncode": 1,
            "stdout": "",
            "stderr": "RuntimeError: CUDA out of memory. ns-train does not list method 'lerf-lite'",
        },
    )
    _write_json(
        run_dir / "training" / "language_train_summary.json",
        {"success": False, "dry_run": False, "config_path": None},
    )

    report = build_failure_diagnostics(run_dir)

    categories = {item.category for item in report.diagnostics}
    assert report.status == "blocked"
    assert "cuda_oom" in categories
    assert "lerf_method_missing" in categories
    assert "command_failed" in categories
    assert "missing_trained_config" in categories


def test_failure_diagnostics_detects_lerf_viewer_fallback(tmp_path: Path) -> None:
    run_dir = _write_base_run(tmp_path)
    _write_json(
        run_dir / "queries" / "mug" / "query_result.json",
        {
            "query": "mug",
            "rendered_images": [{"kind": "viewer_fallback", "path": "interactive_viewer_workflow.md"}],
            "warnings": ["Automated LERF rendering failed; wrote viewer fallback instructions."],
        },
    )

    report = build_failure_diagnostics(run_dir)

    assert report.status == "needs_attention"
    assert any(item.category == "lerf_render_fallback" for item in report.diagnostics)


def test_write_failure_diagnostics_outputs_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_base_run(tmp_path)

    report = write_failure_diagnostics(run_dir)

    payload = json.loads((run_dir / "failure_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["status"] == report.status
    markdown = (run_dir / "failure_diagnostics.md").read_text(encoding="utf-8")
    assert "# Failure Diagnostics" in markdown
    assert "Command logs inspected" in markdown


def test_diagnose_run_failures_cli_returns_nonzero_for_blockers(tmp_path: Path) -> None:
    run_dir = _write_base_run(tmp_path)
    output = tmp_path / "diagnostics.json"
    markdown = tmp_path / "diagnostics.md"
    _write_json(
        run_dir / "logs" / "prepare_data_command.json",
        {"returncode": 1, "stderr": "Expected transforms.json after processing, but it was not found."},
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "diagnose_run_failures.py"),
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

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert any(item["category"] == "transforms_missing" for item in payload["diagnostics"])
    assert markdown.exists()


def _write_base_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": True,
            "backend": "lerf",
            "steps": [{"name": "prepare_data", "status": "success"}],
        },
    )
    _write_json(run_dir / "environment_report.json", {"ok": True, "strict_failures": []})
    _write_json(run_dir / "preflight_report.json", {"status": "ready", "checks": []})
    _write_json(run_dir / "logs" / "prepare_data_command.json", {"returncode": 0, "stdout": "ok"})
    _write_json(
        run_dir / "training" / "language_train_summary.json",
        {"success": True, "dry_run": True, "config_path": str(run_dir / "config.yml")},
    )
    (run_dir / "config.yml").write_text("method_name: lerf-lite\n", encoding="utf-8")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
