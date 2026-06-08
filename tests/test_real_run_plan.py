import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.real_run_plan import (
    build_real_run_plan,
    write_real_run_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def test_build_real_run_plan_from_dry_run_artifacts(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    plan = build_real_run_plan(
        run_dir,
        input_path="captures/desk.mp4",
        input_type="video",
        processed_data="data/processed/desk_scene",
        repo_url="https://github.com/woshiluozhi/nerf-llm-scene-inspector",
    )

    assert plan.scene_name == "desk_scene"
    assert plan.current_mode == "dry-run smoke demo"
    assert plan.backend == "lerf"
    assert plan.variant == "lerf-lite"
    assert plan.input_type == "video"
    assert plan.query_count == 1
    assert any(issue.category == "run_mode" for issue in plan.issues)
    assert any(command.name == "train_language_pipeline" for command in plan.commands)
    assert any(command.name == "merge_annotation_workbench_export" for command in plan.commands)
    assert any("scripts/merge_annotation_workbench.py" in command.command for command in plan.commands)
    assert any("scripts/run_scene_pipeline.py" in command.command for command in plan.commands)
    assert any("without --dry-run" in item for item in plan.claim_upgrade_path)


def test_write_real_run_plan_outputs_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    output_dir = tmp_path / "plan"

    plan = write_real_run_plan(run_dir, output_dir=output_dir, input_path="captures/desk.mp4")

    assert (output_dir / "real_run_plan.json").exists()
    assert (output_dir / "real_run_plan.md").exists()
    payload = json.loads((output_dir / "real_run_plan.json").read_text(encoding="utf-8"))
    assert payload["scene_name"] == plan.scene_name
    assert payload["warning_count"] >= 1
    assert "# Real-Run Action Plan" in (output_dir / "real_run_plan.md").read_text(encoding="utf-8")


def test_create_real_run_plan_cli(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    output_dir = tmp_path / "cli_plan"
    external_packet = tmp_path / "submission_packet.json"
    external_packet.write_text(
        json.dumps({"readiness_level": "shareable_smoke_demo"}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_real_run_plan.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_dir),
            "--input",
            "captures/desk.mp4",
            "--type",
            "video",
            "--submission-packet",
            str(external_packet),
            "--no-require-gpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((output_dir / "real_run_plan.json").read_text(encoding="utf-8"))
    assert payload["require_gpu"] is False
    assert payload["readiness_level"] == "shareable_smoke_demo"
    assert payload["commands"][0]["status"] == "ready"


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "pipeline_runs" / "desk_scene"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": True,
            "backend": "lerf",
            "queries": ["mug"],
            "paths": {"processed_data": "data/processed/desk_scene"},
        },
    )
    _write_json(
        run_dir / "capture_manifest.json",
        {"scene_name": "desk_scene", "input_path": "examples", "input_type": "images"},
    )
    _write_json(run_dir / "capture_manifest_validation.json", {"status": "needs_review"})
    _write_json(run_dir / "preflight_report.json", {"status": "ready"})
    _write_json(run_dir / "scene_data_inspection.json", {"ready_for_training": True})
    _write_json(run_dir / "quality_gate.json", {"status": "warn"})
    _write_json(
        run_dir / "run_recommendations.json",
        {"readiness_level": "dry_run_ready_for_smoke_demo", "recommendations": []},
    )
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {"readiness_level": "shareable_smoke_demo"},
    )
    _write_json(run_dir / "training" / "language_train_summary.json", {"variant": "lerf-lite"})
    (run_dir / "queries.yaml").write_text("queries:\n  - mug\n", encoding="utf-8")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
