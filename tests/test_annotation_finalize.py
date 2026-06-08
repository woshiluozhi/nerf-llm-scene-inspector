import json
import subprocess
import sys
import zipfile
from pathlib import Path

from nerf_llm_scene_inspector.evaluation import annotation_finalize
from nerf_llm_scene_inspector.evaluation.annotation_finalize import finalize_workbench_annotations
from nerf_llm_scene_inspector.pipeline import PipelineConfig, run_scene_pipeline
from nerf_llm_scene_inspector.utils.shell import CommandResult


ROOT = Path(__file__).resolve().parents[1]


def test_finalize_workbench_annotations_refreshes_run_artifacts(tmp_path: Path) -> None:
    run_dir = _pipeline_run(tmp_path, "finalize_scene")
    filled = run_dir / "evaluation" / "annotation_workbench" / "annotation_seed.json"

    report = finalize_workbench_annotations(
        run_dir=run_dir,
        filled_path=filled,
        profile="smoke",
        continue_on_error=False,
    )

    assert report.ok is True, report.errors
    assert (run_dir / "annotations_merged.json").exists()
    assert (run_dir / "annotation_merge_report.json").exists()
    assert (run_dir / "annotation_finalize_report.json").exists()
    assert (run_dir / "annotation_finalize_report.md").exists()
    assert (run_dir / "evaluation" / "eval_summary.json").exists()
    assert (run_dir / "run_result_card.json").exists()
    assert (run_dir / "portfolio_page.html").exists()
    assert (run_dir.parent / "run_index.json").exists()
    assert (run_dir.parent / "run_comparison.json").exists()
    step_names = [step.name for step in report.steps]
    assert step_names[:3] == ["merge_annotation_workbench", "evaluate_queries", "review_annotations"]
    assert "create_run_result_card" in step_names
    assert (run_dir / "logs" / "finalize_merge_annotation_workbench_command.json").exists()
    merge_report = json.loads((run_dir / "annotation_merge_report.json").read_text(encoding="utf-8"))
    assert merge_report["validation"]["ok"] is True


def test_finalize_annotations_cli(tmp_path: Path) -> None:
    run_dir = _pipeline_run(tmp_path, "finalize_cli_scene")
    filled = run_dir / "evaluation" / "annotation_workbench" / "annotation_seed.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "finalize_annotations.py"),
            "--run-dir",
            str(run_dir),
            "--filled",
            str(filled),
            "--profile",
            "smoke",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((run_dir / "annotation_finalize_report.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert any(step["name"] == "evaluate_queries" for step in payload["steps"])


def test_finalize_continues_after_noncritical_failure(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    filled = tmp_path / "filled.json"
    filled.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    def fake_run_command(command, *, cwd=None, log_path=None, check=False):  # noqa: ANN001, ANN202
        del cwd, log_path, check
        script = Path(command[1]).name
        calls.append(script)
        if script == "check_run_quality.py":
            return CommandResult(command=[str(item) for item in command], returncode=1, stderr="quality gate failed")
        return CommandResult(command=[str(item) for item in command], returncode=0)

    monkeypatch.setattr(annotation_finalize, "run_command", fake_run_command)

    report = finalize_workbench_annotations(run_dir=run_dir, filled_path=filled, profile="smoke")

    assert report.ok is True
    assert any("check_run_quality failed" in warning for warning in report.warnings)
    assert "generate_research_report.py" in calls
    assert "compare_runs.py" in calls


def test_finalize_export_pack_refreshes_quality_after_pack_validation(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    filled = tmp_path / "filled.json"
    filled.write_text("{}", encoding="utf-8")

    def fake_run_command(command, *, cwd=None, log_path=None, check=False):  # noqa: ANN001, ANN202
        del cwd, log_path, check
        return CommandResult(command=[str(item) for item in command], returncode=0)

    monkeypatch.setattr(annotation_finalize, "run_command", fake_run_command)

    report = finalize_workbench_annotations(run_dir=run_dir, filled_path=filled, profile="smoke", export_pack=True, zip_pack=True)
    step_names = [step.name for step in report.steps]

    assert report.ok is True
    assert step_names.index("refresh_quality_gate_with_pack") > step_names.index("validate_portfolio_pack")
    assert report.steps[step_names.index("check_run_quality")].command.endswith("--no-require-pack")
    assert "--pack" in report.steps[step_names.index("refresh_quality_gate_with_pack")].command
    assert step_names.index("final_export_portfolio_pack") > step_names.index("refresh_reproduction_bundle")
    assert step_names.index("final_validate_portfolio_pack") > step_names.index("final_export_portfolio_pack")
    assert step_names.index("final_archive_portfolio_pack") > step_names.index("final_validate_portfolio_pack")


def test_finalize_zip_pack_contains_final_validation_report(tmp_path: Path) -> None:
    run_dir = _pipeline_run(tmp_path, "finalize_zip_scene")
    filled = run_dir / "evaluation" / "annotation_workbench" / "annotation_seed.json"
    pack_dir = tmp_path / "portfolio_pack"

    report = finalize_workbench_annotations(
        run_dir=run_dir,
        filled_path=filled,
        profile="smoke",
        pack_dir=pack_dir,
        export_pack=True,
        zip_pack=True,
        continue_on_error=False,
    )

    assert report.ok is True, report.errors
    assert (pack_dir / "portfolio_pack_validation.json").exists()
    archive_path = Path(f"{pack_dir}.zip")
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "portfolio_pack_index.json" in names
        assert "portfolio_pack_validation.json" in names
        assert "run/submission_packet/submission_packet.json" in names
        validation = json.loads(archive.read("portfolio_pack_validation.json").decode("utf-8"))
    assert validation["ok"] is True


def _pipeline_run(tmp_path: Path, scene_name: str) -> Path:
    config_path = tmp_path / f"{scene_name}_config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    summary = run_scene_pipeline(
        PipelineConfig(
            input_path=tmp_path,
            scene_name=scene_name,
            data_type="images",
            queries=["mug"],
            data_root=tmp_path / "data",
            runs_root=tmp_path / "runs",
            output_root=tmp_path / "pipeline_runs",
            config_path=config_path,
            dry_run=True,
            skip_baseline=True,
            skip_language=True,
            analyze_relations=True,
        )
    )
    assert summary.success is True
    return tmp_path / "pipeline_runs" / scene_name
