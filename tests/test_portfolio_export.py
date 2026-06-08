import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack
from nerf_llm_scene_inspector.pipeline import PipelineConfig, run_scene_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_export_portfolio_pack_from_pipeline_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    annotations_path = ROOT / "examples" / "annotations_example.json"
    run_dir = tmp_path / "pipeline_runs" / "export_scene"
    output_dir = tmp_path / "portfolio_pack"

    summary = run_scene_pipeline(
        PipelineConfig(
            input_path=tmp_path,
            scene_name="export_scene",
            data_type="images",
            queries=["mug"],
            data_root=tmp_path / "data",
            runs_root=tmp_path / "runs",
            output_root=tmp_path / "pipeline_runs",
            annotations_path=annotations_path,
            config_path=config_path,
            dry_run=True,
            skip_baseline=True,
            skip_language=True,
        )
    )
    assert summary.success is True

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_portfolio_pack.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    index = json.loads((output_dir / "portfolio_pack_index.json").read_text(encoding="utf-8"))
    assert index["missing"] == []
    assert "baseline_train_summary" not in index["run_summary"]["artifacts"]
    assert "language_train_summary" not in index["run_summary"]["artifacts"]
    assert index["run_summary"]["artifacts"]["preflight_report"] == "run/preflight_report.md"
    assert index["run_summary"]["artifacts"]["evidence_scorecard"] == "run/evidence_scorecard.md"
    assert index["run_summary"]["artifacts"]["portfolio_page"] == "run/portfolio_page.html"
    assert index["run_summary"]["artifacts"]["annotation_review"] == "run/evaluation/annotation_review.md"
    assert index["run_summary"]["scene_name"] == "export_scene"
    assert str(tmp_path) not in json.dumps(index)
    packed_summary = (output_dir / "run" / "pipeline_summary.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in packed_summary
    packed_log = (output_dir / "run" / "logs" / "prepare_data_command.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in packed_log
    demo_log = json.loads(
        (output_dir / "run" / "logs" / "generate_demo_assets_command.json").read_text(encoding="utf-8")
    )
    assert demo_log["command"][0] == "python"
    assert (output_dir / "run" / "pipeline_summary.json").exists()
    assert (output_dir / "project" / "docs" / "index.html").exists()
    assert (output_dir / "run" / "preflight_report.json").exists()
    assert (output_dir / "run" / "preflight_report.md").exists()
    assert (output_dir / "run" / "evidence_scorecard.json").exists()
    assert (output_dir / "run" / "evidence_scorecard.md").exists()
    assert (output_dir / "run" / "portfolio_page.html").exists()
    assert str(tmp_path) not in (output_dir / "run" / "portfolio_page.html").read_text(encoding="utf-8")
    assert (output_dir / "run_index.json").exists()
    assert (output_dir / "run_index.md").exists()
    assert (output_dir / "run" / "logs" / "prepare_data_command.json").exists()
    assert (output_dir / "run" / "logs" / "generate_demo_assets_command.json").exists()
    assert (output_dir / "run" / "run_audit.json").exists()
    assert (output_dir / "run" / "run_audit.md").exists()
    assert (output_dir / "run" / "run_recommendations.json").exists()
    assert (output_dir / "run" / "run_recommendations.md").exists()
    assert (output_dir / "run" / "reproduction_manifest.json").exists()
    assert (output_dir / "run" / "reproduction_report.md").exists()
    assert (output_dir / "run" / "reproduce_run.sh").exists()
    assert (output_dir / "run" / "annotation_template.json").exists()
    assert (output_dir / "run" / "project_report.md").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_validation.json").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_review.json").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_review.md").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_review_contact_sheet.png").exists()
    assert (output_dir / "run" / "evaluation" / "eval_summary.json").exists()
    assert (output_dir / "run" / "demo_assets" / "query_grid.png").exists()
    validation = validate_portfolio_pack(output_dir)
    assert validation.path_leaks == []
    assert validation.ok is True, validation.to_dict()
    assert validation.artifact_issues == []
