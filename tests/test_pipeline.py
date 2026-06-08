from pathlib import Path

from nerf_llm_scene_inspector.pipeline import PipelineConfig, run_scene_pipeline


def test_run_scene_pipeline_dry_run_with_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")

    summary = run_scene_pipeline(
        PipelineConfig(
            input_path=tmp_path,
            scene_name="unit_scene",
            data_type="images",
            queries=["mug"],
            data_root=tmp_path / "data",
            runs_root=tmp_path / "runs",
            output_root=tmp_path / "pipeline_runs",
            config_path=config_path,
            dry_run=True,
            skip_baseline=True,
            skip_language=True,
            skip_demo=True,
            skip_eval=True,
        )
    )

    assert summary.success is True
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "pipeline_summary.json").exists()
    assert (
        tmp_path
        / "pipeline_runs"
        / "unit_scene"
        / "queries"
        / "mug"
        / "scene_query_report.json"
    ).exists()
    inspect_step = next(step for step in summary.steps if step.name == "inspect_scene_data")
    assert inspect_step.summary["ready_for_training"] is True
    assert summary.provenance["project_version"] == "0.1.0"
    assert "git_available" in summary.provenance


def test_run_scene_pipeline_writes_run_scoped_demo_and_evaluation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    annotations_path = Path(__file__).resolve().parents[1] / "examples" / "annotations_example.json"
    run_dir = tmp_path / "pipeline_runs" / "scoped_scene"

    summary = run_scene_pipeline(
        PipelineConfig(
            input_path=tmp_path,
            scene_name="scoped_scene",
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
    assert (run_dir / "queries.yaml").exists()
    assert (run_dir / "demo_assets" / "demo_summary.json").exists()
    assert (run_dir / "demo_assets" / "query_grid.png").exists()
    assert (run_dir / "evaluation" / "eval_summary.json").exists()
    assert (run_dir / "evaluation" / "qualitative_report.md").exists()
    assert (run_dir / "project_report.md").exists()
    eval_step = next(step for step in summary.steps if step.name == "evaluate_queries")
    assert eval_step.outputs["eval_summary"] == str(run_dir / "evaluation" / "eval_summary.json")


def test_run_scene_pipeline_cleans_stale_run_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    stale_file = tmp_path / "pipeline_runs" / "clean_scene" / "queries" / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old", encoding="utf-8")

    summary = run_scene_pipeline(
        PipelineConfig(
            input_path=tmp_path,
            scene_name="clean_scene",
            data_type="images",
            queries=["mug"],
            data_root=tmp_path / "data",
            runs_root=tmp_path / "runs",
            output_root=tmp_path / "pipeline_runs",
            config_path=config_path,
            dry_run=True,
            skip_baseline=True,
            skip_language=True,
            skip_demo=True,
            skip_eval=True,
        )
    )

    assert summary.success is True
    assert not stale_file.exists()
