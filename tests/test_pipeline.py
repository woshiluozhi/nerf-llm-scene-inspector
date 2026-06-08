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
            results_root=tmp_path / "results",
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
