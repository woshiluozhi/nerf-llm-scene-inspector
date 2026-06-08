import json
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
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "capture_manifest.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "capture_manifest_validation.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "preflight_report.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "preflight_report.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "evidence_scorecard.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "evidence_scorecard.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "quality_gate.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "quality_gate.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "claim_audit.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "claim_audit.md").exists()
    assert (
        tmp_path
        / "pipeline_runs"
        / "unit_scene"
        / "evaluation"
        / "annotation_workbench"
        / "annotation_workbench.html"
    ).exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "run_result_card.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "run_result_card.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "portfolio_page.html").exists()
    assert (tmp_path / "pipeline_runs" / "run_index.json").exists()
    assert (tmp_path / "pipeline_runs" / "run_index.md").exists()
    assert (tmp_path / "pipeline_runs" / "run_comparison.json").exists()
    assert (tmp_path / "pipeline_runs" / "run_comparison.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "run_audit.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "run_recommendations.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "reproduction_manifest.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "research_report.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "research_report.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "real_run_plan" / "real_run_plan.json").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "real_run_plan" / "real_run_plan.md").exists()
    assert (tmp_path / "pipeline_runs" / "unit_scene" / "logs" / "prepare_data_command.json").exists()
    assert (
        tmp_path
        / "pipeline_runs"
        / "unit_scene"
        / "queries"
        / "mug"
        / "scene_query_report.json"
    ).exists()
    assert (
        tmp_path
        / "pipeline_runs"
        / "unit_scene"
        / "queries"
        / "mug"
        / "scene_query_report.md"
    ).exists()
    report_payload = json.loads(
        (
            tmp_path
            / "pipeline_runs"
            / "unit_scene"
            / "queries"
            / "mug"
            / "scene_query_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report_payload["scene_name"] == "unit_scene"
    assert report_payload["answer_summary"]["support_level"]
    preflight_step = next(step for step in summary.steps if step.name == "preflight_real_run")
    assert preflight_step.outputs["json"] == str(
        tmp_path / "pipeline_runs" / "unit_scene" / "preflight_report.json"
    )
    inspect_step = next(step for step in summary.steps if step.name == "inspect_scene_data")
    assert inspect_step.summary["ready_for_training"] is True
    assert summary.provenance["project_version"] == "0.1.0"
    assert "git_available" in summary.provenance


def test_run_scene_pipeline_writes_run_scoped_demo_and_evaluation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    prompt_suite = tmp_path / "prompt_suite.yaml"
    prompt_suite.write_text(
        "scene_name: scoped_scene\n"
        "groups:\n"
        "  - name: mug\n"
        "    prompts:\n"
        "      - mug\n"
        "      - coffee mug\n",
        encoding="utf-8",
    )
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
            prompt_suite_path=prompt_suite,
            config_path=config_path,
            dry_run=True,
            analyze_relations=True,
            skip_baseline=True,
            skip_language=True,
        )
    )

    assert summary.success is True
    assert (run_dir / "queries.yaml").exists()
    assert (run_dir / "capture_manifest.json").exists()
    assert (run_dir / "capture_manifest.md").exists()
    assert (run_dir / "capture_manifest_validation.json").exists()
    assert (run_dir / "capture_manifest_validation.md").exists()
    assert (run_dir / "preflight_report.json").exists()
    assert (run_dir / "preflight_report.md").exists()
    assert (run_dir / "evidence_scorecard.json").exists()
    assert (run_dir / "evidence_scorecard.md").exists()
    assert (run_dir / "quality_gate.json").exists()
    assert (run_dir / "quality_gate.md").exists()
    assert (run_dir / "claim_audit.json").exists()
    assert (run_dir / "claim_audit.md").exists()
    assert (run_dir / "run_result_card.json").exists()
    assert (run_dir / "run_result_card.md").exists()
    assert (run_dir / "portfolio_page.html").exists()
    assert "research_report.md" in (run_dir / "portfolio_page.html").read_text(encoding="utf-8")
    assert "claim_audit.md" in (run_dir / "portfolio_page.html").read_text(encoding="utf-8")
    assert "run_result_card.md" in (run_dir / "portfolio_page.html").read_text(encoding="utf-8")
    assert (run_dir / "annotation_template.json").exists()
    assert (run_dir / "demo_assets" / "demo_summary.json").exists()
    assert (run_dir / "demo_assets" / "query_grid.png").exists()
    assert (run_dir / "evaluation" / "annotation_validation.json").exists()
    assert (run_dir / "evaluation" / "annotation_review.json").exists()
    assert (run_dir / "evaluation" / "annotation_review.md").exists()
    assert (run_dir / "evaluation" / "annotation_review_contact_sheet.png").exists()
    assert (run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench.html").exists()
    assert (run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench_manifest.json").exists()
    assert (run_dir / "evaluation" / "annotation_workbench" / "annotation_seed.json").exists()
    assert (run_dir / "evaluation" / "eval_summary.json").exists()
    assert (run_dir / "evaluation" / "qualitative_report.md").exists()
    assert (run_dir / "prompt_sensitivity" / "prompt_sensitivity_summary.json").exists()
    assert (run_dir / "prompt_sensitivity" / "prompt_sensitivity_report.md").exists()
    assert (run_dir / "scene_relations" / "scene_relations_summary.json").exists()
    assert (run_dir / "scene_relations" / "scene_relations_edges.csv").exists()
    assert (run_dir / "scene_relations" / "scene_relations_report.md").exists()
    assert (run_dir / "project_report.md").exists()
    assert (run_dir / "run_audit.json").exists()
    assert (run_dir / "run_audit.md").exists()
    assert (run_dir / "run_recommendations.json").exists()
    assert (run_dir / "run_recommendations.md").exists()
    assert (run_dir / "reproduction_manifest.json").exists()
    assert (run_dir / "reproduction_report.md").exists()
    assert (run_dir / "reproduce_run.sh").exists()
    assert (run_dir / "research_report.json").exists()
    assert (run_dir / "research_report.md").exists()
    assert (run_dir / "real_run_plan" / "real_run_plan.json").exists()
    assert (run_dir / "real_run_plan" / "real_run_plan.md").exists()
    assert "real_run_plan/real_run_plan.md" in (run_dir / "portfolio_page.html").read_text(
        encoding="utf-8"
    )
    assert (run_dir / "submission_packet" / "submission_packet.json").exists()
    assert (run_dir / "submission_packet" / "submission_checklist.md").exists()
    assert (run_dir / "submission_packet" / "cv_project_entry.md").exists()
    assert (run_dir / "submission_packet" / "professor_email_brief.md").exists()
    assert "submission_packet/submission_checklist.md" in (
        run_dir / "portfolio_page.html"
    ).read_text(encoding="utf-8")
    run_index = json.loads((tmp_path / "pipeline_runs" / "run_index.json").read_text(encoding="utf-8"))
    assert run_index["entries"][0]["scene_name"] == "scoped_scene"
    assert run_index["entries"][0]["artifacts"]["capture_manifest"] == "capture_manifest.md"
    assert run_index["entries"][0]["artifacts"]["claim_audit"] == "claim_audit.md"
    assert run_index["entries"][0]["artifacts"]["annotation_workbench"] == (
        "evaluation/annotation_workbench/annotation_workbench.html"
    )
    assert run_index["entries"][0]["artifacts"]["run_result_card"] == "run_result_card.md"
    assert run_index["entries"][0]["artifacts"]["real_run_plan"] == "real_run_plan/real_run_plan.md"
    run_comparison = json.loads(
        (tmp_path / "pipeline_runs" / "run_comparison.json").read_text(encoding="utf-8")
    )
    assert run_comparison["best_run"]["scene_name"] == "scoped_scene"
    assert (run_dir / "logs" / "prepare_data_command.json").exists()
    assert (run_dir / "logs" / "create_annotation_template_command.json").exists()
    assert (run_dir / "logs" / "create_annotation_workbench_command.json").exists()
    assert (run_dir / "logs" / "generate_demo_assets_command.json").exists()
    assert (run_dir / "logs" / "evaluate_queries_command.json").exists()
    assert (run_dir / "logs" / "review_annotations_command.json").exists()
    assert (run_dir / "logs" / "analyze_prompt_sensitivity_command.json").exists()
    assert (run_dir / "logs" / "analyze_scene_relations_command.json").exists()
    eval_step = next(step for step in summary.steps if step.name == "evaluate_queries")
    assert eval_step.outputs["eval_summary"] == str(run_dir / "evaluation" / "eval_summary.json")
    assert eval_step.outputs["annotation_validation"] == str(
        run_dir / "evaluation" / "annotation_validation.json"
    )
    assert eval_step.outputs["command_log"] == str(run_dir / "logs" / "evaluate_queries_command.json")
    review_step = next(step for step in summary.steps if step.name == "review_annotations")
    assert review_step.outputs["annotation_review"] == str(run_dir / "evaluation" / "annotation_review.json")
    assert review_step.outputs["annotation_review_markdown"] == str(
        run_dir / "evaluation" / "annotation_review.md"
    )
    annotation_step = next(step for step in summary.steps if step.name == "create_annotation_template")
    assert annotation_step.outputs["annotation_template"] == str(run_dir / "annotation_template.json")
    workbench_step = next(step for step in summary.steps if step.name == "create_annotation_workbench")
    assert workbench_step.outputs["html"] == str(
        run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench.html"
    )
    query_step = next(step for step in summary.steps if step.name == "query_scene")
    assert query_step.summary["num_queries"] == 2
    assert query_step.outputs["mug_markdown"] == str(run_dir / "queries" / "mug" / "scene_query_report.md")
    assert query_step.outputs["coffee_mug"] == str(
        run_dir / "queries" / "coffee_mug" / "scene_query_report.json"
    )
    prompt_step = next(step for step in summary.steps if step.name == "analyze_prompt_sensitivity")
    assert prompt_step.outputs["markdown"] == str(
        run_dir / "prompt_sensitivity" / "prompt_sensitivity_report.md"
    )
    relation_step = next(step for step in summary.steps if step.name == "analyze_scene_relations")
    assert relation_step.outputs["markdown"] == str(
        run_dir / "scene_relations" / "scene_relations_report.md"
    )
    capture_step = next(step for step in summary.steps if step.name == "capture_manifest")
    assert capture_step.outputs["manifest_json"] == str(run_dir / "capture_manifest.json")
    audit_step = next(step for step in summary.steps if step.name == "audit_run")
    assert audit_step.outputs["json"] == str(run_dir / "run_audit.json")
    recommendation_step = next(step for step in summary.steps if step.name == "recommend_next_steps")
    assert recommendation_step.outputs["json"] == str(run_dir / "run_recommendations.json")
    recommendation_payload = json.loads((run_dir / "run_recommendations.json").read_text(encoding="utf-8"))
    assert recommendation_payload["readiness_level"] == "dry_run_ready_for_smoke_demo"
    scorecard_step = next(step for step in summary.steps if step.name == "create_evidence_scorecard")
    assert scorecard_step.outputs["json"] == str(run_dir / "evidence_scorecard.json")
    scorecard_payload = json.loads((run_dir / "evidence_scorecard.json").read_text(encoding="utf-8"))
    assert scorecard_payload["evidence_level"] == "dry_run_demo_ready"
    portfolio_criterion = next(
        criterion for criterion in scorecard_payload["criteria"] if criterion["name"] == "portfolio_packaging"
    )
    assert "research_report" not in portfolio_criterion["detail"]
    quality_step = next(step for step in summary.steps if step.name == "quality_gate")
    assert quality_step.outputs["json"] == str(run_dir / "quality_gate.json")
    quality_payload = json.loads((run_dir / "quality_gate.json").read_text(encoding="utf-8"))
    assert quality_payload["profile"] == "smoke"
    assert quality_payload["passed"] is True
    page_step = next(step for step in summary.steps if step.name == "generate_portfolio_page")
    assert page_step.outputs["html"] == str(run_dir / "portfolio_page.html")
    reproduction_step = next(step for step in summary.steps if step.name == "create_reproduction_bundle")
    assert reproduction_step.outputs["manifest"] == str(run_dir / "reproduction_manifest.json")
    reproduction_payload = json.loads((run_dir / "reproduction_manifest.json").read_text(encoding="utf-8"))
    assert reproduction_payload["replay_command"].startswith("python scripts/run_scene_pipeline.py")
    assert any(artifact["name"] == "real_run_plan" and artifact["exists"] for artifact in reproduction_payload["artifacts"])
    assert any(artifact["name"] == "claim_audit" and artifact["exists"] for artifact in reproduction_payload["artifacts"])
    assert any(artifact["name"] == "annotation_workbench" and artifact["exists"] for artifact in reproduction_payload["artifacts"])
    assert any(artifact["name"] == "run_result_card" and artifact["exists"] for artifact in reproduction_payload["artifacts"])
    research_step = next(step for step in summary.steps if step.name == "generate_research_report")
    assert research_step.outputs["markdown"] == str(run_dir / "research_report.md")
    submission_step = next(step for step in summary.steps if step.name == "create_submission_packet")
    assert submission_step.outputs["markdown"] == str(
        run_dir / "submission_packet" / "submission_checklist.md"
    )
    submission_payload = json.loads(
        (run_dir / "submission_packet" / "submission_packet.json").read_text(encoding="utf-8")
    )
    assert submission_payload["readiness_level"] == "needs_pack_validation"
    plan_step = next(step for step in summary.steps if step.name == "create_real_run_plan")
    assert plan_step.outputs["markdown"] == str(run_dir / "real_run_plan" / "real_run_plan.md")
    plan_payload = json.loads((run_dir / "real_run_plan" / "real_run_plan.json").read_text(encoding="utf-8"))
    assert plan_payload["current_mode"] == "dry-run smoke demo"
    claim_step = next(step for step in summary.steps if step.name == "audit_claims")
    assert claim_step.outputs["markdown"] == str(run_dir / "claim_audit.md")
    claim_payload = json.loads((run_dir / "claim_audit.json").read_text(encoding="utf-8"))
    assert claim_payload["status"] in {"pass", "warn"}
    result_card_step = next(step for step in summary.steps if step.name == "create_run_result_card")
    assert result_card_step.outputs["markdown"] == str(run_dir / "run_result_card.md")
    result_card_payload = json.loads((run_dir / "run_result_card.json").read_text(encoding="utf-8"))
    assert result_card_payload["result_status"] in {"shareable_smoke_demo", "dry_run_smoke_demo"}
    comparison_step = next(step for step in summary.steps if step.name == "compare_runs")
    assert comparison_step.outputs["json"] == str(tmp_path / "pipeline_runs" / "run_comparison.json")


def test_run_scene_pipeline_writes_run_scoped_training_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "pipeline_runs" / "training_scene"

    summary = run_scene_pipeline(
        PipelineConfig(
            input_path=tmp_path,
            scene_name="training_scene",
            data_type="images",
            queries=["mug"],
            data_root=tmp_path / "data",
            runs_root=tmp_path / "runs",
            output_root=tmp_path / "pipeline_runs",
            dry_run=True,
            skip_queries=True,
            skip_demo=True,
            skip_eval=True,
        )
    )

    assert summary.success is True
    baseline_summary = run_dir / "training" / "baseline_train_summary.json"
    language_summary = run_dir / "training" / "language_train_summary.json"
    assert baseline_summary.exists()
    assert language_summary.exists()
    assert (run_dir / "logs" / "train_baseline_nerf_command.json").exists()
    assert (run_dir / "logs" / "train_language_field_command.json").exists()
    assert json.loads(baseline_summary.read_text(encoding="utf-8"))["run_type"] == "baseline"
    assert json.loads(language_summary.read_text(encoding="utf-8"))["run_type"] == "language"
    baseline_step = next(step for step in summary.steps if step.name == "train_baseline_nerf")
    language_step = next(step for step in summary.steps if step.name == "train_language_field")
    assert baseline_step.outputs["train_summary"] == str(baseline_summary)
    assert language_step.outputs["train_summary"] == str(language_summary)


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
