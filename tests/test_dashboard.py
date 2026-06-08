import json
from pathlib import Path

from nerf_llm_scene_inspector.visualization.dashboard import (
    collect_command_logs,
    collect_query_reports,
    collect_run_images,
    load_run_bundle,
)


def test_load_run_bundle_collects_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "demo_assets" / "mug").mkdir(parents=True)
    (run_dir / "queries" / "mug").mkdir(parents=True)
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "success": True,
            "backend": "lerf",
            "dry_run": True,
            "queries": ["mug"],
            "steps": [
                {"name": "train_baseline_nerf", "status": "success"},
                {"name": "train_language_field", "status": "success"},
                {"name": "query_scene", "status": "success"},
                {"name": "analyze_prompt_sensitivity", "status": "success"},
                {"name": "analyze_scene_relations", "status": "success"},
                {"name": "review_annotations", "status": "success"},
                {"name": "generate_research_report", "status": "success"},
                {"name": "create_real_run_plan", "status": "success"},
                {"name": "audit_claims", "status": "success"},
                {"name": "create_run_result_card", "status": "success"},
                {"name": "create_submission_packet", "status": "success"},
            ],
            "provenance": {"git_commit": "abc123"},
        },
    )
    _write_json(tmp_path / "run_index.json", {"total_runs": 1, "entries": [{"scene_name": "run"}]})
    _write_json(
        tmp_path / "run_comparison.json",
        {
            "total_runs": 1,
            "portfolio_candidate_count": 0,
            "best_run": {"scene_name": "run", "selection_status": "dry_run_smoke_demo"},
            "entries": [],
        },
    )
    (tmp_path / "run_comparison.md").write_text("# Pipeline Run Comparison\n", encoding="utf-8")
    _write_json(run_dir / "capture_manifest.json", {"scene_name": "run"})
    (run_dir / "capture_manifest.md").write_text("# Capture Manifest\n", encoding="utf-8")
    _write_json(run_dir / "capture_manifest_validation.json", {"status": "ready", "ok": True})
    (run_dir / "capture_manifest_validation.md").write_text("# Capture Validation\n", encoding="utf-8")
    _write_json(
        run_dir / "preflight_report.json",
        {"status": "ready", "fail_count": 0, "warn_count": 0},
    )
    (run_dir / "preflight_report.md").write_text("# Real-Run Preflight Report\n", encoding="utf-8")
    _write_json(
        run_dir / "evidence_scorecard.json",
        {"evidence_level": "dry_run_demo_ready", "score": 82, "max_score": 100, "overlay_count": 1},
    )
    (run_dir / "evidence_scorecard.md").write_text("# Evidence Scorecard\n", encoding="utf-8")
    _write_json(
        run_dir / "quality_gate.json",
        {"profile": "smoke", "status": "warn", "passed": True, "fail_count": 0, "warn_count": 3},
    )
    (run_dir / "quality_gate.md").write_text("# Run Quality Gate\n", encoding="utf-8")
    _write_json(
        run_dir / "claim_audit.json",
        {"status": "pass", "ok": True, "fail_count": 0, "warn_count": 0},
    )
    (run_dir / "claim_audit.md").write_text("# Claim Audit\n", encoding="utf-8")
    _write_json(
        run_dir / "run_result_card.json",
        {"result_status": "shareable_smoke_demo", "dry_run": True, "checks": []},
    )
    (run_dir / "run_result_card.md").write_text("# Run Result Card\n", encoding="utf-8")
    (run_dir / "portfolio_page.html").write_text("<!doctype html>\n", encoding="utf-8")
    _write_json(run_dir / "run_audit.json", {"status": "ready", "score": 100})
    _write_json(
        run_dir / "run_recommendations.json",
        {"readiness_level": "ready_for_portfolio", "recommendations": []},
    )
    (run_dir / "run_recommendations.md").write_text("# Run Recommendations\n", encoding="utf-8")
    _write_json(
        run_dir / "reproduction_manifest.json",
        {"scene_name": "run", "replay_command": "python scripts/run_scene_pipeline.py --dry-run"},
    )
    (run_dir / "reproduction_report.md").write_text("# Reproduction Report\n", encoding="utf-8")
    _write_json(run_dir / "research_report.json", {"scene_name": "run", "title": "Research"})
    (run_dir / "research_report.md").write_text("# Research Report\n", encoding="utf-8")
    _write_json(
        run_dir / "real_run_plan" / "real_run_plan.json",
        {"scene_name": "run", "current_mode": "dry-run smoke demo"},
    )
    (run_dir / "real_run_plan" / "real_run_plan.md").write_text(
        "# Real-Run Action Plan\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {"readiness_level": "shareable_smoke_demo"},
    )
    (run_dir / "submission_packet" / "submission_checklist.md").write_text(
        "# Submission Checklist\n",
        encoding="utf-8",
    )
    (run_dir / "submission_packet" / "cv_project_entry.md").write_text("# CV\n", encoding="utf-8")
    (run_dir / "submission_packet" / "professor_email_brief.md").write_text(
        "# Email\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "environment_report.json", {"ok": True})
    _write_json(run_dir / "scene_data_inspection.json", {"ready_for_training": True})
    _write_json(run_dir / "training" / "baseline_train_summary.json", {"run_type": "baseline"})
    _write_json(run_dir / "training" / "language_train_summary.json", {"run_type": "language"})
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True})
    _write_json(run_dir / "evaluation" / "annotation_review.json", {"ok": True, "items": []})
    (run_dir / "evaluation" / "annotation_review.md").write_text("# Annotation Review\n", encoding="utf-8")
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"top_k_hit_rate": 1.0})
    _write_json(
        run_dir / "prompt_sensitivity" / "prompt_sensitivity_summary.json",
        {"stable_group_count": 1, "num_groups": 1},
    )
    (run_dir / "prompt_sensitivity" / "prompt_sensitivity_report.md").write_text(
        "# Prompt Sensitivity Report\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "scene_relations" / "scene_relations_summary.json",
        {"scene_name": "run", "num_entities": 2, "num_relations": 1},
    )
    (run_dir / "scene_relations" / "scene_relations_report.md").write_text(
        "# Scene Relation Report\n",
        encoding="utf-8",
    )
    (run_dir / "scene_relations" / "scene_relations_edges.csv").write_text(
        "subject_label,relation,object_label\n"
        "desk,likely_supports,mug\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "logs" / "prepare_data_command.json", {"returncode": 0, "stdout": "ok"})
    (run_dir / "evaluation" / "eval_table.csv").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "eval_table.csv").write_text(
        "query,topk_hit\nmug,True\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "annotation_template.json", {"queries": [{"query": "mug"}]})
    (run_dir / "demo_assets" / "query_grid.png").write_bytes(b"placeholder")
    _write_json(
        run_dir / "queries" / "mug" / "scene_query_report.json",
        {"task": "mug", "answer": "Likely relevant scene regions are mug.", "query_results": []},
    )
    (run_dir / "queries" / "mug" / "scene_query_report.md").write_text("# Scene Query Report\n", encoding="utf-8")

    bundle = load_run_bundle(run_dir)

    assert bundle["pipeline_summary"]["success"] is True
    assert bundle["run_index"]["total_runs"] == 1
    assert bundle["run_comparison"]["best_run"]["scene_name"] == "run"
    assert "# Pipeline Run Comparison" in bundle["run_comparison_markdown"]
    assert bundle["capture_manifest"]["scene_name"] == "run"
    assert bundle["capture_manifest_validation"]["status"] == "ready"
    assert bundle["evaluation_table"][0]["query"] == "mug"
    assert bundle["annotation_template"]["queries"][0]["query"] == "mug"
    assert bundle["training_summaries"]["baseline"]["run_type"] == "baseline"
    assert bundle["training_summaries"]["language"]["run_type"] == "language"
    assert bundle["preflight_report"]["status"] == "ready"
    assert "# Real-Run Preflight Report" in bundle["preflight_markdown"]
    assert bundle["evidence_scorecard"]["evidence_level"] == "dry_run_demo_ready"
    assert "# Evidence Scorecard" in bundle["evidence_scorecard_markdown"]
    assert bundle["quality_gate"]["profile"] == "smoke"
    assert "# Run Quality Gate" in bundle["quality_gate_markdown"]
    assert bundle["claim_audit"]["status"] == "pass"
    assert "# Claim Audit" in bundle["claim_audit_markdown"]
    assert bundle["run_result_card"]["result_status"] == "shareable_smoke_demo"
    assert "# Run Result Card" in bundle["run_result_card_markdown"]
    assert bundle["portfolio_page"].endswith("portfolio_page.html")
    assert bundle["run_audit"]["status"] == "ready"
    assert bundle["run_recommendations"]["readiness_level"] == "ready_for_portfolio"
    assert "# Run Recommendations" in bundle["run_recommendations_markdown"]
    assert bundle["reproduction_manifest"]["scene_name"] == "run"
    assert "# Reproduction Report" in bundle["reproduction_report"]
    assert bundle["research_report"]["scene_name"] == "run"
    assert "# Research Report" in bundle["research_report_markdown"]
    assert bundle["real_run_plan"]["scene_name"] == "run"
    assert "# Real-Run Action Plan" in bundle["real_run_plan_markdown"]
    assert bundle["submission_packet"]["readiness_level"] == "shareable_smoke_demo"
    assert "# Submission Checklist" in bundle["submission_checklist"]
    assert bundle["annotation_validation"]["ok"] is True
    assert bundle["annotation_review"]["ok"] is True
    assert "# Annotation Review" in bundle["annotation_review_markdown"]
    assert bundle["prompt_sensitivity"]["stable_group_count"] == 1
    assert "# Prompt Sensitivity Report" in bundle["prompt_sensitivity_markdown"]
    assert bundle["scene_relations"]["num_relations"] == 1
    assert "# Scene Relation Report" in bundle["scene_relations_markdown"]
    assert bundle["scene_relations_table"][0]["relation"] == "likely_supports"
    assert bundle["command_logs"][0]["label"] == "logs/prepare_data_command.json"
    assert bundle["images"][0]["label"] == "demo_assets/query_grid.png"
    assert bundle["query_reports"][0]["kind"] == "scene_query_report"
    assert "# Scene Query Report" in bundle["query_reports"][0]["markdown"]
    assert bundle["missing"] == []


def test_dashboard_collectors_tolerate_missing_run(tmp_path: Path) -> None:
    missing_run = tmp_path / "missing"

    assert collect_run_images(missing_run) == []
    assert collect_query_reports(missing_run) == []
    assert collect_command_logs(missing_run) == []
    bundle = load_run_bundle(missing_run)
    assert "pipeline_summary.json" in bundle["missing"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
