import json
import subprocess
import sys
import zipfile
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack
from nerf_llm_scene_inspector.pipeline import PipelineConfig, run_scene_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_export_portfolio_pack_from_pipeline_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    prompt_suite = tmp_path / "prompt_suite.yaml"
    prompt_suite.write_text(
        "scene_name: export_scene\n"
        "groups:\n"
        "  - name: mug\n"
        "    prompts:\n"
        "      - mug\n"
        "      - coffee mug\n",
        encoding="utf-8",
    )
    annotations_path = ROOT / "examples" / "annotations_example.json"
    run_dir = tmp_path / "pipeline_runs" / "export_scene"
    output_dir = tmp_path / "portfolio_pack.bundle"

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
            prompt_suite_path=prompt_suite,
            config_path=config_path,
            dry_run=True,
            analyze_relations=True,
        )
    )
    assert summary.success is True
    (run_dir / "annotations_merged.json").write_text(
        json.dumps({"scene_name": "export_scene", "queries": []}),
        encoding="utf-8",
    )
    (run_dir / "annotation_merge_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (run_dir / "annotation_finalize_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (run_dir / "annotation_finalize_report.md").write_text("# Annotation Finalization Report\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_portfolio_pack.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_dir),
            "--zip",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    index = json.loads((output_dir / "portfolio_pack_index.json").read_text(encoding="utf-8"))
    assert index["missing"] == []
    assert index["review_checklist"] == "professor_review_checklist.md"
    copied_by_destination = {item["destination"]: item for item in index["copied"]}
    assert copied_by_destination["README.md"]["source"] == "generated"
    assert copied_by_destination["README.md"]["size_bytes"] > 0
    assert len(copied_by_destination["README.md"]["sha256"]) == 64
    assert copied_by_destination["professor_review_checklist.md"]["source"] == "generated"
    assert copied_by_destination["professor_review_checklist.md"]["size_bytes"] > 0
    assert len(copied_by_destination["professor_review_checklist.md"]["sha256"]) == 64
    assert copied_by_destination["run/pipeline_summary.json"]["size_bytes"] > 0
    assert len(copied_by_destination["run/pipeline_summary.json"]["sha256"]) == 64
    assert all("sha256" in item and "size_bytes" in item for item in index["copied"])
    assert index["run_summary"]["artifacts"]["baseline_train_summary"] == (
        "run/training/baseline_train_summary.json"
    )
    assert index["run_summary"]["artifacts"]["language_train_summary"] == (
        "run/training/language_train_summary.json"
    )
    assert index["run_summary"]["artifacts"]["capture_manifest"] == "run/capture_manifest.md"
    assert index["run_summary"]["artifacts"]["preflight_report"] == "run/preflight_report.md"
    assert index["run_summary"]["artifacts"]["evidence_scorecard"] == "run/evidence_scorecard.md"
    assert index["run_summary"]["artifacts"]["quality_gate"] == "run/quality_gate.md"
    assert index["run_summary"]["artifacts"]["claim_audit"] == "run/claim_audit.md"
    assert index["run_summary"]["artifacts"]["run_result_card"] == "run/run_result_card.md"
    assert index["run_summary"]["artifacts"]["query_reports"] == "run/queries/"
    assert index["run_summary"]["artifacts"]["prompt_sensitivity"] == "run/prompt_sensitivity/"
    assert index["run_summary"]["artifacts"]["scene_relations"] == "run/scene_relations/"
    assert index["run_summary"]["artifacts"]["run_comparison"] == "run_comparison.md"
    assert index["run_summary"]["artifacts"]["portfolio_page"] == "run/portfolio_page.html"
    assert index["run_summary"]["artifacts"]["annotation_review"] == "run/evaluation/annotation_review.md"
    assert index["run_summary"]["artifacts"]["annotation_workbench"] == (
        "run/evaluation/annotation_workbench/annotation_workbench.html"
    )
    assert index["run_summary"]["artifacts"]["annotations_merged"] == "run/annotations_merged.json"
    assert index["run_summary"]["artifacts"]["annotation_merge_report"] == "run/annotation_merge_report.json"
    assert index["run_summary"]["artifacts"]["annotation_finalize"] == "run/annotation_finalize_report.md"
    assert index["run_summary"]["artifacts"]["research_report"] == "run/research_report.md"
    assert index["run_summary"]["artifacts"]["real_run_plan"] == "run/real_run_plan/real_run_plan.md"
    assert index["run_summary"]["artifacts"]["submission_checklist"] == (
        "run/submission_packet/submission_checklist.md"
    )
    assert index["run_summary"]["scene_name"] == "export_scene"
    assert str(tmp_path) not in json.dumps(index)
    packed_summary = (output_dir / "run" / "pipeline_summary.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in packed_summary
    pack_readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "# NeRF-LLM Scene Inspector Portfolio Pack" in pack_readme
    assert "CPU dry-run smoke demo" in pack_readme
    assert "project/docs/index.html" in pack_readme
    assert "run/portfolio_page.html" in pack_readme
    assert "professor_review_checklist.md" in pack_readme
    assert "state-of-the-art benchmark performance" in pack_readme
    assert str(tmp_path) not in pack_readme
    review_checklist = (output_dir / "professor_review_checklist.md").read_text(encoding="utf-8")
    assert "# Professor Review Checklist" in review_checklist
    assert "Five-Minute Review Path" in review_checklist
    assert "CPU dry-run smoke demo" in review_checklist
    assert "not trained LERF outputs from a real scene" in review_checklist
    assert "run/submission_packet/submission_checklist.md" in review_checklist
    assert str(tmp_path) not in review_checklist
    packed_baseline_summary = (
        output_dir / "run" / "training" / "baseline_train_summary.json"
    ).read_text(encoding="utf-8")
    assert str(tmp_path) not in packed_baseline_summary
    packed_language_summary = (
        output_dir / "run" / "training" / "language_train_summary.json"
    ).read_text(encoding="utf-8")
    assert str(tmp_path) not in packed_language_summary
    packed_log = (output_dir / "run" / "logs" / "prepare_data_command.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in packed_log
    demo_log = json.loads(
        (output_dir / "run" / "logs" / "generate_demo_assets_command.json").read_text(encoding="utf-8")
    )
    assert demo_log["command"][0] == "python"
    assert (output_dir / "run" / "pipeline_summary.json").exists()
    assert (output_dir / "project" / "docs" / "index.html").exists()
    assert (output_dir / "run" / "capture_manifest.json").exists()
    assert (output_dir / "run" / "capture_manifest_validation.json").exists()
    assert (output_dir / "run" / "preflight_report.json").exists()
    assert (output_dir / "run" / "preflight_report.md").exists()
    assert (output_dir / "run" / "evidence_scorecard.json").exists()
    assert (output_dir / "run" / "evidence_scorecard.md").exists()
    assert (output_dir / "run" / "quality_gate.json").exists()
    assert (output_dir / "run" / "quality_gate.md").exists()
    assert (output_dir / "run" / "claim_audit.json").exists()
    assert (output_dir / "run" / "claim_audit.md").exists()
    assert (output_dir / "run" / "run_result_card.json").exists()
    assert (output_dir / "run" / "run_result_card.md").exists()
    assert (output_dir / "run" / "portfolio_page.html").exists()
    assert str(tmp_path) not in (output_dir / "run" / "portfolio_page.html").read_text(encoding="utf-8")
    assert (output_dir / "run" / "training" / "baseline_train_summary.json").exists()
    assert (output_dir / "run" / "training" / "language_train_summary.json").exists()
    assert (output_dir / "run_index.json").exists()
    assert (output_dir / "run_index.md").exists()
    assert (output_dir / "run_comparison.json").exists()
    assert (output_dir / "run_comparison.md").exists()
    assert (output_dir / "run" / "logs" / "prepare_data_command.json").exists()
    assert (output_dir / "run" / "logs" / "generate_demo_assets_command.json").exists()
    assert (output_dir / "run" / "run_audit.json").exists()
    assert (output_dir / "run" / "run_audit.md").exists()
    assert (output_dir / "run" / "run_recommendations.json").exists()
    assert (output_dir / "run" / "run_recommendations.md").exists()
    assert (output_dir / "run" / "reproduction_manifest.json").exists()
    assert (output_dir / "run" / "reproduction_report.md").exists()
    assert (output_dir / "run" / "reproduce_run.sh").exists()
    assert (output_dir / "run" / "research_report.json").exists()
    assert (output_dir / "run" / "research_report.md").exists()
    assert (output_dir / "run" / "real_run_plan" / "real_run_plan.json").exists()
    assert (output_dir / "run" / "real_run_plan" / "real_run_plan.md").exists()
    assert (output_dir / "run" / "submission_packet" / "submission_packet.json").exists()
    assert (output_dir / "run" / "submission_packet" / "submission_checklist.md").exists()
    assert (output_dir / "run" / "submission_packet" / "cv_project_entry.md").exists()
    assert (output_dir / "run" / "submission_packet" / "professor_email_brief.md").exists()
    assert (output_dir / "run" / "annotation_template.json").exists()
    assert (output_dir / "run" / "annotations_merged.json").exists()
    assert (output_dir / "run" / "annotation_merge_report.json").exists()
    assert (output_dir / "run" / "annotation_finalize_report.json").exists()
    assert (output_dir / "run" / "annotation_finalize_report.md").exists()
    assert (output_dir / "run" / "queries" / "mug" / "scene_query_report.json").exists()
    assert (output_dir / "run" / "queries" / "mug" / "scene_query_report.md").exists()
    assert (output_dir / "run" / "prompt_sensitivity" / "prompt_sensitivity_summary.json").exists()
    assert (output_dir / "run" / "prompt_sensitivity" / "prompt_sensitivity_report.md").exists()
    assert (output_dir / "run" / "scene_relations" / "scene_relations_summary.json").exists()
    assert (output_dir / "run" / "scene_relations" / "scene_relations_edges.csv").exists()
    assert (output_dir / "run" / "scene_relations" / "scene_relations_report.md").exists()
    assert (output_dir / "run" / "project_report.md").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_validation.json").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_review.json").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_review.md").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_review_contact_sheet.png").exists()
    assert (output_dir / "run" / "evaluation" / "annotation_workbench" / "annotation_workbench.html").exists()
    assert (
        output_dir
        / "run"
        / "evaluation"
        / "annotation_workbench"
        / "annotation_workbench_manifest.json"
    ).exists()
    assert (output_dir / "run" / "evaluation" / "annotation_workbench" / "annotation_seed.json").exists()
    assert list((output_dir / "run" / "evaluation" / "annotation_workbench" / "assets").glob("*"))
    assert (output_dir / "run" / "evaluation" / "eval_summary.json").exists()
    assert (output_dir / "run" / "demo_assets" / "query_grid.png").exists()
    validation = validate_portfolio_pack(output_dir)
    assert validation.path_leaks == []
    assert validation.ok is True, validation.to_dict()
    assert validation.artifact_issues == []
    archive_path = Path(f"{output_dir}.zip")
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "README.md" in names
        assert "professor_review_checklist.md" in names
        assert "portfolio_pack_index.json" in names
        assert "run/pipeline_summary.json" in names
        zipped_index = json.loads(archive.read("portfolio_pack_index.json").decode("utf-8"))
        zipped_readme = archive.read("README.md").decode("utf-8")
        zipped_review = archive.read("professor_review_checklist.md").decode("utf-8")
    zipped_copied = {item["destination"]: item for item in zipped_index["copied"]}
    assert zipped_index["archive"].endswith("portfolio_pack.bundle.zip")
    assert zipped_index["review_checklist"] == "professor_review_checklist.md"
    assert "portfolio_pack.bundle.zip" in zipped_readme
    assert "portfolio_pack.bundle.zip" in zipped_review
    assert len(zipped_copied["README.md"]["sha256"]) == 64
    assert zipped_copied["README.md"]["size_bytes"] > 0
    assert len(zipped_copied["professor_review_checklist.md"]["sha256"]) == 64
    assert zipped_copied["professor_review_checklist.md"]["size_bytes"] > 0
    assert len(zipped_copied["run/pipeline_summary.json"]["sha256"]) == 64
    assert zipped_copied["run/pipeline_summary.json"]["size_bytes"] > 0


def test_export_portfolio_pack_refreshes_existing_external_pack(tmp_path: Path) -> None:
    output_dir = tmp_path / "portfolio_pack"
    output_dir.mkdir()
    (output_dir / "portfolio_pack_index.json").write_text("{}", encoding="utf-8")
    (output_dir / "stale.txt").write_text("stale", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_portfolio_pack.py"),
            "--output",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "portfolio_pack_index.json").exists()
    assert not (output_dir / "stale.txt").exists()
