import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack


ROOT = Path(__file__).resolve().parents[1]


def test_validate_portfolio_pack_accepts_complete_pack(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)

    report = validate_portfolio_pack(pack)

    assert report.ok is True
    assert report.missing_files == []
    assert report.artifact_issues == []
    assert report.path_leaks == []
    assert "portfolio_pack_index.json" in report.checked_files


def test_validate_portfolio_pack_fails_missing_artifact(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)
    (pack / "run" / "demo_assets" / "query_grid.png").unlink()

    report = validate_portfolio_pack(pack)

    assert report.ok is False
    assert "run/demo_assets/query_grid.png" in report.missing_files
    assert any("demo_grid" in issue for issue in report.artifact_issues)


def test_validate_portfolio_pack_fails_path_leak(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)
    (pack / "run" / "pipeline_summary.json").write_text(
        json.dumps({"source": "C:\\Users\\lz\\private\\video.mp4"}),
        encoding="utf-8",
    )

    report = validate_portfolio_pack(pack)

    assert report.ok is False
    assert report.path_leaks
    assert report.path_leaks[0].file == "run/pipeline_summary.json"


def test_validate_portfolio_pack_fails_quality_gate_failure(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)
    (pack / "run" / "quality_gate.json").write_text(
        json.dumps({"profile": "smoke", "status": "fail", "passed": False}),
        encoding="utf-8",
    )

    report = validate_portfolio_pack(pack)

    assert report.ok is False
    assert "quality_gate.json reports a failed run quality gate." in report.errors


def test_validate_portfolio_pack_fails_claim_audit_failure(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)
    (pack / "run" / "claim_audit.json").write_text(
        json.dumps({"status": "fail", "ok": False, "fail_count": 1}),
        encoding="utf-8",
    )

    report = validate_portfolio_pack(pack)

    assert report.ok is False
    assert "claim_audit.json reports unsupported external-facing claims." in report.errors


def test_validate_portfolio_pack_fails_blocked_result_card(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)
    (pack / "run" / "run_result_card.json").write_text(
        json.dumps({"result_status": "blocked"}),
        encoding="utf-8",
    )

    report = validate_portfolio_pack(pack)

    assert report.ok is False
    assert "run_result_card.json result_status is blocked." in report.errors


def test_validate_portfolio_pack_cli_writes_report(tmp_path: Path) -> None:
    pack = _write_complete_pack(tmp_path)
    output = tmp_path / "validation.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_portfolio_pack.py"),
            "--pack",
            str(pack),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def _write_complete_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "portfolio_pack"
    files = [
        "project/README.md",
        "project/LICENSE",
        "project/CITATION.cff",
        "project/docs/index.html",
        "project/docs/portfolio_result_card.md",
        "project/docs/project_report.md",
        "project/docs/method_summary.md",
        "project/docs/cv_bullets.md",
        "project/docs/cold_email_paragraph.md",
        "project/docs/real_scene_capture_checklist.md",
        "project/docs/real_run_reproducibility.md",
        "project/docs/assets/query_grid.png",
        "project/docs/assets/demo_montage.gif",
        "run/pipeline_summary.json",
        "run/capture_manifest.json",
        "run/capture_manifest.md",
        "run/capture_manifest_validation.json",
        "run/capture_manifest_validation.md",
        "run/preflight_report.json",
        "run/preflight_report.md",
        "run/evidence_scorecard.json",
        "run/evidence_scorecard.md",
        "run/quality_gate.json",
        "run/quality_gate.md",
        "run/claim_audit.json",
        "run/claim_audit.md",
        "run/run_result_card.json",
        "run/run_result_card.md",
        "run/run_audit.json",
        "run/run_audit.md",
        "run/run_recommendations.json",
        "run/run_recommendations.md",
        "run/research_report.json",
        "run/research_report.md",
        "run/real_run_plan/real_run_plan.json",
        "run/real_run_plan/real_run_plan.md",
        "run/submission_packet/submission_packet.json",
        "run/submission_packet/submission_checklist.md",
        "run/submission_packet/cv_project_entry.md",
        "run/submission_packet/professor_email_brief.md",
        "run/reproduction_manifest.json",
        "run/reproduction_report.md",
        "run/reproduce_run.sh",
        "run/environment_report.json",
        "run/scene_data_inspection.json",
        "run/scene_data_inspection.md",
        "run/queries.yaml",
        "run/queries/mug/scene_query_report.json",
        "run/queries/mug/scene_query_report.md",
        "run/annotation_template.json",
        "run/project_report.md",
        "run/portfolio_result_card.md",
        "run/portfolio_page.html",
        "run/evaluation/eval_summary.json",
        "run/evaluation/eval_table.csv",
        "run/evaluation/annotation_validation.json",
        "run/evaluation/annotation_review.json",
        "run/evaluation/annotation_review.md",
        "run/evaluation/annotation_review_contact_sheet.png",
        "run/evaluation/qualitative_report.md",
        "run/demo_assets/demo_summary.json",
        "run/demo_assets/query_grid.png",
        "run/demo_assets/demo_montage.gif",
        "run/logs/prepare_data_command.json",
        "run/logs/query_scene_command.json",
    ]
    for relative_path in files:
        path = pack / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".png", ".gif"}:
            path.write_bytes(b"not-real-image")
        else:
            path.write_text(_file_payload(relative_path), encoding="utf-8")
    index = {
        "copied": [{"source": relative_path, "destination": relative_path} for relative_path in files],
        "missing": [],
        "optional_missing": [],
        "github": "https://github.com/woshiluozhi/nerf-llm-scene-inspector",
        "run_dir": "results/pipeline_runs/desk_scene",
        "run_summary": {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": True,
            "backend": "lerf",
            "artifacts": {
                "pipeline_summary": "run/pipeline_summary.json",
                "capture_manifest": "run/capture_manifest.md",
                "capture_manifest_validation": "run/capture_manifest_validation.md",
                "preflight_report": "run/preflight_report.md",
                "evidence_scorecard": "run/evidence_scorecard.md",
                "quality_gate": "run/quality_gate.md",
                "claim_audit": "run/claim_audit.md",
                "run_result_card": "run/run_result_card.md",
                "portfolio_page": "run/portfolio_page.html",
                "run_index": "run_index.md",
                "run_audit": "run/run_audit.md",
                "run_recommendations": "run/run_recommendations.md",
                "research_report": "run/research_report.md",
                "real_run_plan": "run/real_run_plan/real_run_plan.md",
                "submission_checklist": "run/submission_packet/submission_checklist.md",
                "reproduction_report": "run/reproduction_report.md",
                "reproduce_script": "run/reproduce_run.sh",
                "command_logs": "run/logs/",
                "query_reports": "run/queries/",
                "evaluation_summary": "run/evaluation/eval_summary.json",
                "annotation_validation": "run/evaluation/annotation_validation.json",
                "annotation_review": "run/evaluation/annotation_review.md",
                "annotation_review_contact_sheet": "run/evaluation/annotation_review_contact_sheet.png",
                "demo_grid": "run/demo_assets/query_grid.png",
            },
        },
    }
    (pack / "run_index.md").write_text("# Run Index\n", encoding="utf-8")
    index["copied"].append({"source": "run_index.md", "destination": "run_index.md"})
    (pack / "portfolio_pack_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return pack


def _file_payload(relative_path: str) -> str:
    if relative_path.endswith("run_audit.json"):
        return json.dumps({"status": "ready", "score": 100})
    if relative_path.endswith("run_recommendations.json"):
        return json.dumps({"readiness_level": "ready_for_portfolio", "recommendations": []})
    if relative_path.endswith("research_report.json"):
        return json.dumps({"scene_name": "desk_scene", "title": "NeRF-LLM Scene Inspector Research Report"})
    if relative_path.endswith("real_run_plan.json"):
        return json.dumps({"scene_name": "desk_scene", "current_mode": "dry-run smoke demo"})
    if relative_path.endswith("submission_packet.json"):
        return json.dumps({"scene_name": "desk_scene", "readiness_level": "shareable_smoke_demo"})
    if relative_path.endswith("reproduction_manifest.json"):
        return json.dumps({"scene_name": "desk_scene", "replay_command": "python scripts/run_scene_pipeline.py --dry-run"})
    if relative_path.endswith("preflight_report.json"):
        return json.dumps({"status": "ready", "ready_for_real_run": True})
    if relative_path.endswith("capture_manifest_validation.json"):
        return json.dumps({"status": "ready", "ok": True})
    if relative_path.endswith("evidence_scorecard.json"):
        return json.dumps({"evidence_level": "dry_run_demo_ready", "dry_run": True, "score": 82})
    if relative_path.endswith("quality_gate.json"):
        return json.dumps({"profile": "smoke", "status": "pass", "passed": True})
    if relative_path.endswith("claim_audit.json"):
        return json.dumps({"status": "pass", "ok": True, "fail_count": 0, "warn_count": 0})
    if relative_path.endswith("run_result_card.json"):
        return json.dumps({"result_status": "portfolio_ready", "dry_run": False})
    if relative_path.endswith("annotation_validation.json"):
        return json.dumps({"ok": True, "warnings": []})
    if relative_path.endswith("pipeline_summary.json"):
        return json.dumps({"success": True, "scene_name": "desk_scene"})
    if relative_path.endswith("scene_query_report.json"):
        return json.dumps({"scene_name": "desk_scene", "answer": "Likely relevant scene regions are mug."})
    return f"placeholder for {relative_path}\n"
