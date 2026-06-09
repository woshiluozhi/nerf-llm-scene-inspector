import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.run_index import index_pipeline_runs


ROOT = Path(__file__).resolve().parents[1]


def test_index_pipeline_runs_summarizes_runs(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(root / "scene_a", scene_name="scene_a", audit_status="ready", score=96, dry_run=False)
    _write_run(root / "scene_b", scene_name="scene_b", audit_status="needs_review", score=80)

    index = index_pipeline_runs(root)

    assert index.total_runs == 2
    assert index.successful_runs == 2
    assert index.ready_runs == 1
    assert index.entries[0].scene_name in {"scene_a", "scene_b"}
    assert index.entries[0].artifacts["pipeline_summary"] == "pipeline_summary.json"
    assert index.entries[0].artifacts["capture_manifest"] == "capture_manifest.md"
    assert index.entries[0].artifacts["portfolio_page"] == "portfolio_page.html"
    assert index.entries[0].artifacts["annotation_review"] == "evaluation/annotation_review.md"
    assert index.entries[0].result_status == "portfolio_ready"
    assert index.entries[0].submission_readiness_level == "portfolio_ready"
    assert index.entries[0].query_evidence_status == "pass"
    assert index.entries[0].query_risk_flag_count == 0
    assert index.entries[0].audit_blocker_count == 0
    assert index.entries[0].failure_diagnostics_status == "clear"
    assert index.entries[0].failure_diagnostics_blocker_count == 0
    assert index.entries[0].capture_manifest_status == "ready"
    assert index.entries[0].capture_manifest_fail_count == 0


def test_index_pipeline_runs_exposes_audit_and_capture_counts(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "stale_scene",
        scene_name="stale_scene",
        audit_status="ready",
        score=96,
        audit_blockers=1,
        capture_fail_count=1,
    )

    index = index_pipeline_runs(root)

    entry = index.entries[0]
    assert index.ready_runs == 0
    assert entry.audit_status == "ready"
    assert entry.audit_blocker_count == 1
    assert entry.blocker_count == 1
    assert entry.capture_manifest_status == "ready"
    assert entry.capture_manifest_fail_count == 1


def test_index_pipeline_runs_ready_runs_require_external_sharing_gates(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "stale_scene",
        scene_name="stale_scene",
        audit_status="ready",
        score=96,
        dry_run=False,
        failure_blockers=1,
    )

    index = index_pipeline_runs(root)

    entry = index.entries[0]
    assert index.ready_runs == 0
    assert entry.success is True
    assert entry.dry_run is False
    assert entry.result_status == "portfolio_ready"
    assert entry.submission_readiness_level == "portfolio_ready"
    assert entry.failure_diagnostics_status == "clear"
    assert entry.failure_diagnostics_blocker_count == 1


def test_index_runs_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(root / "scene_a", scene_name="scene_a", audit_status="ready", score=100, dry_run=False)
    output = tmp_path / "index.json"
    markdown = tmp_path / "index.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "index_runs.py"),
            "--root",
            str(root),
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

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["total_runs"] == 1
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "# Pipeline Run Index" in markdown_text
    assert "Result" in markdown_text
    assert "Submission" in markdown_text
    assert "Audit Blockers" in markdown_text
    assert "Diagnostic Blockers" in markdown_text
    assert "Capture Fails" in markdown_text


def _write_run(
    run_dir: Path,
    *,
    scene_name: str,
    audit_status: str,
    score: int,
    audit_blockers: int = 0,
    capture_status: str = "ready",
    capture_fail_count: int = 0,
    failure_blockers: int = 0,
    dry_run: bool = True,
) -> None:
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": scene_name,
            "success": True,
            "dry_run": dry_run,
            "backend": "lerf",
            "timestamp": f"2026-06-08T00:0{score % 10}:00+00:00",
            "queries": ["mug"],
        },
    )
    _write_json(
        run_dir / "run_audit.json",
        {
            "status": audit_status,
            "score": score,
            "blocker_count": audit_blockers,
            "warning_count": 0 if audit_status == "ready" else 1,
        },
    )
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": capture_status, "fail_count": capture_fail_count},
    )
    _write_json(
        run_dir / "failure_diagnostics.json",
        {"status": "clear", "blocker_count": failure_blockers, "warning_count": 0},
    )
    _write_json(run_dir / "scene_data_inspection.json", {"quality_score": 0.9, "pose_coverage_score": 1.0})
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {"num_evaluated_queries": 1, "top_k_hit_rate": 1.0, "mean_iou_2d": 0.7},
    )
    _write_json(
        run_dir / "query_evidence_audit.json",
        {"status": "pass", "ok": True, "task_count": 1, "fail_count": 0, "totals": {}},
    )
    _write_json(
        run_dir / "run_result_card.json",
        {"result_status": "shareable_smoke_demo" if dry_run else "portfolio_ready"},
    )
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {"readiness_level": "shareable_smoke_demo" if dry_run else "portfolio_ready"},
    )
    (run_dir / "run_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_audit.md").write_text("# Audit\n", encoding="utf-8")
    (run_dir / "capture_manifest.md").write_text("# Capture\n", encoding="utf-8")
    (run_dir / "capture_manifest_validation.md").write_text("# Capture Validation\n", encoding="utf-8")
    (run_dir / "portfolio_page.html").write_text("<!doctype html>\n", encoding="utf-8")
    (run_dir / "evidence_scorecard.md").write_text("# Scorecard\n", encoding="utf-8")
    (run_dir / "run_recommendations.md").write_text("# Recommendations\n", encoding="utf-8")
    (run_dir / "evaluation" / "annotation_review.md").write_text("# Annotation Review\n", encoding="utf-8")
    (run_dir / "evaluation" / "annotation_review_contact_sheet.png").write_text("image", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
