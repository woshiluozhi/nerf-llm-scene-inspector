import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.run_comparison import compare_pipeline_runs


ROOT = Path(__file__).resolve().parents[1]


def test_compare_pipeline_runs_prefers_real_portfolio_candidate(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "dry_demo",
        scene_name="dry_demo",
        dry_run=True,
        evidence_level="dry_run_demo_ready",
        evidence_score=95,
        evidence_max_score=110,
        audit_status="needs_review",
        audit_score=92,
        capture_status="needs_review",
    )
    _write_run(
        root / "real_scene",
        scene_name="real_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=110,
        audit_status="ready",
        audit_score=98,
        capture_status="ready",
    )

    comparison = compare_pipeline_runs(root)

    assert comparison.total_runs == 2
    assert comparison.portfolio_candidate_count == 1
    assert comparison.entries[0].scene_name == "real_scene"
    assert comparison.entries[0].selection_status == "portfolio_candidate"
    assert comparison.entries[0].result_status == "portfolio_ready"
    assert comparison.entries[0].submission_readiness_level == "portfolio_ready"
    assert comparison.entries[1].selection_status == "dry_run_smoke_demo"
    assert comparison.best_run is not None
    assert comparison.best_run["scene_name"] == "real_scene"


def test_compare_pipeline_runs_blocks_unready_real_capture(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "real_review",
        scene_name="real_review",
        dry_run=False,
        evidence_level="needs_review",
        evidence_score=84,
        evidence_max_score=110,
        audit_status="needs_review",
        audit_score=82,
        capture_status="needs_review",
    )

    comparison = compare_pipeline_runs(root)

    assert comparison.entries[0].selection_status == "blocked"
    assert comparison.portfolio_candidate_count == 0


def test_compare_pipeline_runs_demotes_query_risk_flags(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "risk_scene",
        scene_name="risk_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=100,
        audit_status="ready",
        audit_score=100,
        capture_status="ready",
        query_risk_flags=2,
    )

    comparison = compare_pipeline_runs(root)

    entry = comparison.entries[0]
    assert entry.selection_status == "needs_review"
    assert entry.query_evidence_status == "warn"
    assert entry.query_risk_flag_count == 2
    assert comparison.portfolio_candidate_count == 0
    assert comparison.best_run is not None
    assert comparison.best_run["query_risk_flag_count"] == 2


def test_compare_pipeline_runs_demotes_unvalidated_submission(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "unvalidated_scene",
        scene_name="unvalidated_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=100,
        audit_status="ready",
        audit_score=100,
        capture_status="ready",
        result_status="real_run_review_ready",
        submission_readiness="needs_pack_validation",
    )

    comparison = compare_pipeline_runs(root)

    entry = comparison.entries[0]
    assert entry.selection_status == "needs_review"
    assert entry.result_status == "real_run_review_ready"
    assert entry.submission_readiness_level == "needs_pack_validation"
    assert comparison.portfolio_candidate_count == 0


def test_compare_pipeline_runs_blocks_result_card_failures(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "blocked_scene",
        scene_name="blocked_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=100,
        audit_status="ready",
        audit_score=100,
        capture_status="ready",
        result_status="blocked",
        submission_readiness="blocked",
    )

    comparison = compare_pipeline_runs(root)

    entry = comparison.entries[0]
    assert entry.selection_status == "blocked"
    assert entry.portfolio_score <= 25.0
    assert comparison.portfolio_candidate_count == 0


def test_compare_pipeline_runs_blocks_run_audit_blocker_count(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "stale_audit_scene",
        scene_name="stale_audit_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=100,
        audit_status="ready",
        audit_score=100,
        audit_blockers=1,
        capture_status="ready",
    )

    comparison = compare_pipeline_runs(root)

    entry = comparison.entries[0]
    assert entry.selection_status == "blocked"
    assert entry.audit_status == "ready"
    assert entry.audit_blocker_count == 1
    assert comparison.portfolio_candidate_count == 0


def test_compare_pipeline_runs_blocks_capture_manifest_fail_count(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "stale_capture_scene",
        scene_name="stale_capture_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=100,
        audit_status="ready",
        audit_score=100,
        capture_status="ready",
        capture_fail_count=1,
    )

    comparison = compare_pipeline_runs(root)

    entry = comparison.entries[0]
    assert entry.selection_status == "blocked"
    assert entry.capture_manifest_status == "ready"
    assert entry.capture_manifest_fail_count == 1
    assert comparison.portfolio_candidate_count == 0


def test_compare_pipeline_runs_blocks_failure_diagnostics_blocker_count(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "stale_diagnostics_scene",
        scene_name="stale_diagnostics_scene",
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        evidence_score=100,
        evidence_max_score=100,
        audit_status="ready",
        audit_score=100,
        capture_status="ready",
        diagnostics_status="clear",
        diagnostics_blockers=1,
    )

    comparison = compare_pipeline_runs(root)

    entry = comparison.entries[0]
    assert entry.selection_status == "blocked"
    assert entry.failure_diagnostics_status == "clear"
    assert entry.failure_diagnostics_blocker_count == 1
    assert comparison.portfolio_candidate_count == 0


def test_compare_runs_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "dry_demo",
        scene_name="dry_demo",
        dry_run=True,
        evidence_level="dry_run_demo_ready",
        evidence_score=82,
        evidence_max_score=110,
        audit_status="needs_review",
        audit_score=72,
        capture_status="needs_review",
    )
    output = tmp_path / "comparison.json"
    markdown = tmp_path / "comparison.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_runs.py"),
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
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_run"]["selection_status"] == "dry_run_smoke_demo"
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "# Pipeline Run Comparison" in markdown_text
    assert "Risk Flags" in markdown_text
    assert "Submission" in markdown_text
    assert "Diagnostic Blockers" in markdown_text


def test_compare_runs_cli_strict_requires_real_candidate(tmp_path: Path) -> None:
    root = tmp_path / "pipeline_runs"
    _write_run(
        root / "dry_demo",
        scene_name="dry_demo",
        dry_run=True,
        evidence_level="dry_run_demo_ready",
        evidence_score=82,
        evidence_max_score=110,
        audit_status="needs_review",
        audit_score=72,
        capture_status="needs_review",
    )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--root", str(root), "--strict"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1


def _write_run(
    run_dir: Path,
    *,
    scene_name: str,
    dry_run: bool,
    evidence_level: str,
    evidence_score: int,
    evidence_max_score: int,
    audit_status: str,
    audit_score: int,
    capture_status: str,
    audit_blockers: int = 0,
    capture_fail_count: int = 0,
    diagnostics_status: str = "clear",
    diagnostics_blockers: int = 0,
    query_risk_flags: int = 0,
    result_status: str | None = None,
    submission_readiness: str | None = None,
) -> None:
    if result_status is None:
        result_status = "shareable_smoke_demo" if dry_run else "portfolio_ready"
    if submission_readiness is None:
        submission_readiness = "shareable_smoke_demo" if dry_run else "portfolio_ready"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": scene_name,
            "success": True,
            "dry_run": dry_run,
            "backend": "lerf",
            "timestamp": "2026-06-08T00:00:00+00:00",
            "queries": ["mug", "laptop", "objects that can hold water"],
        },
    )
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "evidence_level": evidence_level,
            "score": evidence_score,
            "max_score": evidence_max_score,
        },
    )
    _write_json(
        run_dir / "run_audit.json",
        {"status": audit_status, "score": audit_score, "blocker_count": audit_blockers},
    )
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": capture_status, "fail_count": capture_fail_count},
    )
    _write_json(
        run_dir / "failure_diagnostics.json",
        {"status": diagnostics_status, "blocker_count": diagnostics_blockers, "warning_count": 0},
    )
    _write_json(run_dir / "run_result_card.json", {"result_status": result_status})
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {"readiness_level": submission_readiness},
    )
    _write_json(
        run_dir / "run_recommendations.json",
        {"top_next_action": "Review warnings before sharing."},
    )
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn" if query_risk_flags else "pass",
            "ok": True,
            "task_count": 1,
            "fail_count": 0,
            "totals": {
                "counter_evidence_count": 1 if query_risk_flags else 0,
                "risk_flag_count": query_risk_flags,
            },
        },
    )
    _write_json(run_dir / "scene_data_inspection.json", {"quality_score": 0.92, "pose_coverage_score": 1.0})
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {
            "num_evaluated_queries": 3,
            "top_k_hit_rate": 1.0,
            "mean_iou_2d": 0.72,
            "average_relevancy_score": 0.8,
        },
    )
    (run_dir / "portfolio_page.html").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "portfolio_page.html").write_text("<!doctype html>\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
