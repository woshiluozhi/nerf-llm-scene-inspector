import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.visualization.portfolio_page import build_portfolio_page


ROOT = Path(__file__).resolve().parents[1]


def test_build_portfolio_page_uses_relative_links(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    page = build_portfolio_page(run_dir)
    html = page.to_html()

    assert "desk_scene" in html
    assert "dry_run_demo_ready" in html
    assert "shareable_smoke_demo" in html
    assert "Capture manifest" in html
    assert "Capture failures" in html
    assert "Query evidence" in html
    assert "Query risk flags" in html
    assert "counter-evidence" in html
    assert "demo_assets/query_grid.png" in html
    assert "quality_gate.md" in html
    assert "failure_diagnostics.md" in html
    assert "run_readiness.md" in html
    assert "run_result_card.md" in html
    assert "annotation_workbench/annotation_workbench.html" in html
    assert "research_report.md" in html
    assert "submission_packet/submission_checklist.md" in html
    assert "Sharing Readiness" in html
    assert "needs_pack_validation" in html
    assert "Finalize annotations with --export-pack --zip-pack." in html
    assert "portfolio_pack" in html
    assert page.capture_manifest_status == "needs_review"
    assert page.capture_manifest_fail_count == 0
    assert str(tmp_path) not in html
    assert "C:\\Users" not in html


def test_portfolio_page_surfaces_blocked_query_risk_flags(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    _write_json(run_dir / "run_result_card.json", {"result_status": "blocked"})
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "totals": {"counter_evidence_count": 1, "risk_flag_count": 2},
            "tasks": [{"task": "safe place", "counter_evidence_count": 1, "risk_flag_count": 2}],
        },
    )
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "readiness_level": "blocked",
            "pack_ok": True,
            "readiness_summary": {
                "status": "fail",
                "readiness_level": "blocked",
                "failed_check_count": 1,
                "warning_check_count": 0,
                "packet_warning_count": 1,
                "failed_checks": ["query_evidence"],
                "warning_checks": [],
                "top_blockers": ["query_evidence: risk_flags=2, counter_evidence=1"],
                "top_warnings": [],
                "pack_ok": True,
                "query_evidence_status": "warn",
                "query_counter_evidence_count": 1,
                "query_risk_flag_count": 2,
                "recommended_next_action": "Resolve query risk flags before external sharing.",
            },
        },
    )

    page = build_portfolio_page(run_dir)
    html = page.to_html()

    assert page.result_status == "blocked"
    assert page.query_evidence_status == "warn"
    assert page.query_counter_evidence_count == 1
    assert page.query_risk_flag_count == 2
    assert "This run is blocked for external sharing" in html
    assert "risk_flags=2" in html
    assert "Resolve query risk flags before external sharing." in html
    assert "C:\\Users" not in html


def test_portfolio_page_blocks_stale_result_card_on_capture_failures(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False)
    _write_json(run_dir / "run_result_card.json", {"result_status": "portfolio_ready"})
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "readiness_level": "portfolio_ready",
            "pack_ok": True,
            "readiness_summary": {
                "status": "pass",
                "readiness_level": "portfolio_ready",
                "failed_check_count": 0,
                "warning_check_count": 0,
                "packet_warning_count": 0,
                "failed_checks": [],
                "warning_checks": [],
                "top_blockers": [],
                "top_warnings": [],
                "pack_ok": True,
                "capture_manifest_status": "ready",
                "capture_manifest_fail_count": 0,
                "query_evidence_status": "pass",
                "query_counter_evidence_count": 0,
                "query_risk_flag_count": 0,
                "recommended_next_action": "Share with recorded limitations.",
            },
        },
    )
    _write_json(run_dir / "capture_manifest_validation.json", {"status": "ready", "fail_count": "1", "warn_count": 0})

    page = build_portfolio_page(run_dir)
    html = page.to_html()

    assert page.result_status == "blocked"
    assert page.sharing_readiness["status"] == "fail"
    assert page.sharing_readiness["readiness_level"] == "blocked"
    assert page.capture_manifest_status == "ready"
    assert page.capture_manifest_fail_count == 1
    assert "capture_manifest" in page.sharing_readiness["failed_checks"]
    assert "Fix capture validation before external sharing." in html


def test_portfolio_page_blocks_real_run_missing_capture_validation(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False)
    (run_dir / "capture_manifest_validation.json").unlink()

    page = build_portfolio_page(run_dir)

    assert page.result_status == "blocked"
    assert page.sharing_readiness["readiness_level"] == "blocked"
    assert page.sharing_readiness["capture_manifest_status"] == "missing"
    assert "capture_manifest" in page.sharing_readiness["failed_checks"]


def test_generate_portfolio_page_cli_writes_html(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    output = tmp_path / "page.html"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_portfolio_page.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["output"] == str(output)
    html = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "Evidence score" in html
    assert "Sharing Readiness" in html


def _write_run(tmp_path: Path, *, dry_run: bool = True) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": dry_run,
            "backend": "lerf",
            "queries": ["mug"],
        },
    )
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "scene_name": "desk_scene",
            "backend": "lerf",
            "dry_run": dry_run,
            "evidence_level": "dry_run_demo_ready" if dry_run else "portfolio_ready_real_run",
            "score": 77,
            "max_score": 100,
            "summary": "Run is a useful smoke demo.",
            "top_recommendations": ["Run on a CUDA machine."],
            "metrics": {"top_k_hit_rate": 0.0},
        },
    )
    _write_json(run_dir / "run_audit.json", {"status": "needs_review" if dry_run else "ready"})
    _write_json(run_dir / "run_result_card.json", {"result_status": "shareable_smoke_demo" if dry_run else "portfolio_ready"})
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "needs_review" if dry_run else "ready", "fail_count": 0, "warn_count": 1 if dry_run else 0},
    )
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "pass",
            "ok": True,
            "totals": {"counter_evidence_count": 0, "risk_flag_count": 0},
            "tasks": [{"task": "mug", "counter_evidence_count": 0, "risk_flag_count": 0}],
        },
    )
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"mean_iou_2d": 0.2})
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image")
    _write_text(run_dir / "demo_assets" / "mug" / "view_0000_overlay.png", "image")
    _write_text(run_dir / "preflight_report.md", "# Preflight\n")
    _write_text(run_dir / "failure_diagnostics.md", "# Failure Diagnostics\n")
    _write_text(run_dir / "evidence_scorecard.md", "# Scorecard\n")
    _write_text(run_dir / "quality_gate.md", "# Quality Gate\n")
    _write_text(run_dir / "run_readiness.md", "# Run Readiness Gate\n")
    _write_text(run_dir / "run_result_card.md", "# Run Result Card\n")
    _write_text(run_dir / "run_audit.md", "# Audit\n")
    _write_text(run_dir / "run_recommendations.md", "# Recommendations\n")
    _write_text(run_dir / "scene_data_inspection.md", "# Scene\n")
    _write_text(run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench.html", "<!doctype html>\n")
    _write_text(run_dir / "research_report.md", "# Research Report\n")
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "readiness_level": "needs_pack_validation",
            "pack_ok": None,
            "readiness_summary": {
                "status": "warn",
                "readiness_level": "needs_pack_validation",
                "failed_check_count": 0,
                "warning_check_count": 1,
                "packet_warning_count": 0,
                "failed_checks": [],
                "warning_checks": ["portfolio_pack"],
                "top_blockers": [],
                "top_warnings": ["portfolio_pack: portfolio pack was not validated"],
                "pack_ok": None,
                "capture_manifest_status": "needs_review" if dry_run else "ready",
                "capture_manifest_fail_count": 0,
                "query_evidence_status": "pass",
                "query_counter_evidence_count": 0,
                "query_risk_flag_count": 0,
                "recommended_next_action": "Finalize annotations with --export-pack --zip-pack.",
            },
        },
    )
    _write_text(run_dir / "submission_packet" / "submission_checklist.md", "# Submission\n")
    _write_text(run_dir / "portfolio_result_card.md", "# Card\n")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
