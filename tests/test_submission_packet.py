import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.submission_packet import (
    build_submission_packet,
    write_submission_packet,
)


ROOT = Path(__file__).resolve().parents[1]


def test_build_submission_packet_calibrates_dry_run_claims(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    validation = tmp_path / "portfolio_pack_validation.json"
    _write_json(
        validation,
        {
            "ok": True,
            "warnings": ["dry-run pack warning"],
            "errors": [],
            "path_leaks": [],
        },
    )

    packet = build_submission_packet(
        run_dir,
        pack_validation_path=validation,
        repo_url="https://github.com/example/repo",
        ci_url="https://github.com/example/repo/actions/runs/1",
    )

    assert packet.run_dir == "run"
    assert packet.readiness_level == "shareable_smoke_demo"
    assert packet.pack_ok is True
    assert packet.query_evidence_status == "pass"
    assert packet.query_counter_evidence_count == 0
    assert packet.query_risk_flag_count == 0
    assert packet.readiness_summary["status"] == "warn"
    assert packet.readiness_summary["readiness_level"] == "shareable_smoke_demo"
    assert packet.readiness_summary["failed_checks"] == []
    assert packet.readiness_summary["query_evidence_status"] == "pass"
    assert packet.readiness_summary["query_counter_evidence_count"] == 0
    assert packet.readiness_summary["query_risk_flag_count"] == 0
    assert "quality_gate" in packet.readiness_summary["warning_checks"]
    assert "Run a real CUDA-backed scene." in packet.readiness_summary["recommended_next_action"]
    assert any("CPU-safe pipeline wiring" in claim for claim in packet.allowed_claims)
    assert any("trained LERF outputs" in claim for claim in packet.avoid_claims)
    assert any(item.name == "claim_audit" and item.status == "pass" for item in packet.checklist)
    assert any(item.name == "query_evidence" and item.status == "pass" for item in packet.checklist)
    assert any(item.name == "failure_diagnostics" and item.status == "pass" for item in packet.checklist)
    assert any(item.name == "path_leaks" and item.status == "pass" for item in packet.checklist)
    assert packet.recommended_links["research_report"] == "research_report.md"
    assert packet.recommended_links["portfolio_page"] == "portfolio_page.html"
    assert packet.recommended_links["reproduction_report"] == "reproduction_report.md"
    assert packet.recommended_links["failure_diagnostics"] == "failure_diagnostics.md"
    assert packet.recommended_links["quality_gate"] == "quality_gate.md"


def test_write_submission_packet_outputs_markdown_and_briefs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    output_dir = tmp_path / "packet"

    packet = write_submission_packet(run_dir, output_dir=output_dir)

    assert packet.readiness_level == "needs_pack_validation"
    assert packet.readiness_summary["status"] == "warn"
    assert "portfolio_pack" in packet.readiness_summary["warning_checks"]
    assert packet.next_actions[0].startswith("Finalize annotations")
    assert "--export-pack --zip-pack" in packet.next_actions[0]
    assert (output_dir / "submission_packet.json").exists()
    assert (output_dir / "submission_checklist.md").exists()
    assert (output_dir / "cv_project_entry.md").exists()
    assert (output_dir / "professor_email_brief.md").exists()
    markdown = (output_dir / "submission_checklist.md").read_text(encoding="utf-8")
    assert "# Submission Checklist" in markdown
    assert "## Readiness Summary" in markdown
    assert "- Warning checks:" in markdown
    assert "portfolio_pack" in markdown
    assert "dry-run smoke demo" in (output_dir / "professor_email_brief.md").read_text(encoding="utf-8")


def test_submission_packet_uses_share_safe_pack_path(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    pack = tmp_path / "portfolio_pack.zip"
    pack.write_text("not a real zip", encoding="utf-8")
    validation = tmp_path / "portfolio_pack_validation.json"
    _write_json(validation, {"ok": True, "warnings": [], "errors": [], "path_leaks": []})

    packet = build_submission_packet(run_dir, pack_dir=pack, pack_validation_path=validation)

    assert packet.pack_dir == "portfolio_pack.zip"
    assert packet.recommended_links["portfolio_pack"] == "portfolio_pack.zip"
    external_payload = json.dumps(
        {
            "run_dir": packet.run_dir,
            "pack_dir": packet.pack_dir,
            "recommended_links": packet.recommended_links,
        }
    )
    assert str(tmp_path) not in external_payload


def test_submission_packet_missing_pack_error_is_share_safe(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    missing_pack = tmp_path / "missing" / "portfolio_pack.zip"

    packet = build_submission_packet(run_dir, pack_dir=missing_pack)

    assert packet.pack_dir == "portfolio_pack.zip"
    assert packet.recommended_links["portfolio_pack"] == "portfolio_pack.zip"
    assert packet.pack_ok is False
    pack_item = next(item for item in packet.checklist if item.name == "portfolio_pack")
    assert pack_item.status == "fail"
    pack_payload = json.dumps(
        {
            "pack_dir": packet.pack_dir,
            "portfolio_pack": packet.recommended_links["portfolio_pack"],
            "portfolio_pack_check": pack_item.to_dict(),
        }
    )
    assert str(tmp_path) not in pack_payload


def test_create_submission_packet_cli(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    output_dir = tmp_path / "packet"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_submission_packet.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_dir),
            "--repo-url",
            "https://github.com/example/repo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((output_dir / "submission_packet.json").read_text(encoding="utf-8"))
    assert payload["run_dir"] == "run"
    assert payload["repo_url"] == "https://github.com/example/repo"
    assert payload["readiness_level"] == "needs_pack_validation"
    assert payload["readiness_summary"]["readiness_level"] == "needs_pack_validation"
    assert payload["recommended_links"]["research_report"] == "research_report.md"


def test_create_submission_packet_cli_uses_share_safe_pack_path(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    output_dir = tmp_path / "packet"
    pack = tmp_path / "portfolio_pack.zip"
    pack.write_text("not a real zip", encoding="utf-8")
    validation = tmp_path / "portfolio_pack_validation.json"
    _write_json(validation, {"ok": True, "warnings": [], "errors": [], "path_leaks": []})

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_submission_packet.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_dir),
            "--pack",
            str(pack),
            "--pack-validation",
            str(validation),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((output_dir / "submission_packet.json").read_text(encoding="utf-8"))
    assert payload["run_dir"] == "run"
    assert payload["pack_dir"] == "portfolio_pack.zip"
    assert payload["recommended_links"]["portfolio_pack"] == "portfolio_pack.zip"
    external_payload = json.dumps(
        {
            "run_dir": payload["run_dir"],
            "pack_dir": payload["pack_dir"],
            "recommended_links": payload["recommended_links"],
        }
    )
    assert str(tmp_path) not in external_payload


def test_submission_packet_blocks_failed_claim_audit(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    _write_json(run_dir / "claim_audit.json", {"status": "fail", "ok": False, "fail_count": 1})

    packet = build_submission_packet(run_dir)

    assert packet.readiness_level == "blocked"
    assert packet.readiness_summary["status"] == "fail"
    assert "claim_audit" in packet.readiness_summary["failed_checks"]
    assert any("claim_audit" in item for item in packet.readiness_summary["top_blockers"])
    assert any(item.name == "claim_audit" and item.status == "fail" for item in packet.checklist)


def test_submission_packet_warns_claim_audit_warning(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=True)
    _write_json(run_dir / "claim_audit.json", {"status": "warn", "ok": False, "warn_count": 1, "fail_count": 0})

    packet = build_submission_packet(run_dir)

    assert packet.readiness_level == "needs_pack_validation"
    assert "claim_audit" not in packet.readiness_summary["failed_checks"]
    assert "claim_audit" in packet.readiness_summary["warning_checks"]
    assert any(item.name == "claim_audit" and item.status == "warn" for item in packet.checklist)


def test_submission_packet_blocks_query_risk_flags(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=False)
    validation = tmp_path / "portfolio_pack_validation.json"
    _write_json(validation, {"ok": True, "warnings": [], "errors": [], "path_leaks": []})
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "totals": {"counter_evidence_count": 1, "risk_flag_count": 2},
            "tasks": [{"task": "safe place", "counter_evidence_count": 1, "risk_flag_count": 2}],
        },
    )

    packet = build_submission_packet(run_dir, pack_validation_path=validation)

    assert packet.readiness_level == "blocked"
    assert packet.query_evidence_status == "warn"
    assert packet.query_counter_evidence_count == 1
    assert packet.query_risk_flag_count == 2
    assert packet.readiness_summary["status"] == "fail"
    assert packet.readiness_summary["query_risk_flag_count"] == 2
    assert "query_evidence" in packet.readiness_summary["failed_checks"]
    assert any(item.name == "query_evidence" and item.status == "fail" for item in packet.checklist)
    assert any("query-risk flags" in claim for claim in packet.avoid_claims)


def test_submission_packet_warns_on_counter_evidence_without_blocking(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", dry_run=False)
    validation = tmp_path / "portfolio_pack_validation.json"
    _write_json(validation, {"ok": True, "warnings": [], "errors": [], "path_leaks": []})
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "totals": {"counter_evidence_count": 1, "risk_flag_count": 0},
            "tasks": [{"task": "container", "counter_evidence_count": 1, "risk_flag_count": 0}],
        },
    )

    packet = build_submission_packet(run_dir, pack_validation_path=validation)

    assert packet.readiness_level == "real_run_review_ready"
    assert packet.query_counter_evidence_count == 1
    assert packet.query_risk_flag_count == 0
    assert packet.readiness_summary["status"] == "warn"
    assert packet.readiness_summary["query_counter_evidence_count"] == 1
    assert "query_evidence" in packet.readiness_summary["warning_checks"]
    assert any(item.name == "query_evidence" and item.status == "warn" for item in packet.checklist)


def _write_run(run_dir: Path, *, dry_run: bool) -> Path:
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
            "dry_run": dry_run,
            "backend": "lerf",
            "evidence_level": "dry_run_demo_ready" if dry_run else "portfolio_ready_real_run",
            "score": 85,
            "max_score": 100,
        },
    )
    _write_json(
        run_dir / "quality_gate.json",
        {"profile": "smoke" if dry_run else "portfolio", "status": "warn", "passed": True},
    )
    _write_json(run_dir / "run_audit.json", {"status": "needs_review", "score": 70})
    _write_json(run_dir / "failure_diagnostics.json", {"status": "clear", "blocker_count": 0, "warning_count": 0})
    _write_json(run_dir / "claim_audit.json", {"status": "pass", "ok": True, "fail_count": 0, "warn_count": 0})
    _write_json(
        run_dir / "run_recommendations.json",
        {"recommendations": [{"action": "Run a real CUDA-backed scene."}]},
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
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True, "warnings": []})
    _write_json(run_dir / "research_report.json", {"backend": "lerf"})
    _write_text(run_dir / "research_report.md", "# Research\n")
    _write_text(run_dir / "portfolio_page.html", "<!doctype html>\n")
    _write_json(run_dir / "reproduction_manifest.json", {"scene_name": "desk_scene"})
    _write_text(run_dir / "reproduction_report.md", "# Reproduction\n")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
