import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.quality_gate import check_run_quality


ROOT = Path(__file__).resolve().parents[1]


def test_quality_gate_smoke_passes_dry_run_with_warnings(tmp_path: Path) -> None:
    run_dir = _write_smoke_run(tmp_path)

    report = check_run_quality(run_dir, profile="smoke")

    assert report.passed is True
    assert report.status == "warn"
    assert report.dry_run is True
    assert report.evidence_level == "dry_run_demo_ready"
    assert any(criterion.name == "failure_diagnostics" and criterion.status == "pass" for criterion in report.criteria)
    assert any(criterion.name == "run_mode" and criterion.status == "warn" for criterion in report.criteria)
    assert any(criterion.name == "query_evidence" and criterion.status == "pass" for criterion in report.criteria)


def test_quality_gate_portfolio_rejects_dry_run_without_pack(tmp_path: Path) -> None:
    run_dir = _write_smoke_run(tmp_path)

    report = check_run_quality(run_dir, profile="portfolio")

    assert report.passed is False
    assert report.status == "fail"
    failed = {criterion.name for criterion in report.criteria if criterion.status == "fail"}
    assert "run_mode" in failed
    assert "portfolio_pack" in failed
    pack_criterion = next(criterion for criterion in report.criteria if criterion.name == "portfolio_pack")
    assert "finalize_annotations.py" in pack_criterion.recommendation
    assert "--export-pack --zip-pack" in pack_criterion.recommendation


def test_quality_gate_warns_on_query_risk_flags_in_smoke_profile(tmp_path: Path) -> None:
    run_dir = _write_smoke_run(tmp_path)
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "totals": {"counter_evidence_count": 1, "risk_flag_count": 2},
            "tasks": [{"task": "safe place", "counter_evidence_count": 1, "risk_flag_count": 2}],
        },
    )

    report = check_run_quality(run_dir, profile="smoke")

    criterion = next(item for item in report.criteria if item.name == "query_evidence")
    payload = report.to_dict()
    assert report.passed is True
    assert criterion.status == "warn"
    assert "risk flag" in criterion.message
    assert payload["query_counter_evidence_count"] == 1
    assert payload["query_risk_flag_count"] == 2


def test_quality_gate_portfolio_fails_on_query_risk_flags(tmp_path: Path) -> None:
    run_dir = _write_portfolio_run(tmp_path)
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "totals": {"counter_evidence_count": 1, "risk_flag_count": 1},
            "tasks": [{"task": "safe place", "counter_evidence_count": 1, "risk_flag_count": 1}],
        },
    )

    report = check_run_quality(
        run_dir,
        profile="portfolio",
        require_pack=False,
        min_query_reports=3,
        min_evaluated_queries=3,
    )

    criterion = next(item for item in report.criteria if item.name == "query_evidence")
    assert report.passed is False
    assert report.status == "fail"
    assert criterion.status == "fail"
    assert "physical-action" in criterion.recommendation


def test_check_run_quality_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_smoke_run(tmp_path)
    output = tmp_path / "gate.json"
    markdown = tmp_path / "gate.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_run_quality.py"),
            "--run-dir",
            str(run_dir),
            "--profile",
            "smoke",
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
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert "# Run Quality Gate" in markdown.read_text(encoding="utf-8")


def _write_smoke_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": True,
            "backend": "lerf",
            "queries": ["mug"],
        },
    )
    _write_json(run_dir / "run_audit.json", {"status": "needs_review", "score": 84})
    _write_json(run_dir / "failure_diagnostics.json", {"status": "clear", "blocker_count": 0, "warning_count": 0})
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "scene_name": "desk_scene",
            "evidence_level": "dry_run_demo_ready",
            "score": 82,
            "max_score": 110,
            "dry_run": True,
            "query_report_count": 1,
            "overlay_count": 1,
            "evaluated_query_count": 0,
        },
    )
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "needs_review", "warn_count": 2, "fail_count": 0},
    )
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True, "warnings": []})
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"num_evaluated_queries": 0})
    _write_json(
        run_dir / "query_evidence_audit.json",
        {"status": "pass", "ok": True, "task_count": 1, "fail_count": 0, "totals": {}},
    )
    _write_json(run_dir / "queries" / "mug" / "scene_query_report.json", {"query": "mug"})
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image")
    return run_dir


def _write_portfolio_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": False,
            "backend": "lerf",
            "queries": ["mug", "laptop", "safe place"],
        },
    )
    _write_json(run_dir / "run_audit.json", {"status": "ready", "score": 100})
    _write_json(run_dir / "failure_diagnostics.json", {"status": "clear", "blocker_count": 0, "warning_count": 0})
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "scene_name": "desk_scene",
            "evidence_level": "portfolio_ready_real_run",
            "score": 100,
            "max_score": 100,
            "dry_run": False,
            "query_report_count": 3,
            "overlay_count": 3,
            "evaluated_query_count": 3,
        },
    )
    _write_json(run_dir / "capture_manifest_validation.json", {"status": "ready", "warn_count": 0, "fail_count": 0})
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True, "warnings": []})
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"num_evaluated_queries": 3})
    for query in ("mug", "laptop", "safe_place"):
        _write_json(run_dir / "queries" / query / "scene_query_report.json", {"query": query})
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
