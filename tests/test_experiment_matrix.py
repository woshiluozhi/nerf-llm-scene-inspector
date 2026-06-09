import csv
import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.experiment_matrix import run_experiment_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_collects_existing_runs(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: unit_matrix\n"
        "experiments:\n"
        "  - name: lerf_lite\n"
        "    scene_name: scene_lerf\n"
        "    backend: lerf\n"
        "    variant: lerf-lite\n"
        "  - name: opennerf\n"
        "    scene_name: scene_opennerf\n"
        "    backend: opennerf\n",
        encoding="utf-8",
    )
    _write_run(
        output / "pipeline_runs" / "scene_lerf",
        scene_name="scene_lerf",
        backend="lerf",
        score=90,
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        diagnostics_status="clear",
        readiness_level="portfolio_ready",
        ready_for_external_review=True,
        submission_readiness="portfolio_ready",
        quality_status="pass",
    )
    _write_run(
        output / "pipeline_runs" / "scene_opennerf",
        scene_name="scene_opennerf",
        backend="opennerf",
        score=70,
        dry_run=False,
        diagnostics_status="blocked",
        diagnostics_blockers=1,
        readiness_level="blocked",
        quality_status="pass",
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    assert report.total_experiments == 2
    assert report.successful_experiments == 2
    assert report.portfolio_candidate_count == 1
    assert report.blocked_experiment_count == 1
    assert report.best_experiment is not None
    assert report.best_experiment["experiment_name"] == "lerf_lite"
    assert report.best_experiment["candidate_status"] == "portfolio_candidate"
    assert (output / "experiment_matrix_summary.json").exists()
    assert (output / "experiment_matrix_report.md").exists()
    rows = list(csv.DictReader((output / "experiment_matrix_table.csv").open(encoding="utf-8")))
    assert rows[0]["experiment_name"] == "lerf_lite"
    assert rows[0]["candidate_status"] == "portfolio_candidate"
    assert rows[0]["failure_diagnostics_status"] == "clear"
    assert rows[0]["query_evidence_status"] == "pass"
    assert rows[0]["query_risk_flag_count"] == "0"
    assert rows[0]["relation_edge_count"] == "4"
    assert rows[1]["candidate_status"] == "blocked"
    assert "failure diagnostics blockers=1" in rows[1]["blocking_reasons"]
    assert "run readiness is blocked" in rows[1]["blocking_reasons"]
    markdown = (output / "experiment_matrix_report.md").read_text(encoding="utf-8")
    assert "## Selection Summary" in markdown
    assert "Query Evidence" in markdown
    assert "Recommended portfolio run" in markdown


def test_experiment_matrix_blocks_unresolved_query_risk_flags(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: query_risk_matrix\n"
        "experiments:\n"
        "  - name: risky_real_run\n"
        "    scene_name: risky_scene\n"
        "    backend: lerf\n",
        encoding="utf-8",
    )
    _write_run(
        output / "pipeline_runs" / "risky_scene",
        scene_name="risky_scene",
        backend="lerf",
        score=95,
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        diagnostics_status="clear",
        readiness_level="portfolio_ready",
        ready_for_external_review=True,
        submission_readiness="portfolio_ready",
        quality_status="pass",
        query_evidence_status="warn",
        query_counter_evidence_count=1,
        query_risk_flag_count=2,
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    assert report.portfolio_candidate_count == 0
    assert report.blocked_experiment_count == 1
    entry = report.entries[0]
    assert entry.candidate_status == "blocked"
    assert entry.portfolio_score == 0.0
    assert entry.query_evidence_status == "warn"
    assert entry.query_counter_evidence_count == 1
    assert entry.query_risk_flag_count == 2
    assert "query evidence risk flags=2" in entry.blocking_reasons


def test_experiment_matrix_demotes_counter_evidence_without_risk_flags(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: counter_evidence_matrix\n"
        "experiments:\n"
        "  - name: counter_real_run\n"
        "    scene_name: counter_scene\n"
        "    backend: lerf\n",
        encoding="utf-8",
    )
    _write_run(
        output / "pipeline_runs" / "counter_scene",
        scene_name="counter_scene",
        backend="lerf",
        score=95,
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        diagnostics_status="clear",
        readiness_level="portfolio_ready",
        ready_for_external_review=True,
        submission_readiness="portfolio_ready",
        quality_status="pass",
        query_evidence_status="warn",
        query_counter_evidence_count=1,
        query_risk_flag_count=0,
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    assert report.portfolio_candidate_count == 0
    assert report.blocked_experiment_count == 0
    entry = report.entries[0]
    assert entry.candidate_status == "real_run_review_ready"
    assert entry.query_counter_evidence_count == 1
    assert "query evidence counter-evidence=1" in entry.blocking_reasons


def test_experiment_matrix_prefers_fresh_query_audit_over_stale_submission(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: stale_submission_matrix\n"
        "experiments:\n"
        "  - name: stale_submission\n"
        "    scene_name: stale_scene\n"
        "    backend: lerf\n",
        encoding="utf-8",
    )
    run_dir = output / "pipeline_runs" / "stale_scene"
    _write_run(
        run_dir,
        scene_name="stale_scene",
        backend="lerf",
        score=95,
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        diagnostics_status="clear",
        readiness_level="portfolio_ready",
        ready_for_external_review=True,
        submission_readiness="portfolio_ready",
        quality_status="pass",
        query_evidence_status="pass",
        query_counter_evidence_count=0,
        query_risk_flag_count=0,
    )
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "totals": {"counter_evidence_count": 0, "risk_flag_count": 0},
            "tasks": [{"task": "safe place", "counter_evidence_count": 0, "risk_flag_count": 1}],
        },
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    entry = report.entries[0]
    assert entry.candidate_status == "blocked"
    assert entry.query_evidence_status == "warn"
    assert entry.query_risk_flag_count == 1
    assert "query evidence risk flags=1" in entry.blocking_reasons


def test_experiment_matrix_blocks_stale_run_audit_blocker_count(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: stale_audit_matrix\n"
        "experiments:\n"
        "  - name: stale_audit\n"
        "    scene_name: stale_audit_scene\n"
        "    backend: lerf\n",
        encoding="utf-8",
    )
    _write_run(
        output / "pipeline_runs" / "stale_audit_scene",
        scene_name="stale_audit_scene",
        backend="lerf",
        score=95,
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        diagnostics_status="clear",
        readiness_level="portfolio_ready",
        ready_for_external_review=True,
        submission_readiness="portfolio_ready",
        quality_status="pass",
        audit_status="ready",
        audit_blocker_count=1,
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    entry = report.entries[0]
    assert entry.candidate_status == "blocked"
    assert entry.portfolio_score == 0.0
    assert entry.audit_status == "ready"
    assert entry.audit_blocker_count == 1
    assert "run audit blockers=1" in entry.blocking_reasons


def test_experiment_matrix_blocks_stale_capture_manifest_fail_count(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "matrix_name: stale_capture_matrix\n"
        "experiments:\n"
        "  - name: stale_capture\n"
        "    scene_name: stale_capture_scene\n"
        "    backend: lerf\n",
        encoding="utf-8",
    )
    _write_run(
        output / "pipeline_runs" / "stale_capture_scene",
        scene_name="stale_capture_scene",
        backend="lerf",
        score=95,
        dry_run=False,
        evidence_level="portfolio_ready_real_run",
        diagnostics_status="clear",
        readiness_level="portfolio_ready",
        ready_for_external_review=True,
        submission_readiness="portfolio_ready",
        quality_status="pass",
        capture_status="ready",
        capture_fail_count=1,
    )

    report = run_experiment_matrix(config_path=config, output_dir=output, collect_only=True)

    entry = report.entries[0]
    assert entry.candidate_status == "blocked"
    assert entry.portfolio_score == 0.0
    assert entry.capture_manifest_status == "ready"
    assert entry.capture_manifest_fail_count == 1
    assert "capture manifest failures=1" in entry.blocking_reasons


def test_run_experiment_matrix_cli_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("method_name: lerf-lite\n", encoding="utf-8")
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "matrix_output"
    matrix.write_text(
        "matrix_name: cli_matrix\n"
        "base:\n"
        f"  input: {tmp_path.as_posix()}\n"
        "  type: images\n"
        f"  config: {config_path.as_posix()}\n"
        "  dry_run: true\n"
        "  skip_baseline: true\n"
        "  skip_language: true\n"
        "  skip_demo: true\n"
        "  skip_eval: true\n"
        "  analyze_relations: true\n"
        "experiments:\n"
        "  - name: cli_lerf\n"
        "    scene_name: cli_lerf\n"
        "    backend: lerf\n"
        "    queries:\n"
        "      - mug\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment_matrix.py"),
            "--config",
            str(matrix),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "experiment_matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["matrix_name"] == "cli_matrix"
    assert summary["successful_experiments"] == 1
    assert summary["entries"][0]["candidate_status"] == "dry_run_smoke"
    assert (output / "pipeline_runs" / "cli_lerf" / "scene_relations" / "scene_relations_report.md").exists()


def _write_run(
    run_dir: Path,
    *,
    scene_name: str,
    backend: str,
    score: int,
    dry_run: bool = True,
    evidence_level: str = "dry_run_demo_ready",
    audit_status: str = "ready",
    audit_blocker_count: int = 0,
    capture_status: str = "ready",
    capture_fail_count: int = 0,
    diagnostics_status: str = "clear",
    diagnostics_blockers: int = 0,
    diagnostics_warnings: int = 0,
    readiness_level: str = "dry_run_needs_real_run",
    ready_for_external_review: bool = False,
    submission_readiness: str = "shareable_smoke_demo",
    quality_status: str = "warn",
    result_status: str | None = None,
    query_evidence_status: str = "pass",
    query_counter_evidence_count: int = 0,
    query_risk_flag_count: int = 0,
) -> None:
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": scene_name,
            "success": True,
            "dry_run": dry_run,
            "backend": backend,
            "queries": ["mug", "cup"],
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    )
    _write_json(run_dir / "evidence_scorecard.json", {"evidence_level": evidence_level, "score": score, "max_score": 100})
    _write_json(run_dir / "run_audit.json", {"status": audit_status, "blocker_count": audit_blocker_count})
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": capture_status, "fail_count": capture_fail_count},
    )
    _write_json(run_dir / "quality_gate.json", {"status": quality_status, "passed": quality_status != "fail"})
    _write_json(
        run_dir / "failure_diagnostics.json",
        {
            "status": diagnostics_status,
            "blocker_count": diagnostics_blockers,
            "warning_count": diagnostics_warnings,
        },
    )
    _write_json(
        run_dir / "run_readiness.json",
        {
            "readiness_level": readiness_level,
            "ready_to_start_real_run": not dry_run,
            "ready_for_external_review": ready_for_external_review,
        },
    )
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "readiness_level": submission_readiness,
            "query_evidence_status": query_evidence_status,
            "query_counter_evidence_count": query_counter_evidence_count,
            "query_risk_flag_count": query_risk_flag_count,
            "share_decision": "Ready for portfolio sharing with the recorded evidence and limitations.",
        },
    )
    _write_json(
        run_dir / "run_result_card.json",
        {"result_status": result_status or ("portfolio_ready" if not dry_run else "shareable_smoke_demo")},
    )
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": query_evidence_status,
            "ok": query_evidence_status != "fail",
            "totals": {
                "counter_evidence_count": query_counter_evidence_count,
                "risk_flag_count": query_risk_flag_count,
            },
            "tasks": [
                {
                    "task": "mug",
                    "counter_evidence_count": query_counter_evidence_count,
                    "risk_flag_count": query_risk_flag_count,
                }
            ],
        },
    )
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {
            "num_evaluated_queries": 2,
            "top_k_hit_rate": 0.5,
            "mean_iou_2d": 0.25,
            "average_relevancy_score": 0.8,
        },
    )
    _write_json(
        run_dir / "prompt_sensitivity" / "prompt_sensitivity_summary.json",
        {"stable_group_count": 1, "num_groups": 2},
    )
    _write_json(
        run_dir / "scene_relations" / "scene_relations_summary.json",
        {"num_entities": 3, "num_relations": 4},
    )
    _write_json(run_dir / "run_recommendations.json", {"top_next_action": "review real outputs"})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
