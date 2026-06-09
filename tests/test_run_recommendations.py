import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.run_recommendations import build_run_recommendations


ROOT = Path(__file__).resolve().parents[1]


def test_recommendations_for_ready_run_suggest_export(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready")

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "ready_for_portfolio"
    assert report.critical_count == 0
    assert report.recommendations[0].category == "portfolio_export"
    assert "finalize_annotations.py" in report.recommendations[0].command
    assert "--export-pack --zip-pack" in report.recommendations[0].command


def test_recommendations_for_dry_run_prioritize_real_gpu_run(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True, audit_status="needs_review")

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "dry_run_ready_for_smoke_demo"
    assert report.top_next_action.startswith("Run the same pipeline on a real captured scene")
    assert any(item.category == "run_mode" for item in report.recommendations)


def test_recommendations_block_on_audit_blocker(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="blocked")
    _write_json(
        run_dir / "run_audit.json",
        {
            "status": "blocked",
            "findings": [
                {
                    "severity": "blocker",
                    "category": "scene_data",
                    "message": "Processed scene is not ready.",
                    "recommendation": "Recapture the scene.",
                    "artifact": "scene_data_inspection.md",
                }
            ],
        },
    )

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "blocked"
    assert report.critical_count == 1
    assert report.top_next_action == "Recapture the scene."


def test_recommendations_include_preflight_failures(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready")
    _write_json(
        run_dir / "preflight_report.json",
        {
            "status": "blocked",
            "checks": [{"name": "ns-train", "status": "fail"}],
        },
    )

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "blocked"
    assert any(item.category == "preflight" for item in report.recommendations)


def test_recommendations_include_failure_diagnostics_actions(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready")
    _write_json(
        run_dir / "failure_diagnostics.json",
        {
            "status": "blocked",
            "diagnostics": [
                {
                    "severity": "blocker",
                    "category": "cuda_oom",
                    "message": "GPU memory exhaustion was detected in command output.",
                    "recommendation": "Use lerf-lite or reduce rays per batch.",
                    "command": "python scripts/check_env.py --check-upstream --require-gpu --verbose",
                    "artifact": "logs/train_language_field_command.json",
                }
            ],
        },
    )

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "blocked"
    assert any(item.category == "cuda_oom" for item in report.recommendations)
    assert report.top_next_action == "Use lerf-lite or reduce rays per batch."


def test_recommendations_include_capture_manifest_review(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready")
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "needs_review", "warn_count": 3, "fail_count": 0},
    )

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "needs_review"
    assert any(item.category == "capture_manifest" for item in report.recommendations)


def test_recommendations_include_query_evidence_warnings(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready")
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "warn",
            "ok": True,
            "fail_count": 0,
            "warn_count": 1,
            "tasks": [{"task": "mug", "evidence_mode": "2d_fallback"}],
        },
    )

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "needs_review"
    assert any(item.category == "query_evidence" for item in report.recommendations)


def test_recommendations_include_query_risk_flags(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready")
    _write_json(
        run_dir / "query_evidence_audit.json",
        {
            "status": "pass",
            "ok": True,
            "fail_count": 0,
            "warn_count": 0,
            "totals": {"counter_evidence_count": 1, "risk_flag_count": 1},
            "tasks": [
                {
                    "task": "safe place to put a hot cup",
                    "evidence_mode": "3d",
                    "counter_evidence_count": 1,
                    "risk_flag_count": 1,
                }
            ],
        },
    )

    report = build_run_recommendations(run_dir)

    assert report.readiness_level == "needs_review"
    risk_actions = [
        item for item in report.recommendations if item.action.startswith("Review query counter-evidence")
    ]
    assert risk_actions
    assert risk_actions[0].severity == "high"


def test_recommend_next_steps_cli_writes_reports(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True, audit_status="needs_review")
    output = tmp_path / "recommendations.json"
    markdown = tmp_path / "recommendations.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "recommend_next_steps.py"),
            "--run-dir",
            str(run_dir),
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
    assert json.loads(output.read_text(encoding="utf-8"))["readiness_level"] == "dry_run_ready_for_smoke_demo"
    assert "# Run Recommendations" in markdown.read_text(encoding="utf-8")


def _write_run(tmp_path: Path, *, dry_run: bool, audit_status: str) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": dry_run,
            "backend": "lerf",
            "queries": ["mug"],
            "steps": [
                {"name": "query_scene", "status": "success"},
                {"name": "generate_demo_assets", "status": "success"},
                {"name": "evaluate_queries", "status": "success"},
            ],
        },
    )
    _write_json(run_dir / "preflight_report.json", {"status": "ready", "checks": []})
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "ready", "warn_count": 0, "fail_count": 0},
    )
    _write_json(run_dir / "run_audit.json", {"status": audit_status, "findings": []})
    _write_json(run_dir / "environment_report.json", {"ok": True, "strict_failures": []})
    _write_json(
        run_dir / "scene_data_inspection.json",
        {"ready_for_training": True, "quality_score": 0.95},
    )
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True, "warnings": []})
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {"num_evaluated_queries": 1, "num_bbox_annotated_queries": 1},
    )
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
