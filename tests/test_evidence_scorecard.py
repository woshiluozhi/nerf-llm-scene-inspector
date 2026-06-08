import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.evidence_scorecard import build_evidence_scorecard


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_scorecard_marks_dry_run_as_smoke_demo(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True, audit_status="needs_review", preflight_status="needs_attention")

    scorecard = build_evidence_scorecard(run_dir)

    assert scorecard.evidence_level == "dry_run_demo_ready"
    assert scorecard.dry_run is True
    assert scorecard.query_report_count == 3
    assert scorecard.overlay_count >= 3
    assert scorecard.score < scorecard.max_score
    assert any("not real trained" in item.lower() for item in [scorecard.summary])


def test_evidence_scorecard_marks_complete_real_run_portfolio_ready(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready", preflight_status="ready")

    scorecard = build_evidence_scorecard(run_dir)

    assert scorecard.evidence_level == "portfolio_ready_real_run"
    assert scorecard.score >= 85
    assert scorecard.metrics["top_k_hit_rate"] == 1.0


def test_evidence_scorecard_requires_capture_manifest_for_real_portfolio(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False, audit_status="ready", preflight_status="ready")
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "needs_review", "warn_count": 2, "fail_count": 0},
    )

    scorecard = build_evidence_scorecard(run_dir)

    assert scorecard.evidence_level == "needs_review"
    assert any(criterion.name == "capture_manifest_quality" for criterion in scorecard.criteria)


def test_evidence_scorecard_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True, audit_status="needs_review", preflight_status="needs_attention")
    output = tmp_path / "scorecard.json"
    markdown = tmp_path / "scorecard.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_evidence_scorecard.py"),
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
    assert json.loads(output.read_text(encoding="utf-8"))["evidence_level"] == "dry_run_demo_ready"
    assert "# Evidence Scorecard" in markdown.read_text(encoding="utf-8")


def _write_run(
    tmp_path: Path,
    *,
    dry_run: bool,
    audit_status: str,
    preflight_status: str,
) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": dry_run,
            "backend": "lerf",
            "queries": ["mug", "bottle", "safe place to put a hot cup"],
            "steps": [
                {"name": "preflight_real_run", "status": "success" if preflight_status == "ready" else "warning"},
                {"name": "check_environment", "status": "success"},
                {"name": "query_scene", "status": "success"},
                {"name": "generate_demo_assets", "status": "success"},
                {"name": "evaluate_queries", "status": "success"},
            ],
        },
    )
    _write_json(run_dir / "preflight_report.json", {"status": preflight_status})
    _write_json(
        run_dir / "capture_manifest_validation.json",
        {"status": "ready", "warn_count": 0, "fail_count": 0},
    )
    _write_json(run_dir / "environment_report.json", {"ok": True})
    _write_json(
        run_dir / "scene_data_inspection.json",
        {
            "ready_for_training": True,
            "quality_score": 1.0,
            "frame_count": 8 if dry_run else 72,
            "pose_coverage_score": 1.0,
        },
    )
    _write_json(run_dir / "run_audit.json", {"status": audit_status})
    _write_json(run_dir / "run_recommendations.json", {"recommendations": []})
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True, "warnings": []})
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {
            "top_k_hit_rate": 1.0,
            "mean_iou_2d": 0.7,
            "semantic_success_rate": 1.0,
            "average_relevancy_score": 0.8,
            "num_bbox_annotated_queries": 2 if dry_run else 3,
            "num_evaluated_queries": 2 if dry_run else 3,
            "num_result_queries": 3,
        },
    )
    for query in ["mug", "bottle", "safe-place-to-put-a-hot-cup"]:
        _write_json(run_dir / "queries" / query / "scene_query_report.json", {"query": query})
        _write_text(run_dir / "queries" / query / "view_0000_overlay.png", "image")
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image")
    _write_text(run_dir / "logs" / "prepare_data_command.json", "{}")
    _write_text(run_dir / "project_report.md", "# Report\n")
    _write_text(run_dir / "portfolio_result_card.md", "# Card\n")
    _write_text(run_dir / "run_recommendations.md", "# Recommendations\n")
    _write_json(run_dir / "reproduction_manifest.json", {"scene_name": "desk_scene"})
    _write_text(run_dir / "reproduction_report.md", "# Reproduction\n")
    _write_text(run_dir / "reproduce_run.sh", "#!/usr/bin/env bash\n")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
