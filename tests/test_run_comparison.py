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
    assert comparison.entries[1].selection_status == "dry_run_smoke_demo"
    assert comparison.best_run is not None
    assert comparison.best_run["scene_name"] == "real_scene"


def test_compare_pipeline_runs_marks_unready_real_run_for_review(tmp_path: Path) -> None:
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

    assert comparison.entries[0].selection_status == "needs_review"
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
    assert "# Pipeline Run Comparison" in markdown.read_text(encoding="utf-8")


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
) -> None:
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
    _write_json(run_dir / "run_audit.json", {"status": audit_status, "score": audit_score})
    _write_json(run_dir / "capture_manifest_validation.json", {"status": capture_status})
    _write_json(
        run_dir / "run_recommendations.json",
        {"top_next_action": "Review warnings before sharing."},
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
