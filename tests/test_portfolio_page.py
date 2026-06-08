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
    assert "demo_assets/query_grid.png" in html
    assert "quality_gate.md" in html
    assert "run_result_card.md" in html
    assert "annotation_workbench/annotation_workbench.html" in html
    assert "research_report.md" in html
    assert "submission_packet/submission_checklist.md" in html
    assert str(tmp_path) not in html
    assert "C:\\Users" not in html


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


def _write_run(tmp_path: Path) -> Path:
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
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "scene_name": "desk_scene",
            "backend": "lerf",
            "dry_run": True,
            "evidence_level": "dry_run_demo_ready",
            "score": 77,
            "max_score": 100,
            "summary": "Run is a useful smoke demo.",
            "top_recommendations": ["Run on a CUDA machine."],
            "metrics": {"top_k_hit_rate": 0.0},
        },
    )
    _write_json(run_dir / "run_audit.json", {"status": "needs_review"})
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"mean_iou_2d": 0.2})
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image")
    _write_text(run_dir / "demo_assets" / "mug" / "view_0000_overlay.png", "image")
    _write_text(run_dir / "preflight_report.md", "# Preflight\n")
    _write_text(run_dir / "evidence_scorecard.md", "# Scorecard\n")
    _write_text(run_dir / "quality_gate.md", "# Quality Gate\n")
    _write_text(run_dir / "run_result_card.md", "# Run Result Card\n")
    _write_text(run_dir / "run_audit.md", "# Audit\n")
    _write_text(run_dir / "run_recommendations.md", "# Recommendations\n")
    _write_text(run_dir / "scene_data_inspection.md", "# Scene\n")
    _write_text(run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench.html", "<!doctype html>\n")
    _write_text(run_dir / "research_report.md", "# Research Report\n")
    _write_text(run_dir / "submission_packet" / "submission_checklist.md", "# Submission\n")
    _write_text(run_dir / "portfolio_result_card.md", "# Card\n")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
