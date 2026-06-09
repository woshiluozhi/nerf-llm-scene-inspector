import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.visualization.project_site import build_project_site


ROOT = Path(__file__).resolve().parents[1]


def test_build_project_site_links_run_pages_without_absolute_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assets = repo / "docs" / "assets"
    assets.mkdir(parents=True)
    (assets / "query_grid.png").write_bytes(b"png")
    (assets / "demo_montage.gif").write_bytes(b"gif")
    (assets / "mug_overlay.png").write_bytes(b"png")
    runs_root = repo / "results" / "pipeline_runs"
    run_dir = runs_root / "desk_scene"
    run_dir.mkdir(parents=True)
    (run_dir / "portfolio_page.html").write_text("<!doctype html>\n", encoding="utf-8")
    run_index = {
        "entries": [
            {
                "scene_name": "desk_scene",
                "success": True,
                "dry_run": True,
                "backend": "lerf",
                "query_count": 3,
                "audit_score": 92,
                "result_status": "blocked",
                "submission_readiness_level": "blocked",
                "query_risk_flag_count": 1,
                "top_k_hit_rate": 1.0,
                "mean_iou_2d": 0.5,
                "run_dir": "desk_scene",
                "artifacts": {"portfolio_page": "portfolio_page.html"},
            }
        ]
    }
    run_index_path = runs_root / "run_index.json"
    run_index_path.write_text(json.dumps(run_index), encoding="utf-8")
    (runs_root / "run_comparison.md").write_text("# Comparison\n", encoding="utf-8")
    output = repo / "docs" / "index.html"

    site = build_project_site(output, repo_root=repo, run_index_path=run_index_path)
    html = site.to_html()

    assert "NeRF-LLM Scene Inspector" in html
    assert "Counter-evidence answers" in html
    assert "Query evidence audit" in html
    assert "risk flags" in html
    assert "Risk Flags" in html
    assert "Result" in html
    assert "Submission" in html
    assert "blocked" in html
    assert "desk_scene" in html
    assert "../results/pipeline_runs/desk_scene/portfolio_page.html" in html
    assert "../results/pipeline_runs/run_comparison.md" in html
    assert "assets/query_grid.png" in html
    assert str(tmp_path) not in html


def test_generate_project_site_cli_writes_html(tmp_path: Path) -> None:
    output = tmp_path / "site" / "index.html"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_project_site.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    payload = json.loads(result.stdout)
    assert payload["output"] == str(output)
    assert "NeRF-LLM Scene Inspector" in output.read_text(encoding="utf-8")
