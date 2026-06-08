import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.reproducibility import build_reproduction_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_build_reproduction_bundle_from_pipeline_summary(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    bundle = build_reproduction_bundle(run_dir)

    assert bundle.scene_name == "desk_scene"
    assert bundle.dry_run is True
    assert bundle.replay_command == "python scripts/run_scene_pipeline.py --dry-run --query mug"
    assert "python scripts/audit_run.py --run-dir run" in bundle.verification_commands
    assert "python scripts/create_evidence_scorecard.py --run-dir run" in bundle.verification_commands
    assert "python scripts/generate_portfolio_page.py --run-dir run" in bundle.verification_commands
    assert any("python scripts/review_annotations.py" in command for command in bundle.verification_commands)
    assert any(artifact.name == "pipeline_summary" and artifact.exists for artifact in bundle.artifacts)
    assert any(artifact.name == "preflight_report" and artifact.exists for artifact in bundle.artifacts)
    assert any(artifact.name == "evidence_scorecard" and artifact.exists for artifact in bundle.artifacts)
    assert any(artifact.name == "portfolio_page" and artifact.exists for artifact in bundle.artifacts)
    assert any(artifact.name == "annotation_review" and artifact.exists for artifact in bundle.artifacts)


def test_reproduction_bundle_writes_json_markdown_and_script(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    bundle = build_reproduction_bundle(run_dir)
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.md"
    script = tmp_path / "reproduce.sh"

    bundle.to_json(manifest)
    bundle.to_markdown(report)
    bundle.to_shell_script(script)

    assert json.loads(manifest.read_text(encoding="utf-8"))["scene_name"] == "desk_scene"
    assert "# Reproduction Report" in report.read_text(encoding="utf-8")
    script_text = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in script_text
    assert "python scripts/run_scene_pipeline.py --dry-run --query mug" in script_text


def test_create_reproduction_bundle_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.md"
    script = tmp_path / "reproduce.sh"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_reproduction_bundle.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(manifest),
            "--markdown-output",
            str(report),
            "--script-output",
            str(script),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(manifest.read_text(encoding="utf-8"))["replay_command"].startswith("python ")
    assert report.exists()
    assert script.exists()


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
            "provenance": {
                "command": ["scripts\\run_scene_pipeline.py", "--dry-run", "--query", "mug"]
            },
        },
    )
    _write_json(run_dir / "environment_report.json", {"ok": True})
    _write_text(run_dir / "preflight_report.md", "# Preflight\n")
    _write_text(run_dir / "evidence_scorecard.md", "# Scorecard\n")
    _write_text(run_dir / "portfolio_page.html", "<!doctype html>\n")
    _write_text(run_dir / "scene_data_inspection.md", "# Scene\n")
    _write_text(run_dir / "run_audit.md", "# Audit\n")
    _write_text(run_dir / "run_recommendations.md", "# Recommendations\n")
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image")
    _write_json(run_dir / "evaluation" / "eval_summary.json", {"num_evaluated_queries": 1})
    _write_text(run_dir / "evaluation" / "annotation_review.md", "# Annotation Review\n")
    _write_text(run_dir / "evaluation" / "annotation_review_contact_sheet.png", "image")
    _write_text(run_dir / "portfolio_result_card.md", "# Card\n")
    _write_json(run_dir / "logs" / "prepare_data_command.json", {"returncode": 0})
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
