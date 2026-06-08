import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.run_result_card import (
    build_run_result_card,
    write_run_result_card,
)


ROOT = Path(__file__).resolve().parents[1]


def test_build_run_result_card_calibrates_dry_run_claims(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    card = build_run_result_card(run_dir)

    assert card.scene_name == "desk_scene"
    assert card.result_status == "shareable_smoke_demo"
    assert card.dry_run is True
    assert "does not prove trained NeRF/LERF" in card.primary_takeaway
    assert any("dry-run" in claim.lower() for claim in card.do_not_claim)
    assert card.evidence_snapshot["claim_audit"] == "pass"
    assert card.metrics["mean_iou_2d"] == 0.42
    assert any(check.name == "claim_audit" and check.status == "pass" for check in card.checks)


def test_write_run_result_card_outputs_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    card = write_run_result_card(run_dir)

    payload = json.loads((run_dir / "run_result_card.json").read_text(encoding="utf-8"))
    markdown = (run_dir / "run_result_card.md").read_text(encoding="utf-8")
    assert payload["result_status"] == card.result_status
    assert "# Run Result Card" in markdown
    assert "## Do Not Claim" in markdown


def test_create_run_result_card_cli(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    output = tmp_path / "card.md"
    json_output = tmp_path / "card.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_run_result_card.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--json-output",
            str(json_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(json_output.read_text(encoding="utf-8"))["scene_name"] == "desk_scene"
    assert "# Run Result Card" in output.read_text(encoding="utf-8")


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "scene_name": "desk_scene",
            "success": True,
            "dry_run": True,
            "backend": "lerf",
            "queries": ["mug", "laptop"],
        },
    )
    _write_json(
        run_dir / "evidence_scorecard.json",
        {
            "evidence_level": "dry_run_demo_ready",
            "score": 85,
            "max_score": 113,
            "dry_run": True,
            "overlay_count": 2,
            "metrics": {
                "preflight_status": "needs_attention",
                "capture_manifest_status": "needs_review",
            },
        },
    )
    _write_json(run_dir / "quality_gate.json", {"profile": "smoke", "status": "warn", "passed": True})
    _write_json(run_dir / "run_audit.json", {"status": "needs_review", "score": 52})
    _write_json(run_dir / "claim_audit.json", {"status": "pass", "ok": True, "fail_count": 0, "warn_count": 0})
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "readiness_level": "shareable_smoke_demo",
            "avoid_claims": [
                "Do not claim state-of-the-art performance.",
                "Do not describe dry-run overlays as trained LERF outputs from a real scene.",
            ],
            "next_actions": ["Run the same pipeline on a CUDA machine."],
        },
    )
    _write_json(
        run_dir / "evaluation" / "eval_summary.json",
        {
            "top_k_hit_rate": 1.0,
            "mean_iou_2d": 0.42,
            "semantic_success_rate": 1.0,
            "average_relevancy_score": 0.88,
            "num_evaluated_queries": 2,
            "num_bbox_annotated_queries": 1,
        },
    )
    _write_json(run_dir / "evaluation" / "annotation_validation.json", {"ok": True, "warnings": []})
    _write_json(run_dir / "scene_data_inspection.json", {"quality_score": 0.9, "pose_coverage_score": 0.8})
    _write_json(run_dir / "run_recommendations.json", {"recommendations": []})
    _write_json(run_dir / "scene_relations" / "scene_relations_summary.json", {"num_entities": 2, "num_relations": 1})
    _write_text(run_dir / "portfolio_page.html", "<!doctype html>\n")
    _write_text(run_dir / "research_report.md", "# Report\n")
    _write_text(run_dir / "evidence_scorecard.md", "# Scorecard\n")
    _write_text(run_dir / "quality_gate.md", "# Quality\n")
    _write_text(run_dir / "claim_audit.md", "# Claim\n")
    _write_text(run_dir / "submission_packet" / "submission_checklist.md", "# Submission\n")
    _write_text(run_dir / "real_run_plan" / "real_run_plan.md", "# Plan\n")
    _write_text(run_dir / "reproduction_report.md", "# Reproduction\n")
    _write_text(run_dir / "demo_assets" / "query_grid.png", "image\n")
    _write_text(run_dir / "evaluation" / "annotation_review.md", "# Annotation\n")
    _write_text(run_dir / "scene_relations" / "scene_relations_report.md", "# Relations\n")
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
