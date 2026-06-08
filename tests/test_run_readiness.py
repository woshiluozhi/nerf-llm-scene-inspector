import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.run_readiness import build_run_readiness, write_run_readiness


ROOT = Path(__file__).resolve().parents[1]


def test_run_readiness_calibrates_dry_run(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True)

    report = build_run_readiness(run_dir)

    assert report.readiness_level == "dry_run_needs_real_run"
    assert report.ready_to_start_real_run is False
    assert report.ready_for_external_review is False
    assert report.fail_count == 0
    assert any(gate.name == "evidence_mode" and gate.status == "warn" for gate in report.gates)
    assert any(gate.name == "environment_gpu_upstream" and gate.status == "warn" for gate in report.gates)
    assert any("without --dry-run" in action or "CUDA-backed" in action for action in report.next_actions)


def test_run_readiness_real_run_with_validated_pack_is_review_ready(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=False)
    validation = tmp_path / "portfolio_pack_validation.json"
    _write_json(validation, {"ok": True, "errors": [], "warnings": [], "path_leaks": []})

    report = build_run_readiness(run_dir, pack_validation_path=validation)

    assert report.readiness_level == "portfolio_ready"
    assert report.ready_to_start_real_run is True
    assert report.ready_for_external_review is True
    assert report.fail_count == 0
    assert any(gate.name == "portfolio_pack" and gate.status == "pass" for gate in report.gates)


def test_write_run_readiness_outputs_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True)

    report = write_run_readiness(run_dir)

    payload = json.loads((run_dir / "run_readiness.json").read_text(encoding="utf-8"))
    assert payload["readiness_level"] == report.readiness_level
    assert payload["ready_to_start_real_run"] is False
    markdown = (run_dir / "run_readiness.md").read_text(encoding="utf-8")
    assert "# Run Readiness Gate" in markdown
    assert "Ready to start real run: False" in markdown


def test_create_run_readiness_cli(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, dry_run=True)
    output = tmp_path / "readiness.json"
    markdown = tmp_path / "readiness.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_run_readiness.py"),
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
    assert json.loads(output.read_text(encoding="utf-8"))["readiness_level"] == "dry_run_needs_real_run"
    assert markdown.exists()


def _write_run(tmp_path: Path, *, dry_run: bool) -> Path:
    run_dir = tmp_path / "run"
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
    _write_json(run_dir / "capture_manifest_validation.json", {"status": "ready"})
    _write_json(run_dir / "preflight_report.json", {"status": "ready", "ready_for_real_run": True})
    _write_json(run_dir / "environment_report.json", _environment_report(strict=not dry_run))
    _write_json(run_dir / "scene_data_inspection.json", {"ready_for_training": True})
    _write_json(
        run_dir / "training" / "language_train_summary.json",
        {
            "success": True,
            "dry_run": dry_run,
            "backend": "lerf",
            "variant": "lerf-lite",
            "config_path": "runs/language_desk_scene/config.yml",
        },
    )
    _write_json(run_dir / "quality_gate.json", {"status": "pass", "passed": True})
    _write_json(run_dir / "claim_audit.json", {"status": "pass", "ok": True})
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "readiness_level": "needs_pack_validation" if dry_run else "portfolio_ready",
            "pack_ok": False if dry_run else True,
        },
    )
    _write_json(
        run_dir / "run_result_card.json",
        {"dry_run": dry_run, "result_status": "dry_run_smoke_demo" if dry_run else "portfolio_ready"},
    )
    return run_dir


def _environment_report(*, strict: bool) -> dict[str, object]:
    checks = [
        {"name": "cuda", "ok": strict, "required": strict},
        {"name": "ns-process-data", "ok": strict, "required": strict},
        {"name": "ns-train", "ok": strict, "required": strict},
        {"name": "ns-train method:lerf-lite", "ok": strict, "required": strict},
    ]
    return {"ok": True, "strict_failures": [], "checks": checks}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
