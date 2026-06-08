import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.claim_audit import audit_claims, write_claim_audit


ROOT = Path(__file__).resolve().parents[1]


def test_claim_audit_flags_unqualified_sota_claim(tmp_path: Path) -> None:
    _write_project_docs(
        tmp_path,
        extra="This system achieves state-of-the-art 3D segmentation.",
    )

    report = audit_claims(root=tmp_path)

    assert report.status == "fail"
    assert any(finding.rule == "sota_or_benchmark_claim" for finding in report.findings)


def test_claim_audit_allows_negated_sota_disclaimer(tmp_path: Path) -> None:
    _write_project_docs(
        tmp_path,
        extra="This is not a state-of-the-art segmentation or detection model.",
    )
    run_dir = _write_run(tmp_path)

    report = audit_claims(root=tmp_path, run_dir=run_dir)

    assert report.status == "pass"
    assert report.findings == []
    assert any(check.name == "dry_run_run_artifact_disclaimer" for check in report.checks)


def test_write_claim_audit_outputs_json_and_markdown(tmp_path: Path) -> None:
    _write_project_docs(tmp_path)
    run_dir = _write_run(tmp_path)

    report = write_claim_audit(root=tmp_path, run_dir=run_dir)

    assert report.ok is True
    assert (run_dir / "claim_audit.json").exists()
    assert (run_dir / "claim_audit.md").exists()
    assert "# Claim Audit" in (run_dir / "claim_audit.md").read_text(encoding="utf-8")


def test_audit_claims_cli(tmp_path: Path) -> None:
    _write_project_docs(tmp_path)
    run_dir = _write_run(tmp_path)
    output = tmp_path / "claim_audit.json"
    markdown = tmp_path / "claim_audit.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_claims.py"),
            "--root",
            str(tmp_path),
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
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert markdown.exists()


def _write_project_docs(root: Path, *, extra: str = "") -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    readme = (
        "This is a research engineering project built on Nerfstudio and LERF.\n"
        "This is not a new NeRF architecture.\n"
        "Dry-run outputs are synthetic and validate pipeline behavior only.\n"
        "Full training requires an NVIDIA GPU.\n"
        "Do not claim state-of-the-art performance.\n"
        f"{extra}\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    (docs / "method_summary.md").write_text(
        "The project is not claiming a novel architecture.\n",
        encoding="utf-8",
    )


def _write_run(root: Path) -> Path:
    run_dir = root / "results" / "pipeline_runs" / "desk_scene"
    _write_json(
        run_dir / "pipeline_summary.json",
        {"scene_name": "desk_scene", "success": True, "dry_run": True},
    )
    _write_json(
        run_dir / "submission_packet" / "submission_packet.json",
        {
            "avoid_claims": [
                "Do not claim state-of-the-art performance.",
                "Do not describe dry-run overlays as trained LERF outputs from a real scene.",
            ]
        },
    )
    (run_dir / "research_report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "research_report.md").write_text(
        "This is a dry-run smoke demo with synthetic outputs.\n",
        encoding="utf-8",
    )
    (run_dir / "submission_packet" / "submission_checklist.md").write_text(
        "Dry-run smoke demo. Claims to avoid: state-of-the-art.\n",
        encoding="utf-8",
    )
    return run_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
