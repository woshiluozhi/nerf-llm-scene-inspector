import json
import subprocess
import sys
from pathlib import Path

from nerf_llm_scene_inspector.evaluation.query_evidence_audit import audit_query_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_query_evidence_audit_marks_2d_fallback_as_warning(tmp_path: Path) -> None:
    run_dir = _write_query_report(tmp_path, with_region=True, with_point=False)

    audit = audit_query_evidence(run_dir)

    assert audit.ok is True
    assert audit.status == "warn"
    assert audit.task_count == 1
    assert audit.tasks[0].evidence_mode == "2d_fallback"
    assert audit.tasks[0].overlay_count == 1
    assert audit.tasks[0].query_grid_exists is True
    assert audit.tasks[0].visual_summary_exists is True
    assert audit.totals["mode_counts"]["2d_fallback"] == 1


def test_query_evidence_audit_passes_3d_query_evidence(tmp_path: Path) -> None:
    run_dir = _write_query_report(tmp_path, with_region=True, with_point=True)

    audit = audit_query_evidence(run_dir)

    assert audit.ok is True
    assert audit.status == "pass"
    assert audit.tasks[0].evidence_mode == "3d"
    assert audit.tasks[0].candidate_point_count == 1


def test_query_evidence_audit_warns_on_visual_summary_mismatch(tmp_path: Path) -> None:
    run_dir = _write_query_report(tmp_path, with_region=True, with_point=True)
    summary_path = run_dir / "queries" / "mug" / "query_visual_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "expanded_queries": ["wrong query"],
                "num_overlay_images": 99,
                "query_grid": "missing_grid.png",
                "query_montage": "missing_montage.gif",
                "scene_query_report": "scene_query_report.json",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_query_evidence(run_dir)

    assert audit.ok is True
    assert audit.status == "warn"
    assert audit.tasks[0].evidence_mode == "3d"
    assert audit.tasks[0].visual_summary_issue_count == 4
    assert audit.totals["visual_summary_issue_count"] == 4
    assert any("expanded_queries do not match" in warning for warning in audit.tasks[0].warnings)
    assert any("num_overlay_images does not match" in warning for warning in audit.tasks[0].warnings)


def test_query_evidence_audit_warns_on_counter_evidence_risk_flags(tmp_path: Path) -> None:
    run_dir = _write_query_report(
        tmp_path,
        with_region=True,
        with_point=True,
        with_counter_evidence=True,
        risk_flags=["Positive mug overlaps avoid prompt screen in view_0000."],
    )

    audit = audit_query_evidence(run_dir)

    assert audit.ok is True
    assert audit.status == "warn"
    assert audit.tasks[0].evidence_mode == "3d"
    assert audit.tasks[0].counter_evidence_count == 1
    assert audit.tasks[0].counter_evidence_labels == ["screen"]
    assert audit.tasks[0].risk_flag_count == 1
    assert audit.totals["counter_evidence_count"] == 1
    assert audit.totals["risk_flag_count"] == 1
    assert any("risk flag" in warning for warning in audit.tasks[0].warnings)
    assert any("counter-evidence" in item for item in audit.tasks[0].recommendations)


def test_query_evidence_audit_fails_viewer_fallback_without_visual_outputs(tmp_path: Path) -> None:
    run_dir = _write_viewer_fallback_report(tmp_path)

    audit = audit_query_evidence(run_dir)

    assert audit.ok is False
    assert audit.status == "fail"
    assert audit.tasks[0].status == "fail"
    assert audit.tasks[0].evidence_mode == "missing"
    assert audit.tasks[0].rendered_image_count == 0
    assert audit.tasks[0].fallback_artifact_count == 2
    assert audit.totals["fallback_artifact_count"] == 2
    assert any("not visual evidence" in warning for warning in audit.tasks[0].warnings)


def test_query_evidence_audit_fails_missing_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "queries").mkdir(parents=True)

    audit = audit_query_evidence(run_dir)

    assert audit.ok is False
    assert audit.status == "fail"
    assert audit.fail_count == 1
    assert "No scene_query_report.json" in audit.warnings[0]


def test_audit_query_evidence_cli_writes_reports(tmp_path: Path) -> None:
    run_dir = _write_query_report(tmp_path, with_region=True, with_point=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_query_evidence.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((run_dir / "query_evidence_audit.json").read_text(encoding="utf-8"))
    assert payload["status"] == "warn"
    assert (run_dir / "query_evidence_audit.md").exists()


def _write_query_report(
    tmp_path: Path,
    *,
    with_region: bool,
    with_point: bool,
    with_counter_evidence: bool = False,
    risk_flags: list[str] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    task_dir = run_dir / "queries" / "mug"
    expanded_dir = task_dir / "mug"
    expanded_dir.mkdir(parents=True)
    for name in ("view_0000_rgb.png", "view_0000_relevancy.png", "view_0000_overlay.png"):
        (expanded_dir / name).write_bytes(b"image")
    (task_dir / "query_grid.png").write_bytes(b"grid")
    (task_dir / "query_visual_summary.json").write_text(
        json.dumps({"expanded_queries": ["mug"], "query_grid": "query_grid.png"}),
        encoding="utf-8",
    )
    bounding_regions = []
    if with_region:
        bounding_regions.append(
            {
                "label": "mug",
                "score": 0.75,
                "coordinate_frame": "image",
                "bbox_2d": [1, 2, 10, 20],
                "source_view": "view_0000",
            }
        )
    candidate_points = []
    support_level = "2d_relevancy_fallback"
    if with_point:
        candidate_points.append({"label": "mug", "x": 1.0, "y": 2.0, "z": 3.0, "score": 0.8})
        support_level = "3d_candidate_point"
    answer_summary = {"support_level": support_level}
    if with_counter_evidence:
        answer_summary["counter_evidence"] = [
            {
                "label": "screen",
                "score": 0.7,
                "source_query": "screen",
                "source_role": "negative",
                "source_view": "view_0000",
            }
        ]
    if risk_flags:
        answer_summary["risk_flags"] = risk_flags
    report = {
        "scene_name": "unit_scene",
        "task": "mug",
        "query_results": [
            {
                "query": "mug",
                "backend_name": "lerf",
                "config_path": "config.yml",
                "rendered_images": [
                    {
                        "path": str(expanded_dir / "view_0000_rgb.png"),
                        "kind": "rgb",
                        "query": "mug",
                    },
                    {
                        "path": str(expanded_dir / "view_0000_relevancy.png"),
                        "kind": "relevancy",
                        "query": "mug",
                    },
                    {
                        "path": str(expanded_dir / "view_0000_overlay.png"),
                        "kind": "overlay",
                        "query": "mug",
                    },
                ],
                "candidate_points": candidate_points,
                "bounding_regions": bounding_regions,
                "confidence": 0.8,
                "warnings": [],
            }
        ],
        "answer_summary": answer_summary,
        "warnings": [],
    }
    (task_dir / "scene_query_report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir


def _write_viewer_fallback_report(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    task_dir = run_dir / "queries" / "mug"
    expanded_dir = task_dir / "mug"
    expanded_dir.mkdir(parents=True)
    workflow = expanded_dir / "interactive_viewer_workflow.md"
    template = expanded_dir / "mug_manual_report_template.json"
    workflow.write_text("# Viewer fallback\n", encoding="utf-8")
    template.write_text("{}", encoding="utf-8")
    report = {
        "scene_name": "unit_scene",
        "task": "mug",
        "query_results": [
            {
                "query": "mug",
                "backend_name": "lerf",
                "config_path": "config.yml",
                "rendered_images": [
                    {"path": str(workflow), "kind": "viewer_fallback", "query": "mug"},
                    {"path": str(template), "kind": "manual_template", "query": "mug"},
                ],
                "candidate_points": [],
                "bounding_regions": [],
                "confidence": None,
                "warnings": ["Automated LERF rendering failed; wrote viewer fallback instructions."],
            }
        ],
        "answer_summary": {"support_level": "no_localization_evidence"},
        "warnings": [],
    }
    (task_dir / "scene_query_report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir
