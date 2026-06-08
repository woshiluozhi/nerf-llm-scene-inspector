"""Streamlit dashboard for reviewing runs and querying semantic NeRF backends."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner
from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend
from nerf_llm_scene_inspector.querying.answer_synthesis import synthesize_scene_answer


def load_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Load a pipeline run directory into a Streamlit-friendly payload."""

    root = Path(run_dir)
    pipeline_summary = _read_json(root / "pipeline_summary.json")
    return {
        "run_dir": str(root),
        "pipeline_summary": pipeline_summary,
        "run_index": _read_json(root.parent / "run_index.json"),
        "run_comparison": _read_json(root.parent / "run_comparison.json"),
        "run_comparison_markdown": _read_text(root.parent / "run_comparison.md"),
        "capture_manifest": _read_json(root / "capture_manifest.json"),
        "capture_manifest_markdown": _read_text(root / "capture_manifest.md"),
        "capture_manifest_validation": _read_json(root / "capture_manifest_validation.json"),
        "capture_manifest_validation_markdown": _read_text(root / "capture_manifest_validation.md"),
        "preflight_report": _read_json(root / "preflight_report.json"),
        "preflight_markdown": _read_text(root / "preflight_report.md"),
        "evidence_scorecard": _read_json(root / "evidence_scorecard.json"),
        "evidence_scorecard_markdown": _read_text(root / "evidence_scorecard.md"),
        "quality_gate": _read_json(root / "quality_gate.json"),
        "quality_gate_markdown": _read_text(root / "quality_gate.md"),
        "claim_audit": _read_json(root / "claim_audit.json"),
        "claim_audit_markdown": _read_text(root / "claim_audit.md"),
        "run_audit": _read_json(root / "run_audit.json"),
        "run_recommendations": _read_json(root / "run_recommendations.json"),
        "run_recommendations_markdown": _read_text(root / "run_recommendations.md"),
        "reproduction_manifest": _read_json(root / "reproduction_manifest.json"),
        "reproduction_report": _read_text(root / "reproduction_report.md"),
        "real_run_plan": _read_json(root / "real_run_plan" / "real_run_plan.json"),
        "real_run_plan_markdown": _read_text(root / "real_run_plan" / "real_run_plan.md"),
        "research_report": _read_json(root / "research_report.json"),
        "research_report_markdown": _read_text(root / "research_report.md"),
        "submission_packet": _read_json(root / "submission_packet" / "submission_packet.json"),
        "submission_checklist": _read_text(root / "submission_packet" / "submission_checklist.md"),
        "submission_cv_entry": _read_text(root / "submission_packet" / "cv_project_entry.md"),
        "submission_email_brief": _read_text(root / "submission_packet" / "professor_email_brief.md"),
        "environment_report": _read_json(root / "environment_report.json"),
        "scene_inspection": _read_json(root / "scene_data_inspection.json"),
        "training_summaries": {
            "baseline": _read_json(root / "training" / "baseline_train_summary.json"),
            "language": _read_json(root / "training" / "language_train_summary.json"),
        },
        "annotation_validation": _read_json(root / "evaluation" / "annotation_validation.json"),
        "annotation_review": _read_json(root / "evaluation" / "annotation_review.json"),
        "annotation_review_markdown": _read_text(root / "evaluation" / "annotation_review.md"),
        "evaluation_summary": _read_json(root / "evaluation" / "eval_summary.json"),
        "evaluation_table": _read_csv(root / "evaluation" / "eval_table.csv"),
        "prompt_sensitivity": _read_json(
            root / "prompt_sensitivity" / "prompt_sensitivity_summary.json"
        ),
        "prompt_sensitivity_markdown": _read_text(
            root / "prompt_sensitivity" / "prompt_sensitivity_report.md"
        ),
        "scene_relations": _read_json(root / "scene_relations" / "scene_relations_summary.json"),
        "scene_relations_table": _read_csv(root / "scene_relations" / "scene_relations_edges.csv"),
        "scene_relations_markdown": _read_text(root / "scene_relations" / "scene_relations_report.md"),
        "annotation_template": _read_json(root / "annotation_template.json"),
        "portfolio_card": _read_text(root / "portfolio_result_card.md"),
        "portfolio_page": str(root / "portfolio_page.html") if (root / "portfolio_page.html").exists() else "",
        "project_report": _read_text(root / "project_report.md"),
        "command_logs": collect_command_logs(root),
        "images": collect_run_images(root),
        "query_reports": collect_query_reports(root),
        "missing": _missing_run_files(root, pipeline_summary),
    }


def collect_run_images(run_dir: str | Path, *, limit: int = 40) -> list[dict[str, str]]:
    """Collect representative run images without requiring Streamlit."""

    root = Path(run_dir)
    candidates: list[Path] = []
    candidates.extend(
        [
            root / "demo_assets" / "query_grid.png",
            root / "demo_assets" / "demo_montage.gif",
            root / "evaluation" / "annotation_review_contact_sheet.png",
        ]
    )
    for pattern in (
        "demo_assets/**/*overlay.png",
        "evaluation/annotation_review_images/*.png",
        "queries/**/*overlay.png",
        "queries/**/*relevancy.png",
        "queries/**/*rgb.png",
    ):
        candidates.extend(sorted(root.glob(pattern)))
    images: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in candidates:
        if len(images) >= limit:
            break
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        images.append(
            {
                "path": str(path),
                "label": _relative_label(path, root),
                "kind": _image_kind(path),
            }
        )
    return images


def collect_query_reports(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load query report JSON files produced by the pipeline."""

    root = Path(run_dir)
    reports: list[dict[str, Any]] = []
    for path in sorted((root / "queries").rglob("scene_query_report.json")):
        payload = _read_json(path)
        if payload:
            reports.append(
                {
                    "path": str(path),
                    "kind": "scene_query_report",
                    "payload": payload,
                    "markdown": _read_text(path.with_suffix(".md")),
                }
            )
    for path in sorted((root / "queries").rglob("query_result.json")):
        payload = _read_json(path)
        if payload:
            reports.append({"path": str(path), "kind": "query_result", "payload": payload})
    return reports


def collect_command_logs(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load run-scoped command logs produced by the pipeline."""

    root = Path(run_dir)
    logs: list[dict[str, Any]] = []
    for path in sorted((root / "logs").glob("*.json")):
        payload = _read_json(path)
        if payload:
            logs.append({"path": str(path), "label": _relative_label(path, root), "payload": payload})
    return logs


def main() -> None:
    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
        raise SystemExit("Install dashboard dependencies with: pip install -e .[dashboard]") from exc

    st.set_page_config(page_title="NeRF-LLM Scene Inspector", layout="wide")
    st.title("NeRF-LLM Scene Inspector")
    run_dir = st.sidebar.text_input("Pipeline run directory", value="results/pipeline_runs/desk_scene")
    config_path = st.sidebar.text_input("Config path", value="runs/language_desk_scene/config.yml")
    backend_name = st.sidebar.selectbox("Backend", options=["lerf", "opennerf"])
    dry_run = st.sidebar.checkbox("Dry query run", value=True)

    bundle = load_run_bundle(run_dir)
    tabs = st.tabs(["Run Review", "Artifacts", "Evaluation", "Query Runner"])
    with tabs[0]:
        _render_run_review(st, bundle)
    with tabs[1]:
        _render_artifacts(st, bundle)
    with tabs[2]:
        _render_evaluation(st, bundle)
    with tabs[3]:
        _render_query_runner(st, config_path, backend_name, dry_run)


def _render_run_review(st: Any, bundle: dict[str, Any]) -> None:
    summary = bundle["pipeline_summary"]
    if bundle["missing"]:
        st.warning("Missing expected run files: " + ", ".join(bundle["missing"]))
    if not summary:
        st.info("Run summary not found. Run scripts/run_scene_pipeline.py first.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Success", str(summary.get("success")))
    col_b.metric("Backend", str(summary.get("backend", "unknown")))
    col_c.metric("Dry Run", str(summary.get("dry_run")))
    col_d.metric("Queries", str(len(summary.get("queries") or [])))

    if bundle["run_audit"]:
        audit = bundle["run_audit"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Audit", str(audit.get("status", "unknown")))
        col_b.metric("Audit Score", str(audit.get("score", "unknown")))
        col_c.metric("Findings", str(len(audit.get("findings") or [])))
        with st.expander("Run Audit Findings", expanded=audit.get("status") != "ready"):
            st.json(audit)
    if bundle["run_recommendations"]:
        recommendations = bundle["run_recommendations"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Readiness", str(recommendations.get("readiness_level", "unknown")))
        col_b.metric("Critical Actions", str(recommendations.get("critical_count", 0)))
        col_c.metric("High Actions", str(recommendations.get("high_count", 0)))
        with st.expander("Recommended Next Steps", expanded=True):
            if bundle["run_recommendations_markdown"]:
                st.markdown(bundle["run_recommendations_markdown"])
            else:
                st.json(recommendations)
    if bundle["run_index"]:
        index = bundle["run_index"]
        with st.expander("Run Index"):
            st.json(
                {
                    "total_runs": index.get("total_runs"),
                    "successful_runs": index.get("successful_runs"),
                    "ready_runs": index.get("ready_runs"),
                    "entries": index.get("entries"),
                }
            )
    if bundle["run_comparison"]:
        comparison = bundle["run_comparison"]
        best_run = comparison.get("best_run") if isinstance(comparison.get("best_run"), dict) else {}
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Best Run", str(best_run.get("scene_name", "unknown")))
        col_b.metric("Best Status", str(best_run.get("selection_status", "unknown")))
        col_c.metric("Candidates", str(comparison.get("portfolio_candidate_count", 0)))
        with st.expander("Run Comparison", expanded=not comparison.get("portfolio_candidate_count")):
            if bundle["run_comparison_markdown"]:
                st.markdown(bundle["run_comparison_markdown"])
            else:
                st.json(comparison)

    if bundle["preflight_report"]:
        preflight = bundle["preflight_report"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Preflight", str(preflight.get("status", "unknown")))
        col_b.metric("Preflight Fails", str(preflight.get("fail_count", 0)))
        col_c.metric("Preflight Warnings", str(preflight.get("warn_count", 0)))
        with st.expander("Real-Run Preflight", expanded=preflight.get("status") == "blocked"):
            if bundle["preflight_markdown"]:
                st.markdown(bundle["preflight_markdown"])
            else:
                st.json(preflight)
    if bundle["capture_manifest_validation"]:
        validation = bundle["capture_manifest_validation"]
        with st.expander("Capture Manifest", expanded=validation.get("status") != "ready"):
            if bundle["capture_manifest_markdown"]:
                st.markdown(bundle["capture_manifest_markdown"])
            if bundle["capture_manifest_validation_markdown"]:
                st.markdown(bundle["capture_manifest_validation_markdown"])
            else:
                st.json(validation)

    if bundle["evidence_scorecard"]:
        scorecard = bundle["evidence_scorecard"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Evidence", str(scorecard.get("evidence_level", "unknown")))
        col_b.metric("Evidence Score", f"{scorecard.get('score', 0)}/{scorecard.get('max_score', 100)}")
        col_c.metric("Overlays", str(scorecard.get("overlay_count", 0)))
        with st.expander("Evidence Scorecard", expanded=scorecard.get("evidence_level") == "blocked"):
            if bundle["evidence_scorecard_markdown"]:
                st.markdown(bundle["evidence_scorecard_markdown"])
            else:
                st.json(scorecard)
    if bundle["quality_gate"]:
        gate = bundle["quality_gate"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Quality Gate", str(gate.get("status", "unknown")))
        col_b.metric("Gate Profile", str(gate.get("profile", "unknown")))
        col_c.metric("Gate Findings", f"{gate.get('fail_count', 0)} fail / {gate.get('warn_count', 0)} warn")
        with st.expander("Quality Gate", expanded=gate.get("status") == "fail"):
            if bundle["quality_gate_markdown"]:
                st.markdown(bundle["quality_gate_markdown"])
            else:
                st.json(gate)
    if bundle["claim_audit"]:
        audit = bundle["claim_audit"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Claim Audit", str(audit.get("status", "unknown")))
        col_b.metric("Claim Fails", str(audit.get("fail_count", 0)))
        col_c.metric("Claim Warnings", str(audit.get("warn_count", 0)))
        with st.expander("Claim Audit", expanded=audit.get("status") == "fail"):
            if bundle["claim_audit_markdown"]:
                st.markdown(bundle["claim_audit_markdown"])
            else:
                st.json(audit)
    if bundle["portfolio_page"]:
        st.markdown(f"[Open static portfolio page]({bundle['portfolio_page']})")

    st.subheader("Provenance")
    st.json(summary.get("provenance") or {})
    if bundle["reproduction_manifest"]:
        with st.expander("Reproduction Recipe", expanded=False):
            if bundle["reproduction_report"]:
                st.markdown(bundle["reproduction_report"])
            else:
                st.json(bundle["reproduction_manifest"])
    if bundle["real_run_plan_markdown"]:
        with st.expander("Real-Run Action Plan", expanded=False):
            st.markdown(bundle["real_run_plan_markdown"])
    elif bundle["real_run_plan"]:
        with st.expander("Real-Run Action Plan", expanded=False):
            st.json(bundle["real_run_plan"])
    if bundle["research_report_markdown"]:
        with st.expander("Research Report", expanded=False):
            st.markdown(bundle["research_report_markdown"])
    elif bundle["research_report"]:
        with st.expander("Research Report", expanded=False):
            st.json(bundle["research_report"])
    if bundle["submission_checklist"]:
        with st.expander("Submission Checklist", expanded=False):
            st.markdown(bundle["submission_checklist"])
            if bundle["submission_cv_entry"]:
                st.markdown("### CV Entry")
                st.markdown(bundle["submission_cv_entry"])
            if bundle["submission_email_brief"]:
                st.markdown("### Outreach Brief")
                st.markdown(bundle["submission_email_brief"])
    elif bundle["submission_packet"]:
        with st.expander("Submission Packet", expanded=False):
            st.json(bundle["submission_packet"])

    st.subheader("Step Status")
    rows = [
        {
            "step": step.get("name"),
            "status": step.get("status"),
            "error": step.get("error") or "",
        }
        for step in summary.get("steps", [])
    ]
    st.table(rows)

    if bundle["scene_inspection"]:
        st.subheader("Scene Data Inspection")
        st.json(bundle["scene_inspection"])
    training_summaries = {
        key: value for key, value in bundle["training_summaries"].items() if value
    }
    if training_summaries:
        st.subheader("Training Summaries")
        for name, payload in training_summaries.items():
            with st.expander(f"{name.title()} Training Summary"):
                st.json(payload)
    if bundle["environment_report"]:
        with st.expander("Environment Report"):
            st.json(bundle["environment_report"])
    if bundle["command_logs"]:
        st.subheader("Command Logs")
        for command_log in bundle["command_logs"]:
            payload = command_log["payload"]
            status = "ok" if payload.get("returncode") == 0 else "failed"
            with st.expander(f"{command_log['label']} ({status})"):
                st.json(payload)


def _render_artifacts(st: Any, bundle: dict[str, Any]) -> None:
    images = bundle["images"]
    st.subheader("Visual Artifacts")
    if not images:
        st.info("No rendered images found in this run directory.")
    for index in range(0, len(images), 2):
        cols = st.columns(2)
        for column, image in zip(cols, images[index : index + 2]):
            column.image(image["path"], caption=image["label"], use_container_width=True)

    st.subheader("Query Reports")
    reports = bundle["query_reports"]
    if not reports:
        st.info("No query report JSON files found.")
    for report in reports[:20]:
        with st.expander(f"{report['kind']}: {_relative_label(report['path'], bundle['run_dir'])}"):
            if report.get("markdown"):
                st.markdown(report["markdown"])
            st.json(report["payload"])


def _render_evaluation(st: Any, bundle: dict[str, Any]) -> None:
    st.subheader("Evaluation Summary")
    if bundle["evaluation_summary"]:
        st.json(bundle["evaluation_summary"])
    else:
        st.info("No evaluation summary found.")
    if bundle["evaluation_table"]:
        st.table(bundle["evaluation_table"])

    st.subheader("Prompt Sensitivity")
    if bundle["prompt_sensitivity_markdown"]:
        st.markdown(bundle["prompt_sensitivity_markdown"])
    elif bundle["prompt_sensitivity"]:
        st.json(bundle["prompt_sensitivity"])
    else:
        st.info("No prompt-sensitivity report found.")

    st.subheader("Scene Relations")
    if bundle["scene_relations_markdown"]:
        st.markdown(bundle["scene_relations_markdown"])
    elif bundle["scene_relations"]:
        st.json(bundle["scene_relations"])
    else:
        st.info("No scene-relation report found. Run the pipeline with --analyze-relations.")
    if bundle["scene_relations_table"]:
        with st.expander("Relation Edge Table"):
            st.table(bundle["scene_relations_table"])

    st.subheader("Annotation Template")
    if bundle["annotation_validation"]:
        with st.expander("Annotation Validation"):
            st.json(bundle["annotation_validation"])
    if bundle["annotation_review"]:
        with st.expander("Annotation Review", expanded=True):
            if bundle["annotation_review_markdown"]:
                st.markdown(bundle["annotation_review_markdown"])
            else:
                st.json(bundle["annotation_review"])
    if bundle["annotation_template"]:
        st.json(bundle["annotation_template"])
    else:
        st.info("No annotation_template.json found.")

    for title, body in (
        ("Portfolio Result Card", bundle["portfolio_card"]),
        ("Project Report", bundle["project_report"]),
    ):
        if body:
            with st.expander(title):
                st.markdown(body)


def _render_query_runner(st: Any, config_path: str, backend_name: str, dry_run: bool) -> None:
    query = st.text_input("Text query", value="mug")
    output_dir = st.text_input("Output directory", value="results/dashboard_query")

    planner = LocalRulePlanner()
    plan = planner.plan(query)
    st.subheader("Planner")
    st.json(plan.to_dict())

    if st.button("Run query"):
        backend = LERFBackend(dry_run=dry_run) if backend_name == "lerf" else OpenNeRFBackend(dry_run=dry_run)
        backend.load(config_path)
        result = backend.query_text(query, output_dir, top_k=5)
        st.subheader("QueryResult JSON")
        st.json(result.to_dict())
        for view in result.rendered_images:
            if Path(view.path).exists() and view.kind in {"overlay", "relevancy", "rgb"}:
                st.image(view.path, caption=view.caption or view.kind, use_container_width=True)
        answer = synthesize_scene_answer(
            task=query,
            plan=plan.to_dict(),
            results=[result],
            top_k=5,
        )
        st.subheader("Scene Answer")
        st.write(answer.answer)
        report_path = Path(output_dir) / "dashboard_answer.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"answer": answer.answer, "answer_summary": answer.to_dict(), "plan": plan.to_dict()}, indent=2),
            encoding="utf-8",
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _missing_run_files(run_dir: Path, pipeline_summary: dict[str, Any] | None = None) -> list[str]:
    expected = [
        "pipeline_summary.json",
        "capture_manifest.json",
        "capture_manifest_validation.json",
        "preflight_report.json",
        "evidence_scorecard.json",
        "quality_gate.json",
        "claim_audit.json",
        "run_audit.json",
        "run_recommendations.json",
        "reproduction_manifest.json",
        "real_run_plan/real_run_plan.json",
        "research_report.json",
        "submission_packet/submission_packet.json",
        "environment_report.json",
        "scene_data_inspection.json",
        "annotation_template.json",
        "evaluation/annotation_validation.json",
        "evaluation/eval_summary.json",
        "portfolio_page.html",
    ]
    if _step_succeeded(pipeline_summary, "review_annotations"):
        expected.extend(
            [
                "evaluation/annotation_review.json",
                "evaluation/annotation_review.md",
            ]
        )
    if _step_succeeded(pipeline_summary, "analyze_prompt_sensitivity"):
        expected.extend(
            [
                "prompt_sensitivity/prompt_sensitivity_summary.json",
                "prompt_sensitivity/prompt_sensitivity_report.md",
            ]
        )
    if _step_succeeded(pipeline_summary, "analyze_scene_relations"):
        expected.extend(
            [
                "scene_relations/scene_relations_summary.json",
                "scene_relations/scene_relations_edges.csv",
                "scene_relations/scene_relations_report.md",
            ]
        )
    if _step_succeeded(pipeline_summary, "train_baseline_nerf"):
        expected.append("training/baseline_train_summary.json")
    if _step_succeeded(pipeline_summary, "train_language_field"):
        expected.append("training/language_train_summary.json")
    if _step_succeeded(pipeline_summary, "generate_research_report"):
        expected.append("research_report.md")
    if _step_succeeded(pipeline_summary, "create_real_run_plan"):
        expected.append("real_run_plan/real_run_plan.md")
    if _step_succeeded(pipeline_summary, "audit_claims"):
        expected.append("claim_audit.md")
    if _step_succeeded(pipeline_summary, "create_submission_packet"):
        expected.extend(
            [
                "submission_packet/submission_checklist.md",
                "submission_packet/cv_project_entry.md",
                "submission_packet/professor_email_brief.md",
            ]
        )
    return [relative for relative in expected if not (run_dir / relative).exists()]


def _step_succeeded(pipeline_summary: dict[str, Any] | None, step_name: str) -> bool:
    if not pipeline_summary:
        return False
    for step in pipeline_summary.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == step_name:
            return step.get("status") == "success"
    return False


def _relative_label(path: str | Path, root: str | Path) -> str:
    path_obj = Path(path)
    root_obj = Path(root)
    try:
        return str(path_obj.relative_to(root_obj)).replace("\\", "/")
    except ValueError:
        return path_obj.name


def _image_kind(path: Path) -> str:
    name = path.name.lower()
    if "overlay" in name:
        return "overlay"
    if "relevancy" in name:
        return "relevancy"
    if "rgb" in name:
        return "rgb"
    if path.suffix.lower() == ".gif":
        return "montage"
    return "image"


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
