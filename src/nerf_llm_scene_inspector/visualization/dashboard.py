"""Streamlit dashboard for reviewing runs and querying semantic NeRF backends."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner
from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend


def load_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Load a pipeline run directory into a Streamlit-friendly payload."""

    root = Path(run_dir)
    return {
        "run_dir": str(root),
        "pipeline_summary": _read_json(root / "pipeline_summary.json"),
        "environment_report": _read_json(root / "environment_report.json"),
        "scene_inspection": _read_json(root / "scene_data_inspection.json"),
        "evaluation_summary": _read_json(root / "evaluation" / "eval_summary.json"),
        "evaluation_table": _read_csv(root / "evaluation" / "eval_table.csv"),
        "annotation_template": _read_json(root / "annotation_template.json"),
        "portfolio_card": _read_text(root / "portfolio_result_card.md"),
        "project_report": _read_text(root / "project_report.md"),
        "images": collect_run_images(root),
        "query_reports": collect_query_reports(root),
        "missing": _missing_run_files(root),
    }


def collect_run_images(run_dir: str | Path, *, limit: int = 40) -> list[dict[str, str]]:
    """Collect representative run images without requiring Streamlit."""

    root = Path(run_dir)
    candidates: list[Path] = []
    candidates.extend(
        [
            root / "demo_assets" / "query_grid.png",
            root / "demo_assets" / "demo_montage.gif",
        ]
    )
    for pattern in (
        "demo_assets/**/*overlay.png",
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
            reports.append({"path": str(path), "kind": "scene_query_report", "payload": payload})
    for path in sorted((root / "queries").rglob("query_result.json")):
        payload = _read_json(path)
        if payload:
            reports.append({"path": str(path), "kind": "query_result", "payload": payload})
    return reports


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

    st.subheader("Provenance")
    st.json(summary.get("provenance") or {})

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
    if bundle["environment_report"]:
        with st.expander("Environment Report"):
            st.json(bundle["environment_report"])


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
            st.json(report["payload"])


def _render_evaluation(st: Any, bundle: dict[str, Any]) -> None:
    st.subheader("Evaluation Summary")
    if bundle["evaluation_summary"]:
        st.json(bundle["evaluation_summary"])
    else:
        st.info("No evaluation summary found.")
    if bundle["evaluation_table"]:
        st.table(bundle["evaluation_table"])

    st.subheader("Annotation Template")
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
        labels = [region.label for region in result.bounding_regions[:5]]
        answer = plan.final_answer_template.format(items=", ".join(labels) if labels else query)
        st.subheader("Scene Answer")
        st.write(answer)
        report_path = Path(output_dir) / "dashboard_answer.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"answer": answer, "plan": plan.to_dict()}, indent=2), encoding="utf-8")


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


def _missing_run_files(run_dir: Path) -> list[str]:
    expected = [
        "pipeline_summary.json",
        "environment_report.json",
        "scene_data_inspection.json",
        "annotation_template.json",
        "evaluation/eval_summary.json",
    ]
    return [relative for relative in expected if not (run_dir / relative).exists()]


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
