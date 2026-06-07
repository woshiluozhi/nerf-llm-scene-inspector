"""Minimal Streamlit dashboard for semantic NeRF querying."""

from __future__ import annotations

import json
from pathlib import Path

from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner
from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend


def main() -> None:
    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
        raise SystemExit("Install dashboard dependencies with: pip install -e .[dashboard]") from exc

    st.set_page_config(page_title="NeRF-LLM Scene Inspector", layout="wide")
    st.title("NeRF-LLM Scene Inspector")
    config_path = st.text_input("Config path", value="runs/language_desk_scene/config.yml")
    backend_name = st.selectbox("Backend", options=["lerf", "opennerf"])
    query = st.text_input("Text query", value="mug")
    dry_run = st.checkbox("Dry run", value=True)
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
                st.image(view.path, caption=view.caption or view.kind)
        labels = [region.label for region in result.bounding_regions[:5]]
        answer = plan.final_answer_template.format(items=", ".join(labels) if labels else query)
        st.subheader("Scene Answer")
        st.write(answer)
        report_path = Path(output_dir) / "dashboard_answer.json"
        report_path.write_text(json.dumps({"answer": answer, "plan": plan.to_dict()}, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
