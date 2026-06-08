"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path


def write_project_report(
    path: str | Path,
    *,
    title: str,
    scene_name: str,
    backend: str,
    query_rows: list[dict[str, object]] | None = None,
    metrics: dict[str, object] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write a concise markdown project report."""

    query_rows = query_rows or []
    metrics = metrics or {}
    notes = notes or []
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        "This report summarizes a NeRF-LLM Scene Inspector run. The system is built on",
        "Nerfstudio and LERF and is intended as a reproducible research engineering demo.",
        "For real scenes, the pipeline records environment diagnostics and processed-scene",
        "inspection artifacts under `results/pipeline_runs/<scene>/`.",
        "",
        "## Scene",
        "",
        f"- Scene name: `{scene_name}`",
        f"- Backend: `{backend}`",
        "",
        "## Query Results",
        "",
        "| Query | Target | Status | Top-k Hit | Best IoU | Confidence | Warnings |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if query_rows:
        for row in query_rows:
            lines.append(
                "| {query} | {target} | {status} | {hit} | {iou} | {confidence} | {warnings} |".format(
                    query=row.get("query", ""),
                    target=row.get("target_description", ""),
                    status=row.get("evaluation_status", ""),
                    hit=_display_value(row.get("topk_hit", "")),
                    iou=_display_iou(row.get("best_iou_2d")),
                    confidence=row.get("confidence", ""),
                    warnings=str(row.get("warnings", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("| Pending | Pending | Pending | Pending | Pending | Pending | Pending |")

    lines.extend(["", "## Evaluation Summary", "", "| Metric | Value |", "| --- | --- |"])
    if metrics:
        for key, value in metrics.items():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("| Pending | Pending |")

    lines.extend(["", *_research_workflow_sections()])

    lines.extend(["", "## Notes", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append(
            "- This project demonstrates open-vocabulary 3D scene querying without claiming new state-of-the-art results."
        )
    lines.append("- Review `scene_data_inspection.md` before interpreting real trained outputs.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _research_workflow_sections() -> list[str]:
    """Return durable guidance that should stay in generated portfolio reports."""

    return [
        "## Scene Relation Analysis",
        "",
        "When `--analyze-relations` is enabled, the pipeline writes a deterministic relation graph",
        "under `results/pipeline_runs/<scene>/scene_relations/`. It summarizes query-derived",
        "entities, relation edges such as `near`, `left_of`, `likely_supports`, and",
        "`likely_contained_in`, and whether each edge came from `3d` points or `2d_fallback`",
        "rendered boxes. These relations are qualitative evidence, not learned physical-relation",
        "predictions.",
        "",
        "## Experiment Matrix",
        "",
        "For small ablations, run `scripts/run_experiment_matrix.py` with",
        "`examples/experiment_matrix.yaml`. The matrix report summarizes each configured pipeline",
        "run with evidence score, prompt stability, relation-edge count, localization metrics,",
        "failure diagnostics, readiness status, candidate status, blocking reasons, and links to",
        "run-scoped artifacts. This is intended for reproducible comparison across variants and",
        "for selecting the strongest portfolio run, not as a benchmark claim.",
        "",
        "## Annotation Workbench",
        "",
        "Run `scripts/create_annotation_workbench.py --annotations results/pipeline_runs/<scene>/annotation_template.json",
        "--results results/pipeline_runs/<scene>/queries --output results/pipeline_runs/<scene>/evaluation/annotation_workbench`",
        "to generate an offline HTML bbox-labeling workspace. The workbench copies query render",
        "images, preloads candidate boxes, and exports filled annotation JSON for validation, visual",
        "review, and evaluation. For run-scoped work, prefer the finalizer instead of manually",
        "chaining merge, validation, review, evaluation, quality gates, pack export, and submission",
        "updates.",
        "For a run-scoped refresh, use `scripts/finalize_annotations.py --run-dir",
        "results/pipeline_runs/<scene> --filled path/to/annotations_filled.json --profile real-run",
        "--export-pack --zip-pack` to merge labels and regenerate evaluation, QA, scorecards,",
        "reports, result cards, portfolio pages, reproduction bundles, pack validation, and",
        "submission materials.",
        "",
        "## Research Report",
        "",
        "Run `scripts/generate_research_report.py --run-dir results/pipeline_runs/<scene>` after a",
        "pipeline run to generate `research_report.md` and `research_report.json`. The report",
        "combines the evidence scorecard, evaluation metrics, prompt-sensitivity diagnostics,",
        "scene-relation analysis, reproducibility artifacts, limitations, and next steps into a",
        "paper-style project summary suitable for portfolio review.",
        "",
        "## Run Result Card",
        "",
        "Run `scripts/create_run_result_card.py --run-dir results/pipeline_runs/<scene>` to generate",
        "`run_result_card.md` and `run_result_card.json`. This one-page card gives a",
        "reviewer-facing takeaway, shareable blurb, evidence snapshot, metrics, limitations,",
        "checks, and next actions without claiming more than the run artifacts support.",
        "",
        "## Submission Packet",
        "",
        "Run `scripts/create_submission_packet.py --run-dir results/pipeline_runs/<scene> --pack",
        "results/portfolio_pack` after validating a portfolio pack. The output records share",
        "readiness, allowed claims, claims to avoid, recommended links, and next actions for CV or",
        "professor-outreach use. The JSON includes a `readiness_summary` block and the Markdown",
        "checklist includes a `Readiness Summary` section that surfaces failed checks, warning",
        "checks, pack status, and the single next action to take before sharing. Dry-run packets",
        "explicitly mark the run as smoke-demo evidence only.",
        "",
        "## Real-Run Action Plan",
        "",
        "Run `scripts/create_real_run_plan.py --run-dir results/pipeline_runs/<scene> --input",
        "path/to/video.mp4 --type video` to generate a concrete command playbook for moving from",
        "dry-run smoke evidence to a real CUDA/Nerfstudio/LERF scene run. The plan is an execution",
        "checklist, not a claim that the real run has already succeeded.",
        "",
        "## Claim Audit",
        "",
        "Run `scripts/audit_claims.py --run-dir results/pipeline_runs/<scene> --pack",
        "results/portfolio_pack` before sharing. The audit checks portfolio-facing text for",
        "unsupported SOTA, novelty, benchmark-superiority, production-readiness, or robotics-policy",
        "claims and verifies required disclaimers.",
    ]


def _display_value(value: object) -> str:
    return "n/a" if value in {"", None} else str(value)


def _display_iou(value: object) -> str:
    if value in {"", None}:
        return "n/a"
    return f"{float(value):.3f}"
