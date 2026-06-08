#!/usr/bin/env python
"""Generate portfolio demo assets from example queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend  # noqa: E402
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend  # noqa: E402
from nerf_llm_scene_inspector.config import load_mapping  # noqa: E402
from nerf_llm_scene_inspector.evaluation.report import write_project_report  # noqa: E402
from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner  # noqa: E402
from nerf_llm_scene_inspector.backends.base import QueryResult  # noqa: E402
from nerf_llm_scene_inspector.querying.semantic_query import SemanticQueryEngine  # noqa: E402
from nerf_llm_scene_inspector.utils.paths import slugify  # noqa: E402
from nerf_llm_scene_inspector.visualization.make_video import make_mp4_or_gif  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/language_desk_scene/config.yml")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], default="lerf")
    parser.add_argument("--queries", default="examples/queries_demo.yaml")
    parser.add_argument("--output", default="results/demo_assets")
    parser.add_argument("--report-output", default="docs/project_report.md")
    parser.add_argument("--portfolio-card-output", default="docs/portfolio_result_card.md")
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument(
        "--planner-mode",
        choices=["planned", "direct"],
        default="planned",
        help="Use SemanticQueryEngine planning for demo tasks, or direct backend queries.",
    )
    parser.add_argument(
        "--include-tasks",
        action="store_true",
        help="Also run entries under the YAML tasks field.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_queries = load_mapping(args.queries)
    scene_name = str(raw_queries.get("scene_name", "desk_scene"))
    queries = _collect_demo_queries(raw_queries, include_tasks=args.include_tasks)
    if not queries:
        print("No queries found in query file.", file=sys.stderr)
        return 1
    backend = (
        LERFBackend(dry_run=args.dry_run, num_views=args.num_views)
        if args.backend == "lerf"
        else OpenNeRFBackend(dry_run=args.dry_run, num_views=args.num_views)
    )
    try:
        backend.load(args.config)
        if args.planner_mode == "planned":
            results, overlay_paths, scene_report_paths = _run_planned_queries(
                backend=backend,
                scene_name=scene_name,
                queries=queries,
                output=output,
                top_k=args.top_k,
                max_queries=args.max_queries,
            )
        else:
            results, overlay_paths = _run_direct_queries(
                backend=backend,
                queries=queries,
                output=output,
                top_k=args.top_k,
            )
            scene_report_paths = []
        video_path = None
        if overlay_paths:
            video_path = make_mp4_or_gif(overlay_paths, output / "demo_montage.gif")
        grid_path = _make_query_grid(overlay_paths, output / "query_grid.png")
        query_rows = [
            {
                "query": result.query,
                "target_description": "demo query",
                "topk_hit": "qualitative",
                "best_iou_2d": 0.0,
                "confidence": result.confidence if result.confidence is not None else "",
                "warnings": "; ".join(result.warnings),
            }
            for result in results
        ]
        write_project_report(
            args.report_output,
            title="NeRF-LLM Scene Inspector Report",
            scene_name=scene_name,
            backend=args.backend,
            query_rows=query_rows,
            metrics={"num_queries": len(results), "demo_video": str(video_path) if video_path else "not generated"},
            notes=[
                "Demo assets may be dry-run synthetic artifacts unless generated from a trained LERF config.",
                "This report is portfolio-ready but does not claim state-of-the-art performance.",
            ],
        )
        _write_portfolio_result_card(
            Path(args.portfolio_card_output),
            scene_name=scene_name,
            backend=args.backend,
            num_queries=len(queries),
            num_backend_results=len(results),
            planner_mode=args.planner_mode,
            grid_path=grid_path,
            video_path=video_path,
        )
    except Exception as exc:
        print(f"generate_demo_assets failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "scene_name": scene_name,
        "backend": args.backend,
        "planner_mode": args.planner_mode,
        "num_queries": len(results),
        "num_user_queries": len(queries),
        "num_backend_results": len(results),
        "output": str(output),
        "video": str(video_path) if video_path else None,
        "query_grid": str(grid_path) if grid_path else None,
        "scene_reports": [str(path) for path in scene_report_paths],
        "user_queries": queries,
        "results": [result.to_dict() for result in results],
    }
    summary_path = output / "demo_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _collect_demo_queries(raw_queries: dict[str, object], *, include_tasks: bool) -> list[str]:
    queries = [str(item) for item in raw_queries.get("queries", []) if str(item).strip()]
    if include_tasks:
        queries.extend(str(item) for item in raw_queries.get("tasks", []) if str(item).strip())
    return _dedupe_preserving_order(queries)


def _run_planned_queries(
    *,
    backend,
    scene_name: str,
    queries: list[str],
    output: Path,
    top_k: int,
    max_queries: int,
) -> tuple[list[QueryResult], list[Path], list[Path]]:
    engine = SemanticQueryEngine(
        backend=backend,
        planner=LocalRulePlanner(),
        top_k=top_k,
        max_queries=max_queries,
        scene_name=scene_name,
    )
    results: list[QueryResult] = []
    overlay_paths: list[Path] = []
    scene_report_paths: list[Path] = []
    for query in queries:
        task_dir = output / slugify(query)
        report = engine.run_task(query, task_dir)
        scene_report_paths.append(report.to_json(task_dir / "scene_query_report.json"))
        report.to_markdown(task_dir / "scene_query_report.md")
        results.extend(report.query_results)
        overlay_paths.extend(
            Path(view.path)
            for result in report.query_results
            for view in result.rendered_images
            if view.kind == "overlay"
        )
    return results, overlay_paths, scene_report_paths


def _run_direct_queries(
    *,
    backend,
    queries: list[str],
    output: Path,
    top_k: int,
) -> tuple[list[QueryResult], list[Path]]:
    results: list[QueryResult] = []
    overlay_paths: list[Path] = []
    for query in queries:
        result = backend.query_text(query, str(output / slugify(query)), top_k=top_k)
        results.append(result)
        overlay_paths.extend(Path(view.path) for view in result.rendered_images if view.kind == "overlay")
    return results, overlay_paths


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _make_query_grid(image_paths: list[Path], output_path: Path, max_columns: int = 2) -> Path | None:
    existing = [path for path in image_paths if path.exists()]
    if not existing:
        return None
    thumbs = []
    target_width = 720
    for path in existing:
        image = Image.open(path).convert("RGB")
        scale = target_width / image.width
        thumb = image.resize((target_width, int(image.height * scale)))
        thumbs.append((path, thumb))

    columns = min(max_columns, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = target_width
    cell_h = max(image.height for _path, image in thumbs) + 34
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, image) in enumerate(thumbs):
        col = index % columns
        row = index // columns
        x = col * cell_w
        y = row * cell_h
        label = path.parent.name.replace("_", " ")
        draw.text((x + 10, y + 10), label[:80], fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y + 34))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _write_portfolio_result_card(
    path: Path,
    *,
    scene_name: str,
    backend: str,
    num_queries: int,
    num_backend_results: int,
    planner_mode: str,
    grid_path: Path | None,
    video_path: Path | None,
) -> None:
    lines = [
        "# NeRF-LLM Scene Inspector: Portfolio Result Card",
        "",
        "NeRF-LLM Scene Inspector is a research engineering project that connects",
        "Nerfstudio-style scene reconstruction with LERF-style language-field querying.",
        "It demonstrates how phone video or images can be converted into a 3D scene",
        "representation that supports open-vocabulary text queries and structured reports.",
        "The current checked-in demo is synthetic dry-run output; real results require",
        "a trained semantic field and an NVIDIA GPU environment.",
        "",
        "## Architecture",
        "",
        "- Nerfstudio data processing and baseline NeRF training wrappers.",
        "- LERF primary backend with OpenNeRF secondary adapter, multi-view dry-run, and viewer repair fallback.",
        "- Deterministic local query planner for object, affordance, material, and relation prompts, with intent tags and relation-anchor provenance.",
        "- Typed QueryResult JSON, overlay generation, evaluation metrics, and report writing.",
        "- Annotation review contact sheets for checking manual `bbox_2d` labels before reporting metrics.",
        "- Offline annotation workbench for drawing bbox labels from query render artifacts and exporting filled JSON.",
        "- Workbench-merge tooling for converting filled browser annotations into validated evaluation JSON.",
        "- Annotation finalization workflow that refreshes evaluation, QA, scorecards, reports, result cards, portfolio pages, and optional portfolio packs after manual labels.",
        "- Capture manifests for device, lighting, camera motion, overlap, static-scene, and privacy metadata, with validation surfaced in run audit and evidence scoring.",
        "- Real-scene pipeline runner with environment reports and processed-scene validation.",
        "- Real-run preflight checks for capture inputs, upstream tools, CUDA, backend registration, processed scenes, and config paths.",
        "- Failure-diagnostics report generation that classifies CUDA, LERF registration, COLMAP/FFmpeg, missing-config, and viewer-fallback issues from saved logs.",
        "- Manual viewer-output import and full scene-query repair for recovering structured evidence when automated LERF rendering falls back to the interactive viewer.",
        "- Conservative evidence scorecard that separates dry-run smoke demos from real portfolio-ready runs.",
        "- Multi-run comparison report for ranking repeated captures or training attempts before selecting a portfolio candidate.",
        "- Experiment-matrix runner for small backend/query/variant ablations with CSV/Markdown summaries, candidate status, readiness gates, diagnostics, and blocking reasons.",
        "- Paper-style research report generation from run artifacts, metrics, limitations, and next steps.",
        "- Submission packet generation for claim-calibrated CV, portfolio, and professor-outreach sharing.",
        "- Run result-card generation that summarizes one run into a concise takeaway, evidence snapshot, safe blurb, limitations, and next actions.",
        "- Real-run action-plan generation that lists capture, CUDA/LERF training, annotation review, quality-gate, pack validation, and sharing commands.",
        "- Run-readiness gate generation that consolidates real-run launch and external-review decisions.",
        "- Claim-audit generation that checks external-facing text for unsupported SOTA, novelty, production, benchmark, or robotics-policy claims.",
        "- Static project-level portfolio site for GitHub Pages or local review.",
        "- Static HTML portfolio page with evidence score, metrics, visual artifacts, and artifact links.",
        "",
        "## Implemented",
        "",
        f"- Scene name: `{scene_name}`",
        f"- Backend: `{backend}`",
        f"- Demo queries generated: `{num_queries}`",
        f"- Demo backend calls rendered: `{num_backend_results}`",
        f"- Demo planner mode: `{planner_mode}`",
        "- CPU-only dry-run demo with mock RGB/relevancy/overlay outputs.",
        "- Real-mode wrappers for Nerfstudio/LERF commands when upstream tools are installed.",
        "- Run-scoped pipeline artifacts avoid stale query/evaluation results across reruns.",
        "- Shareable preflight, audit, recommendation, and reproduction artifacts for portfolio review.",
        "- Research report artifacts summarize evidence, limitations, and next steps in a paper-style format.",
        "- Submission checklist records allowed claims, claims to avoid, pack validation status, and next actions.",
        "- Real-run plan records the exact next commands needed to turn smoke evidence into a real captured-scene run.",
        "- Claim audit records whether README/docs/run artifacts stay within supported research-engineering claims.",
        "- Run evidence scorecard summarizes capture readiness, query artifacts, overlays, annotation coverage, evaluation metrics, and reproducibility files.",
        "",
        "## Dry-Run vs Real GPU Mode",
        "",
        "- Dry-run mode validates pipeline structure and produces synthetic visual artifacts.",
        "- Real mode runs Nerfstudio processing/training and attempts LERF internal relevancy rendering.",
        "- Viewer fallback artifacts are generated when upstream internals are incompatible.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m pip install -e \".[dev,video]\"",
        "python scripts/run_dry_run_demo.py",
        "python scripts/run_scene_pipeline.py --dry-run",
        "python scripts/compare_runs.py --root results/pipeline_runs",
        "python scripts/run_experiment_matrix.py --config examples/experiment_matrix.yaml --dry-run --limit 1",
        "python scripts/create_annotation_workbench.py --annotations results/pipeline_runs/desk_scene/annotation_template.json --results results/pipeline_runs/desk_scene/queries --output results/pipeline_runs/desk_scene/evaluation/annotation_workbench",
        "python scripts/finalize_annotations.py --run-dir results/pipeline_runs/desk_scene --filled results/pipeline_runs/desk_scene/evaluation/annotation_workbench/annotation_seed.json --profile smoke --export-pack --zip-pack",
        "python scripts/generate_research_report.py --run-dir results/pipeline_runs/desk_scene",
        "python scripts/diagnose_run_failures.py --run-dir results/pipeline_runs/desk_scene",
        "python scripts/create_run_result_card.py --run-dir results/pipeline_runs/desk_scene",
        "python scripts/generate_project_site.py --run-index results/pipeline_runs/run_index.json",
        "python scripts/validate_portfolio_pack.py --pack results/portfolio_pack",
        "python scripts/validate_portfolio_pack.py --pack results/portfolio_pack.zip",
        "python scripts/create_run_readiness.py --run-dir results/pipeline_runs/desk_scene --pack results/portfolio_pack",
        "python scripts/create_real_run_plan.py --run-dir results/pipeline_runs/desk_scene --output results/real_run_plan --input path/to/video.mp4 --type video --submission-packet results/pipeline_runs/desk_scene/submission_packet/submission_packet.json",
        "```",
        "",
        "## Outputs",
        "",
        f"- Query grid: `{grid_path}`" if grid_path else "- Query grid: not generated",
        f"- Demo montage: `{video_path}`" if video_path else "- Demo montage: not generated",
        "- Project site: `docs/index.html`",
        "- Pipeline summary: `results/pipeline_runs/desk_scene/pipeline_summary.json`",
        "- Run comparison: `results/pipeline_runs/run_comparison.md`",
        "- Capture manifest: `results/pipeline_runs/desk_scene/capture_manifest.md`",
        "- Capture validation: `results/pipeline_runs/desk_scene/capture_manifest_validation.md`",
        "- Preflight report: `results/pipeline_runs/desk_scene/preflight_report.md`",
        "- Failure diagnostics: `results/pipeline_runs/desk_scene/failure_diagnostics.md`",
        "- Evidence scorecard: `results/pipeline_runs/desk_scene/evidence_scorecard.md`",
        "- Claim audit: `results/pipeline_runs/desk_scene/claim_audit.md`",
        "- Run result card: `results/pipeline_runs/desk_scene/run_result_card.md`",
        "- Research report: `results/pipeline_runs/desk_scene/research_report.md`",
        "- Real-run action plan: `results/pipeline_runs/desk_scene/real_run_plan/real_run_plan.md`",
        "- Run readiness gate: `results/pipeline_runs/desk_scene/run_readiness.md`",
        "- Submission checklist: `results/pipeline_runs/desk_scene/submission_packet/submission_checklist.md`",
        "- Static portfolio page: `results/pipeline_runs/desk_scene/portfolio_page.html`",
        "- Annotation review: `results/pipeline_runs/desk_scene/evaluation/annotation_review.md`",
        "- Annotation workbench: `results/pipeline_runs/desk_scene/evaluation/annotation_workbench/annotation_workbench.html`",
        "- Annotation finalization: `results/pipeline_runs/desk_scene/annotation_finalize_report.md`",
        "- Scene inspection: `results/pipeline_runs/desk_scene/scene_data_inspection.md`",
        "- Scene relations: `results/pipeline_runs/desk_scene/scene_relations/scene_relations_report.md`",
        "- Run-scoped evaluation: `results/pipeline_runs/desk_scene/evaluation/eval_summary.json`",
        "- Experiment matrix: `results/experiment_matrix/dry_run_semantic_backend_matrix/experiment_matrix_report.md`",
        "- Portfolio pack: `results/portfolio_pack.zip`",
        "- Evaluation summary: `results/evaluation/eval_summary.json`",
        "- Qualitative report: `results/evaluation/qualitative_report.md`",
        "",
        "`portfolio_page.html` surfaces the same sharing-readiness summary from the submission",
        "packet, so a reviewer can see the current send/no-send status and next action without",
        "opening the raw JSON first.",
        "",
        "## Limitations",
        "",
        "- This is not a new NeRF architecture or state-of-the-art segmentation model.",
        "- Dry-run outputs are synthetic and should not be interpreted as scene understanding results.",
        "- Real LERF quality depends on capture quality, camera poses, GPU training, and upstream versions.",
        "",
        "## CV Bullets",
        "",
        "- Built a reproducible open-vocabulary 3D scene inspection system using Nerfstudio-style reconstruction and LERF-style language-field querying.",
        "- Implemented deterministic query planning with intent tags and relation anchors, semantic relevancy artifacts, spatial/evaluation utilities, and CPU-only dry-run demos.",
        "",
        "## Cold-Email Paragraph",
        "",
        "I built NeRF-LLM Scene Inspector as a research engineering project connecting NeRF reconstruction, language-embedded radiance fields, and natural-language scene querying. The project focuses on reproducible wrappers, typed artifacts, visualization, and lightweight evaluation rather than claiming algorithmic novelty. I am interested in extending this kind of system toward embodied AI and physical scene understanding.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
