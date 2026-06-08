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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_queries = load_mapping(args.queries)
    scene_name = str(raw_queries.get("scene_name", "desk_scene"))
    queries = [str(item) for item in raw_queries.get("queries", [])]
    if not queries:
        print("No queries found in query file.", file=sys.stderr)
        return 1
    backend = (
        LERFBackend(dry_run=args.dry_run, num_views=args.num_views)
        if args.backend == "lerf"
        else OpenNeRFBackend(dry_run=args.dry_run)
    )
    try:
        backend.load(args.config)
        results = []
        overlay_paths: list[Path] = []
        for query in queries:
            result = backend.query_text(query, str(output / slugify(query)), top_k=5)
            results.append(result)
            overlay_paths.extend(Path(view.path) for view in result.rendered_images if view.kind == "overlay")
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
            num_queries=len(results),
            grid_path=grid_path,
            video_path=video_path,
        )
    except Exception as exc:
        print(f"generate_demo_assets failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "scene_name": scene_name,
        "backend": args.backend,
        "num_queries": len(results),
        "output": str(output),
        "video": str(video_path) if video_path else None,
        "query_grid": str(grid_path) if grid_path else None,
        "results": [result.to_dict() for result in results],
    }
    summary_path = output / "demo_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


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
        "- LERF primary backend with OpenNeRF secondary adapter.",
        "- Deterministic local query planner for object, affordance, material, and relation prompts.",
        "- Typed QueryResult JSON, overlay generation, evaluation metrics, and report writing.",
        "- Prompt-sensitivity diagnostics and scene-relation graph reports.",
        "- Annotation review contact sheets for checking manual `bbox_2d` labels before reporting metrics.",
        "- Capture manifests and real-run preflight checks for reproducible scene collection.",
        "- Experiment-matrix runner for small backend/query/variant ablations with CSV and Markdown summaries.",
        "- Paper-style research report generation from run artifacts, metrics, limitations, and next steps.",
        "- Real-scene pipeline runner with environment reports and processed-scene validation.",
        "- Conservative evidence scorecard and multi-run comparison for selecting portfolio candidates.",
        "- Static project/run-level portfolio pages and share-safe portfolio packs.",
        "",
        "## Implemented",
        "",
        f"- Scene name: `{scene_name}`",
        f"- Backend: `{backend}`",
        f"- Demo queries generated: `{num_queries}`",
        "- CPU-only dry-run demo with mock RGB/relevancy/overlay outputs.",
        "- Real-mode wrappers for Nerfstudio/LERF commands when upstream tools are installed.",
        "- Run-scoped pipeline artifacts avoid stale query/evaluation results across reruns.",
        "- Shareable preflight, audit, recommendation, and reproduction artifacts for portfolio review.",
        "- Research report artifacts summarize evidence, limitations, and next steps in a paper-style format.",
        "- Run evidence scorecard summarizes capture readiness, query artifacts, annotations, and evaluation metrics.",
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
        "python scripts/run_experiment_matrix.py --config examples/experiment_matrix.yaml --dry-run --limit 1",
        "python scripts/generate_research_report.py --run-dir results/pipeline_runs/desk_scene",
        "python scripts/export_portfolio_pack.py --run-dir results/pipeline_runs/desk_scene --zip",
        "```",
        "",
        "## Outputs",
        "",
        f"- Query grid: `{grid_path}`" if grid_path else "- Query grid: not generated",
        f"- Demo montage: `{video_path}`" if video_path else "- Demo montage: not generated",
        "- Pipeline summary: `results/pipeline_runs/desk_scene/pipeline_summary.json`",
        "- Scene inspection: `results/pipeline_runs/desk_scene/scene_data_inspection.md`",
        "- Scene relations: `results/pipeline_runs/desk_scene/scene_relations/scene_relations_report.md`",
        "- Research report: `results/pipeline_runs/desk_scene/research_report.md`",
        "- Run-scoped evaluation: `results/pipeline_runs/desk_scene/evaluation/eval_summary.json`",
        "- Experiment matrix: `results/experiment_matrix/dry_run_semantic_backend_matrix/experiment_matrix_report.md`",
        "- Portfolio pack: `results/portfolio_pack.zip`",
        "- Evaluation summary: `results/evaluation/eval_summary.json`",
        "- Qualitative report: `results/evaluation/qualitative_report.md`",
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
        "- Implemented deterministic query planning, semantic relevancy artifacts, experiment-matrix summaries, paper-style research reports, spatial/evaluation utilities, and CPU-only dry-run demos.",
        "",
        "## Cold-Email Paragraph",
        "",
        "I built NeRF-LLM Scene Inspector as a research engineering project connecting NeRF reconstruction, language-embedded radiance fields, and natural-language scene querying. The project focuses on reproducible wrappers, typed artifacts, visualization, and lightweight evaluation rather than claiming algorithmic novelty. I am interested in extending this kind of system toward embodied AI and physical scene understanding.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
