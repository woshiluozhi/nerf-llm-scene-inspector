# NeRF-LLM Scene Inspector: Portfolio Result Card

NeRF-LLM Scene Inspector is a research engineering project that connects
Nerfstudio-style scene reconstruction with LERF-style language-field querying.
It demonstrates how phone video or images can be converted into a 3D scene
representation that supports open-vocabulary text queries and structured reports.
The current checked-in demo is synthetic dry-run output; real results require
a trained semantic field and an NVIDIA GPU environment.

## Architecture

- Nerfstudio data processing and baseline NeRF training wrappers.
- LERF primary backend with OpenNeRF secondary adapter.
- Deterministic local query planner for object, affordance, material, and relation prompts.
- Typed QueryResult JSON, overlay generation, evaluation metrics, and report writing.
- Real-scene pipeline runner with environment reports and processed-scene validation.
- Real-run preflight checks for capture inputs, upstream tools, CUDA, backend registration, processed scenes, and config paths.

## Implemented

- Scene name: `desk_scene`
- Backend: `lerf`
- Demo queries generated: `5`
- CPU-only dry-run demo with mock RGB/relevancy/overlay outputs.
- Real-mode wrappers for Nerfstudio/LERF commands when upstream tools are installed.
- Run-scoped pipeline artifacts avoid stale query/evaluation results across reruns.
- Shareable preflight, audit, recommendation, and reproduction artifacts for portfolio review.

## Dry-Run vs Real GPU Mode

- Dry-run mode validates pipeline structure and produces synthetic visual artifacts.
- Real mode runs Nerfstudio processing/training and attempts LERF internal relevancy rendering.
- Viewer fallback artifacts are generated when upstream internals are incompatible.

## Reproduce

```bash
python -m pip install -e ".[dev,video]"
python scripts/run_dry_run_demo.py
python scripts/run_scene_pipeline.py --dry-run
python scripts/export_portfolio_pack.py --run-dir results/pipeline_runs/desk_scene --zip
```

## Outputs

- Query grid: `results\demo_assets\query_grid.png`
- Demo montage: `results\demo_assets\demo_montage.gif`
- Pipeline summary: `results/pipeline_runs/desk_scene/pipeline_summary.json`
- Preflight report: `results/pipeline_runs/desk_scene/preflight_report.md`
- Scene inspection: `results/pipeline_runs/desk_scene/scene_data_inspection.md`
- Run-scoped evaluation: `results/pipeline_runs/desk_scene/evaluation/eval_summary.json`
- Portfolio pack: `results/portfolio_pack.zip`
- Evaluation summary: `results/evaluation/eval_summary.json`
- Qualitative report: `results/evaluation/qualitative_report.md`

## Limitations

- This is not a new NeRF architecture or state-of-the-art segmentation model.
- Dry-run outputs are synthetic and should not be interpreted as scene understanding results.
- Real LERF quality depends on capture quality, camera poses, GPU training, and upstream versions.

## CV Bullets

- Built a reproducible open-vocabulary 3D scene inspection system using Nerfstudio-style reconstruction and LERF-style language-field querying.
- Implemented deterministic query planning, semantic relevancy artifacts, spatial/evaluation utilities, and CPU-only dry-run demos.

## Cold-Email Paragraph

I built NeRF-LLM Scene Inspector as a research engineering project connecting NeRF reconstruction, language-embedded radiance fields, and natural-language scene querying. The project focuses on reproducible wrappers, typed artifacts, visualization, and lightweight evaluation rather than claiming algorithmic novelty. I am interested in extending this kind of system toward embodied AI and physical scene understanding.
