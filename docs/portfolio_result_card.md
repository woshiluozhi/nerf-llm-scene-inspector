# NeRF-LLM Scene Inspector: Portfolio Result Card

NeRF-LLM Scene Inspector is a research engineering project that connects
Nerfstudio-style scene reconstruction with LERF-style language-field querying.
It demonstrates how phone video or images can be converted into a 3D scene
representation that supports open-vocabulary text queries and structured reports.
The current checked-in demo is synthetic dry-run output; real results require
a trained semantic field and an NVIDIA GPU environment.

## Architecture

- Nerfstudio data processing and baseline NeRF training wrappers.
- LERF primary backend with OpenNeRF secondary adapter, including multi-view dry-run artifacts and viewer-repair fallback.
- Deterministic local query planner for object, affordance, material, and relation prompts.
- Typed QueryResult JSON, overlay generation, evaluation metrics, and report writing.
- Annotation review contact sheets for checking manual `bbox_2d` labels before reporting metrics.
- Offline annotation workbench for drawing bbox labels from query render artifacts and exporting filled JSON.
- Workbench-merge tooling for converting filled browser annotations into validated evaluation JSON.
- Annotation finalization workflow that refreshes evaluation, QA, scorecards, reports, result cards, portfolio pages, and optional portfolio packs after manual labels.
- Capture manifests for device, lighting, camera motion, overlap, static-scene, and privacy metadata, with validation surfaced in run audit and evidence scoring.
- Real-scene pipeline runner with environment reports and processed-scene validation.
- Real-run preflight checks for capture inputs, upstream tools, CUDA, backend registration, processed scenes, and config paths.
- Failure-diagnostics report generation that classifies CUDA, LERF registration, COLMAP/FFmpeg, missing-config, and viewer-fallback issues from saved logs.
- Manual viewer-output import and full scene-query repair for recovering structured evidence when automated LERF rendering falls back to the interactive viewer.
- Conservative evidence scorecard that separates dry-run smoke demos from real portfolio-ready runs.
- Multi-run comparison report for ranking repeated captures or training attempts before selecting a portfolio candidate.
- Experiment-matrix runner for small backend/query/variant ablations with CSV/Markdown summaries, candidate status, readiness gates, diagnostics, and blocking reasons.
- Paper-style research report generation from run artifacts, metrics, limitations, and next steps.
- Submission packet generation for claim-calibrated CV, portfolio, and professor-outreach sharing.
- Run result-card generation that summarizes one run into a concise takeaway, evidence snapshot, safe blurb, limitations, and next actions.
- Real-run action-plan generation that lists capture, CUDA/LERF training, annotation review, quality-gate, pack validation, and sharing commands.
- Run-readiness gate generation that consolidates real-run launch and external-review decisions.
- Claim-audit generation that checks external-facing text for unsupported SOTA, novelty, production, benchmark, or robotics-policy claims.
- Static project-level portfolio site for GitHub Pages or local review.
- Static HTML portfolio page with evidence score, metrics, visual artifacts, and artifact links.

## Implemented

- Scene name: `desk_scene`
- Backend: `lerf`
- Demo queries generated: `5`
- CPU-only dry-run demo with mock RGB/relevancy/overlay outputs.
- Real-mode wrappers for Nerfstudio/LERF commands when upstream tools are installed.
- Run-scoped pipeline artifacts avoid stale query/evaluation results across reruns.
- Shareable preflight, audit, recommendation, and reproduction artifacts for portfolio review.
- Research report artifacts summarize evidence, limitations, and next steps in a paper-style format.
- Submission checklist records allowed claims, claims to avoid, pack validation status, and next actions.
- Real-run plan records the exact next commands needed to turn smoke evidence into a real captured-scene run.
- Claim audit records whether README/docs/run artifacts stay within supported research-engineering claims.
- Run evidence scorecard summarizes capture readiness, query artifacts, overlays, annotation coverage, evaluation metrics, and reproducibility files.

## Dry-Run vs Real GPU Mode

- Dry-run mode validates pipeline structure and produces synthetic visual artifacts.
- Real mode runs Nerfstudio processing/training and attempts LERF internal relevancy rendering.
- Viewer fallback artifacts are generated when upstream internals are incompatible.

## Reproduce

```bash
python -m pip install -e ".[dev,video]"
python scripts/run_dry_run_demo.py
python scripts/run_scene_pipeline.py --dry-run
python scripts/compare_runs.py --root results/pipeline_runs
python scripts/run_experiment_matrix.py --config examples/experiment_matrix.yaml --dry-run --limit 1
python scripts/create_annotation_workbench.py --annotations results/pipeline_runs/desk_scene/annotation_template.json --results results/pipeline_runs/desk_scene/queries --output results/pipeline_runs/desk_scene/evaluation/annotation_workbench
python scripts/finalize_annotations.py --run-dir results/pipeline_runs/desk_scene --filled results/pipeline_runs/desk_scene/evaluation/annotation_workbench/annotation_seed.json --profile smoke --export-pack --zip-pack
python scripts/generate_research_report.py --run-dir results/pipeline_runs/desk_scene
python scripts/diagnose_run_failures.py --run-dir results/pipeline_runs/desk_scene
python scripts/create_run_result_card.py --run-dir results/pipeline_runs/desk_scene
python scripts/generate_project_site.py --run-index results/pipeline_runs/run_index.json
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack.zip
python scripts/create_run_readiness.py --run-dir results/pipeline_runs/desk_scene --pack results/portfolio_pack
python scripts/create_real_run_plan.py --run-dir results/pipeline_runs/desk_scene --output results/real_run_plan --input path/to/video.mp4 --type video --submission-packet results/pipeline_runs/desk_scene/submission_packet/submission_packet.json
```

## Outputs

- Query grid: `results\demo_assets\query_grid.png`
- Demo montage: `results\demo_assets\demo_montage.gif`
- Project site: `docs/index.html`
- Pipeline summary: `results/pipeline_runs/desk_scene/pipeline_summary.json`
- Run comparison: `results/pipeline_runs/run_comparison.md`
- Capture manifest: `results/pipeline_runs/desk_scene/capture_manifest.md`
- Capture validation: `results/pipeline_runs/desk_scene/capture_manifest_validation.md`
- Preflight report: `results/pipeline_runs/desk_scene/preflight_report.md`
- Failure diagnostics: `results/pipeline_runs/desk_scene/failure_diagnostics.md`
- Evidence scorecard: `results/pipeline_runs/desk_scene/evidence_scorecard.md`
- Claim audit: `results/pipeline_runs/desk_scene/claim_audit.md`
- Run result card: `results/pipeline_runs/desk_scene/run_result_card.md`
- Research report: `results/pipeline_runs/desk_scene/research_report.md`
- Real-run action plan: `results/pipeline_runs/desk_scene/real_run_plan/real_run_plan.md`
- Run readiness gate: `results/pipeline_runs/desk_scene/run_readiness.md`
- Submission checklist: `results/pipeline_runs/desk_scene/submission_packet/submission_checklist.md`
- Static portfolio page: `results/pipeline_runs/desk_scene/portfolio_page.html`
- Annotation review: `results/pipeline_runs/desk_scene/evaluation/annotation_review.md`
- Annotation workbench: `results/pipeline_runs/desk_scene/evaluation/annotation_workbench/annotation_workbench.html`
- Annotation finalization: `results/pipeline_runs/desk_scene/annotation_finalize_report.md`
- Scene inspection: `results/pipeline_runs/desk_scene/scene_data_inspection.md`
- Scene relations: `results/pipeline_runs/desk_scene/scene_relations/scene_relations_report.md`
- Run-scoped evaluation: `results/pipeline_runs/desk_scene/evaluation/eval_summary.json`
- Experiment matrix: `results/experiment_matrix/dry_run_semantic_backend_matrix/experiment_matrix_report.md`
- Portfolio pack: `results/portfolio_pack.zip`
- Evaluation summary: `results/evaluation/eval_summary.json`
- Qualitative report: `results/evaluation/qualitative_report.md`

`portfolio_page.html` surfaces the same sharing-readiness summary from the submission
packet, so a reviewer can see the current send/no-send status and next action without
opening the raw JSON first.

## Limitations

- This is not a new NeRF architecture or state-of-the-art segmentation model.
- Dry-run outputs are synthetic and should not be interpreted as scene understanding results.
- Real LERF quality depends on capture quality, camera poses, GPU training, and upstream versions.

## CV Bullets

- Built a reproducible open-vocabulary 3D scene inspection system using Nerfstudio-style reconstruction and LERF-style language-field querying.
- Implemented deterministic query planning, semantic relevancy artifacts, spatial/evaluation utilities, and CPU-only dry-run demos.

## Cold-Email Paragraph

I built NeRF-LLM Scene Inspector as a research engineering project connecting NeRF reconstruction, language-embedded radiance fields, and natural-language scene querying. The project focuses on reproducible wrappers, typed artifacts, visualization, and lightweight evaluation rather than claiming algorithmic novelty. I am interested in extending this kind of system toward embodied AI and physical scene understanding.
