# Real-Run Reproducibility Notes

This project is designed so a dry-run portfolio demo and a real GPU run produce the same
artifact shape. For a real scene, keep the full run directory and the exported portfolio pack.

## Before Running

```bash
git status --short
python scripts/check_env.py --check-upstream --require-gpu --verbose
python scripts/create_capture_manifest.py --input path/to/video.mp4 --type video --scene-name desk_scene --capture-device "phone model" --lighting "bright diffuse indoor" --camera-motion "slow orbit" --static-scene --high-overlap --privacy-reviewed --output results/capture_manifest
python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --require-gpu --allow-warnings
```

Record the upstream versions that matter for interpretation:

```bash
python -c "import sys; print(sys.version)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
ns-train -h
colmap -h
ffmpeg -version
```

## Recommended Real Run

```bash
python scripts/run_scene_pipeline.py \
  --input path/to/video.mp4 \
  --scene-name desk_scene \
  --type video \
  --capture-manifest results/capture_manifest/capture_manifest.json \
  --backend lerf \
  --variant lerf-lite \
  --query "mug" \
  --query "objects that can hold water" \
  --query "safe place to put a hot cup" \
  --annotations examples/annotations_example.json \
  --num-views 3 \
  --min-frames 50 \
  --min-pose-extent 0.05 \
  --strict
```

## What To Inspect

- `pipeline_summary.json`: step status, commands, warnings, and reproducibility provenance.
- `capture_manifest.md`: capture device, scene type, lighting, camera motion, overlap, static-scene, and privacy metadata.
- `capture_manifest_validation.md`: checks for missing capture metadata and privacy-review readiness; non-ready status is reflected in the run audit, recommendations, and evidence scorecard.
- `preflight_report.md`: raw input, processed scene, config path, CUDA/upstream, and backend-method readiness checks.
- `../run_index.md`: compact comparison table across pipeline runs in the same root.
- `../run_comparison.md`: ranked comparison across repeated captures/training attempts, with dry-runs separated from real portfolio candidates.
- `environment_report.json`: Python, platform, CUDA, Nerfstudio, LERF, COLMAP, and FFmpeg checks.
- `logs/*.json`: full command, return code, stdout, stderr, and dry-run flag for subprocess-backed steps.
- `scene_data_inspection.md`: frame count, missing images, pose validity, pose coverage, and capture recommendations.
- `training/baseline_train_summary.json`: baseline Nerfstudio command, status, final config, and viewer command.
- `training/language_train_summary.json`: LERF/OpenNeRF command, status, final config, and viewer command.
- `queries/<query>/scene_query_report.json`: query plan, backend outputs, warnings, and provenance.
- `annotation_template.json`: fill-in manual annotation scaffold generated from query outputs.
- `evaluation/annotation_validation.json`: annotation coverage, duplicate-label, bbox, and view-id checks.
- `evaluation/annotation_review.md`: visual QA table for manual bbox annotations.
- `evaluation/annotation_review_contact_sheet.png`: contact sheet with bboxes drawn over rendered views.
- `run_audit.md`: run-level health summary for environment, data, query, annotation, and evaluation readiness.
- `run_recommendations.md`: prioritized next actions for turning a smoke run into stronger evidence.
- `evidence_scorecard.md`: conservative multi-criterion scorecard for whether the run is strong enough to share, including capture/privacy metadata readiness.
- `portfolio_page.html`: static, relative-link HTML page for reviewing or sharing run evidence.
- `reproduction_manifest.json`: machine-readable replay command, verification commands, and key artifact map.
- `reproduction_report.md`: human-readable reproduction recipe for sharing with collaborators.
- `reproduce_run.sh`: shell recipe that installs local dependencies, runs checks, replays the pipeline, and verifies the pack.
- `demo_assets/query_grid.png`: compact qualitative query visualization.
- `evaluation/eval_summary.json`: lightweight quantitative summary when annotations are available.
- `portfolio_result_card.md`: short result narrative suitable for a project page.

You can inspect these files in one place with the Streamlit dashboard:

```bash
python -m pip install -e ".[dashboard]"
streamlit run src/nerf_llm_scene_inspector/visualization/dashboard.py
```

Set the dashboard's pipeline run directory to `results/pipeline_runs/desk_scene`.

If automated LERF query rendering falls back to the interactive viewer, save the viewer
outputs and convert them back into the standard query schema before annotation/evaluation:

```bash
python scripts/import_viewer_outputs.py \
  --query "mug" \
  --config runs/language_desk_scene/config.yml \
  --input results/manual_viewer/mug \
  --output results/pipeline_runs/desk_scene/queries/mug
```

## Provenance Fields

Each pipeline run stores a `provenance` block with:

- project package version
- Python version and platform
- CLI command used for the run
- git commit, branch, dirty state, and sanitized origin remote when available
- non-fatal warnings if git metadata cannot be read

The exported portfolio pack keeps the original run files but sanitizes machine-specific paths
inside the packaged copy. The top-level `portfolio_pack_index.json` exposes only a compact,
share-safe provenance excerpt.

## Export

```bash
python scripts/export_portfolio_pack.py --run-dir results/pipeline_runs/desk_scene --zip
```

Refresh the multi-run index after manual edits or copied-in run folders:

```bash
python scripts/index_runs.py --root results/pipeline_runs
python scripts/compare_runs.py --root results/pipeline_runs
python scripts/generate_project_site.py --run-index results/pipeline_runs/run_index.json
```

Share `results/portfolio_pack.zip` together with the GitHub repository link when sending a
portfolio or cold-email artifact. Do not claim benchmark superiority from a dry-run or a
single qualitative scene; report it as a reproducible research-engineering demo unless you
run a larger annotated evaluation.

## Manual Annotation Loop

After a real query run, open `annotation_template.json` and inspect the overlay images. For
each query, fill:

- `target_description`: what the correct object or region is
- `acceptable_views`: view ids where the target is visible, such as `view_0000`
- `bbox_2d`: `[x1, y1, x2, y2]` in the selected rendered view
- `notes`: uncertainty, ambiguity, or qualitative-only rationale

Then rerun:

```bash
python scripts/validate_annotations.py \
  --queries results/pipeline_runs/desk_scene/queries.yaml \
  --annotations results/pipeline_runs/desk_scene/annotation_template.json \
  --results results/pipeline_runs/desk_scene/queries \
  --output results/pipeline_runs/desk_scene/evaluation/annotation_validation.json

python scripts/review_annotations.py \
  --annotations results/pipeline_runs/desk_scene/annotation_template.json \
  --results results/pipeline_runs/desk_scene/queries \
  --output results/pipeline_runs/desk_scene/evaluation \
  --allow-warnings

python scripts/evaluate_queries.py \
  --queries results/pipeline_runs/desk_scene/queries.yaml \
  --annotations results/pipeline_runs/desk_scene/annotation_template.json \
  --results results/pipeline_runs/desk_scene/queries \
  --output results/pipeline_runs/desk_scene/evaluation \
  --report-output results/pipeline_runs/desk_scene/project_report.md

python scripts/audit_run.py \
  --run-dir results/pipeline_runs/desk_scene

python scripts/recommend_next_steps.py \
  --run-dir results/pipeline_runs/desk_scene

python scripts/create_evidence_scorecard.py \
  --run-dir results/pipeline_runs/desk_scene

python scripts/generate_portfolio_page.py \
  --run-dir results/pipeline_runs/desk_scene

python scripts/create_reproduction_bundle.py \
  --run-dir results/pipeline_runs/desk_scene

python scripts/preflight_real_run.py \
  --input path/to/video.mp4 \
  --type video \
  --data data/processed/desk_scene \
  --require-gpu \
  --output results/pipeline_runs/desk_scene
```

Only rows with a filled `bbox_2d` are included in localization metrics such as
`top_k_hit_rate` and `mean_iou_2d`. Rows without bbox annotations stay in the qualitative
table as `unannotated` or `qualitative_only_no_bbox`, so missing labels do not get counted as
failed detections. If a visual prompt appears more than once because several tasks expanded
to the same backend query, the CSV preserves all rows while summary metrics use the best row
per unique query.
