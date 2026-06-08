# Real Scene Capture Checklist

## Phone Video Capture

- Capture a static, desk-scale scene for 30-90 seconds.
- Move slowly and keep high overlap between frames.
- Circle the scene from multiple heights instead of panning from one point.
- Avoid motion blur, glossy-only surfaces, and strong exposure changes.
- Keep lighting stable and bright.
- Include textured background regions so COLMAP has trackable features.

## Data Processing

```bash
python scripts/create_capture_manifest.py --input path/to/video.mp4 --type video --scene-name desk_scene --capture-device "phone model" --lighting "bright diffuse indoor" --camera-motion "slow orbit" --static-scene --high-overlap --privacy-reviewed --output results/capture_manifest
python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --require-gpu --allow-warnings
python scripts/prepare_data.py --input path/to/video.mp4 --output data/processed/desk_scene --type video
python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --capture-manifest results/capture_manifest/capture_manifest.json --data data/processed/desk_scene --require-gpu
python scripts/inspect_scene_data.py --data data/processed/desk_scene --min-frames 50 --min-pose-extent 0.05
```

The capture manifest records device, lighting, camera motion, overlap, static-scene status,
and privacy review. The first preflight command checks the raw capture and upstream
environment before data processing. The second command checks the processed Nerfstudio scene
and capture manifest before training.
The inspection report checks more than frame count: it reports missing images, invalid
camera transforms, camera translation extent, approximate camera path length, median camera
step, duplicate adjacent poses, and a pose coverage score. If pose coverage is low, the
camera likely rotated in place or COLMAP recovered near-duplicate poses.

If COLMAP fails, try:

- Extract fewer frames by trimming or downsampling the video before processing.
- Capture with slower motion and more overlap.
- Add background texture or use a scene with more visual features.
- Verify `ffmpeg`, `colmap`, and `ns-process-data` are on PATH.

## Training

Full training requires an NVIDIA GPU and a CUDA-compatible PyTorch/Nerfstudio environment.

```bash
python scripts/train_baseline_nerf.py --data data/processed/desk_scene --method nerfacto --output runs/baseline_desk_scene
python scripts/train_language_field.py --data data/processed/desk_scene --backend lerf --variant lerf-lite --output runs/language_desk_scene
```

Start with `lerf-lite` for smaller GPUs. Increase training iterations only after data processing and baseline rendering look reasonable.

## One-Command Pipeline

After `python scripts/check_env.py --check-upstream --require-gpu` passes on a GPU machine, run:

```bash
python scripts/run_scene_pipeline.py \
  --input path/to/video.mp4 \
  --scene-name desk_scene \
  --type video \
  --backend lerf \
  --variant lerf-lite \
  --query "mug" \
  --query "objects that can hold water" \
  --annotations examples/annotations_example.json \
  --num-views 3 \
  --min-pose-extent 0.05 \
  --strict
```

Review `results/pipeline_runs/desk_scene/preflight_report.md`,
`results/pipeline_runs/desk_scene/pipeline_summary.json`, and
`results/pipeline_runs/desk_scene/scene_data_inspection.md` before using the results in a portfolio report.
Then review `results/pipeline_runs/desk_scene/evidence_scorecard.md` to see whether the run is only
a smoke demo, still needs review, or has enough real-run evidence for portfolio sharing.
After repeated captures or training attempts, run `python scripts/compare_runs.py --root results/pipeline_runs`
and inspect `results/pipeline_runs/run_comparison.md` before selecting a portfolio candidate.
Open `results/pipeline_runs/desk_scene/portfolio_page.html` for a compact static review page
with score, metrics, visual evidence, and links to the underlying artifacts.
The pipeline cleans the current run's `queries/`, `demo_assets/`, and `evaluation/` folders by default so
reruns do not accidentally evaluate stale artifacts. Use `--no-clean-run` only when preserving prior files is intentional.
For a stricter reproducibility checklist, see `docs/real_run_reproducibility.md`.
Use `results/pipeline_runs/desk_scene/annotation_template.json` as the starting point for manual
2D localization annotations before reporting quantitative query metrics.
Open `results/pipeline_runs/desk_scene/evaluation/annotation_workbench/annotation_workbench.html`
to draw or adjust `bbox_2d` labels in a browser and export filled annotation JSON.
Finalize the run from that export:

```bash
python scripts/finalize_annotations.py --run-dir results/pipeline_runs/desk_scene --filled path/to/annotations_filled.json --profile real-run --export-pack --zip-pack
```

Then open `results/pipeline_runs/desk_scene/evaluation/annotation_review.md` and the contact sheet
to catch wrong view ids, shifted boxes, or qualitative-only labels before sharing scores.

The finalizer also exports and validates the shareable pack. To rerun only the sharing checks:

```bash
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack
python scripts/audit_claims.py --run-dir results/pipeline_runs/desk_scene --pack results/portfolio_pack
```

## LERF Prompt Examples

- `mug`
- `laptop`
- `objects that can hold water`
- `safe place to put a hot cup`
- `metallic tools`
- `coffee-making objects`

## Viewer Screenshot Workflow

```bash
ns-viewer --load-config path/to/config.yml
```

In the viewer:

1. Enter the text prompt.
2. Select `rgb`, `relevancy_0`, and `composited_0` outputs when available.
3. Save screenshots into the query output directory.
4. Use names such as `view_0000_rgb.png`, `view_0000_relevancy.png`, and `view_0000_overlay.png`.
5. Fill the generated manual `QueryResult` template if automated rendering falls back.
