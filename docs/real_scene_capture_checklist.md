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
python scripts/prepare_data.py --input path/to/video.mp4 --output data/processed/desk_scene --type video
python scripts/inspect_scene_data.py --data data/processed/desk_scene --min-frames 50
```

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

After `scripts/check_env.py --check-upstream --require-gpu` passes on a GPU machine, run:

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
  --strict
```

Review `results/pipeline_runs/desk_scene/pipeline_summary.json` and
`results/pipeline_runs/desk_scene/scene_data_inspection.md` before using the results in a portfolio report.
The pipeline cleans the current run's `queries/`, `demo_assets/`, and `evaluation/` folders by default so
reruns do not accidentally evaluate stale artifacts. Use `--no-clean-run` only when preserving prior files is intentional.

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
