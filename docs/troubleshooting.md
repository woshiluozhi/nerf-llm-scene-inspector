# Troubleshooting

## `ns-process-data` Not Found

Install Nerfstudio in the active environment and run:

```bash
python -m pip install nerfstudio
ns-install-cli
ns-process-data --help
```

## LERF Methods Do Not Appear In `ns-train -h`

Install LERF as an editable package in the same environment as Nerfstudio:

```bash
git clone https://github.com/kerrj/lerf
cd lerf
python -m pip install -e .
ns-install-cli
ns-train -h
```

The help output should include `lerf`, `lerf-lite`, and `lerf-big` as separate method
tokens. The project checks exact method names, so seeing only `lerf-lite` is not enough for
commands that request `--variant lerf`.

## COLMAP Or FFmpeg Missing

Nerfstudio's data processing relies on FFmpeg for video extraction and COLMAP for camera pose estimation. Install them through conda when possible:

```bash
conda install -c conda-forge ffmpeg colmap
```

## CUDA Or Tiny CUDA NN Errors

Full training requires an NVIDIA GPU, a CUDA-compatible PyTorch build, and Tiny CUDA NN support. Start with `lerf-lite` on smaller GPUs.

## Low Pose Coverage Or Duplicate Camera Poses

`inspect_scene_data.py` reports camera translation extent, path length, median step, and
duplicate adjacent poses. If the pose coverage score is low, the capture is usually missing
parallax. Re-capture the scene by moving around the object or desk from multiple angles and
heights instead of standing in one place and rotating the phone.

## Query Rendering Falls Back To Viewer Instructions

Upstream LERF documents prompt entry through the Nerfstudio viewer. This project attempts an internal API render first. If the installed Nerfstudio/LERF versions expose incompatible internals, `query_scene.py` still writes a structured report and an interactive viewer workflow.

Use the viewer fallback as a recoverable path:

```bash
ns-viewer --load-config path/to/config.yml
```

Enter the text prompt, save images named like `view_0000_rgb.png`,
`view_0000_relevancy.png`, and optionally `view_0000_overlay.png`, then import them:

```bash
python scripts/import_viewer_outputs.py \
  --query "mug" \
  --config path/to/config.yml \
  --input results/manual_viewer/mug \
  --output results/query_outputs/mug
```

The importer writes `query_result.json`, estimates image-space boxes from relevancy maps,
and can be used by `create_annotation_template.py` and `evaluate_queries.py`.

For a full `scene_query_report.json` created by `query_scene.py`, save each viewer prompt
under a query-slugged directory such as `results/manual_viewer/mug/` or
`results/manual_viewer/cup/`, then repair the report in place:

```bash
python scripts/repair_scene_query_from_viewer.py \
  --report results/pipeline_runs/desk_scene/queries/mug/scene_query_report.json \
  --viewer-root results/manual_viewer
```

This replaces only queries with matching manual viewer outputs, keeps the original plan,
regenerates the natural-language answer, writes `viewer_repair_summary.json`, and refreshes
the Markdown report.
