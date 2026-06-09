# Real-Run Reproducibility Notes

This project is designed so a dry-run portfolio demo and a real GPU run produce the same
artifact shape. For a real scene, keep the full run directory and the exported portfolio pack.

## Before Running

```bash
git status --short
python scripts/check_env.py --check-upstream --require-gpu --verbose
python scripts/create_capture_manifest.py --input path/to/video.mp4 --type video --scene-name desk_scene --capture-device "phone model" --lighting "bright diffuse indoor" --camera-motion "slow orbit" --static-scene --high-overlap --privacy-reviewed --output results/capture_manifest
python scripts/preflight_real_run.py --input path/to/video.mp4 --type video --require-gpu --allow-warnings
python scripts/create_real_run_plan.py --run-dir results/pipeline_runs/desk_scene --input path/to/video.mp4 --type video
python scripts/diagnose_run_failures.py --run-dir results/pipeline_runs/desk_scene
python scripts/create_run_readiness.py --run-dir results/pipeline_runs/desk_scene
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
  --prompt-suite examples/prompt_sensitivity.yaml \
  --analyze-relations \
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
- `logs/*.json`: full command, return code, stdout, stderr, and dry-run flag for subprocess-backed steps, including exact `ns-train -h` method checks before real baseline and language training.
- `failure_diagnostics.md`: classified CUDA, Nerfstudio/LERF, COLMAP/FFmpeg, missing-config, and viewer-fallback repair actions inferred from saved logs and run artifacts.
- `scene_data_inspection.md`: frame count, missing images, pose validity, pose coverage, and capture recommendations.
- `training/baseline_train_summary.json`: baseline Nerfstudio command, status, final config, and viewer command.
- `training/language_train_summary.json`: LERF/OpenNeRF command, status, final config, and viewer command.
- `queries/<query>/scene_query_report.json`: query plan, backend outputs, synthesized answer summary, counter-evidence, risk flags, warnings, and provenance.
- `queries/<query>/scene_query_report.md`: human-readable natural-language answer with ranked evidence, counter-evidence, support level, limitations, and follow-up checks.
- `queries/<query>/query_grid.png` and `query_visual_summary.json`: compact visual overview and machine-readable summary for expanded backend prompts in one task; the query-evidence audit checks the summary's expanded queries, overlay count, query grid, and optional montage paths against the actual report and files.
- `query_evidence_audit.md`: run-level audit of each query task's overlays, localization evidence, fallback mode, confidence, counter-evidence/risk-flag counts, and missing artifacts.
- `prompt_sensitivity/prompt_sensitivity_report.md`: prompt wording stability diagnostic across grouped query variants.
- `prompt_sensitivity/prompt_sensitivity_summary.json`: machine-readable confidence, view-agreement, and top-region consistency metrics.
- `scene_relations/scene_relations_report.md`: deterministic relation graph over query-derived entities, with `3d` or `2d_fallback` evidence tags.
- `scene_relations/scene_relations_edges.csv`: relation edge table for support/proximity/containment/layout review.
- `annotation_template.json`: fill-in manual annotation scaffold generated from query outputs.
- `evaluation/annotation_workbench/annotation_workbench.html`: offline browser workbench for drawing and exporting manual `bbox_2d` labels.
- `evaluation/annotation_workbench/annotation_seed.json`: seed annotation JSON generated for the workbench; use browser-downloaded filled JSON for final labels.
- `annotations_merged.json`: clean annotation schema produced by merging filled workbench JSON back into the template.
- `evaluation/annotation_validation.json`: annotation coverage, duplicate-label, bbox, and view-id checks.
- `evaluation/annotation_review.md`: visual QA table for manual bbox annotations.
- `evaluation/annotation_review_contact_sheet.png`: contact sheet with bboxes drawn over rendered views.
- `run_audit.md`: run-level health summary for environment, data, query, annotation, and evaluation readiness.
- `run_recommendations.md`: prioritized next actions for turning a smoke run into stronger evidence.
- `evidence_scorecard.md`: conservative multi-criterion scorecard for whether the run is strong enough to share, including capture/privacy metadata readiness.
- `quality_gate.md`: pass/warn/fail gate for smoke, real-run, or final portfolio sharing profiles.
- `run_readiness.md`: consolidated gate for `ready_to_start_real_run` and `ready_for_external_review` decisions.
- `claim_audit.md`: scan of README/docs/run-facing text for unsupported SOTA, novelty, production, benchmark, or robotics-policy claims.
- `run_result_card.md`: concise reviewer-facing takeaway, evidence snapshot, limitations, and safe sharing language for the run.
- `portfolio_page.html`: static, relative-link HTML page for reviewing or sharing run evidence.
- `research_report.md`: paper-style summary of run evidence, limitations, reproducibility artifacts, and next steps.
- `research_report.json`: machine-readable version of the same research report.
- `submission_packet/submission_checklist.md`: claim-calibrated checklist for CV, portfolio, and professor outreach.
- `submission_packet/submission_packet.json`: machine-readable sharing decision and checklist status.
- `reproduction_manifest.json`: machine-readable replay command, verification commands, artifact summary, key artifact map, file sizes, SHA256 digests, and query-level evidence paths.
- `reproduction_manifest_validation.json` and `.md`: integrity check for files and directories recorded in the reproduction manifest.
- `reproduction_report.md`: human-readable reproduction recipe for sharing with collaborators.
- `reproduce_run.sh`: shell recipe that installs local dependencies, runs checks, replays the pipeline, and verifies the pack.
- `real_run_plan/real_run_plan.md`: concrete action plan for upgrading a dry-run or partially reviewed run into real CUDA/Nerfstudio/LERF evidence.
- `real_run_plan/real_run_plan.json`: machine-readable version of the same plan, including blocker/warning counts and command phases.
- `demo_assets/query_grid.png`: compact qualitative query visualization.
- `evaluation/eval_summary.json`: lightweight quantitative summary when annotations are available.
- `portfolio_result_card.md`: short result narrative suitable for a project page.

For external sharing, start with `submission_packet/submission_checklist.md`. Its
`Readiness Summary` section is the compact send/no-send view: failed checks, warning checks,
pack validation state, capture-manifest status/failure count, query-evidence counter/risk
counts, and the recommended next action.

The real-run action plan reads the same audit evidence before recommending the next GPU
run. Capture-validation `fail_count`, preflight `fail_count`, quality-gate `fail_count`,
run-audit `blocker_count`, failure-diagnostics `blocker_count`, query-evidence risk flags,
and a blocked run-readiness gate are all surfaced as blocker-level plan issues even when a
status string still looks ready.
Unresolved query risk flags are treated as a blocker in this packet; non-overlapping
counter-evidence is retained as a warning for calibrated review. The same information is available in
`submission_packet/submission_packet.json` under `readiness_summary` and the top-level
`capture_manifest_*` and `query_*` fields for automation. Pack
references in this packet use share-safe artifact names such as `portfolio_pack.zip` instead
of machine-specific absolute paths.
`run_result_card.md` consumes the same submission and query-evidence fields and directly
checks local capture validation, so it will mark a run as blocked when query evidence
fails, unresolved risk flags remain, or real-run capture validation is missing/failed,
even if the evidence scorecard is otherwise strong.
Blocked run-audit findings and blocker-level failure diagnostics are also treated as
external-sharing blockers in both the submission packet and the result card; refresh those
artifacts after any manual edits or copied run folders.
The submission packet directly checks capture validation as well: a real run with missing
capture validation, `status=blocked`, or nonzero `capture_manifest_validation.fail_count`
is blocked before outreach.
The run quality gate applies the same count-based check: nonzero
`run_audit.blocker_count`, `failure_diagnostics.blocker_count`, or
`capture_manifest_validation.fail_count` fails the gate even if a stale status field still
looks ready.
Portfolio-pack validation also reads `real_run_plan/real_run_plan.json`: nonzero
`real_run_plan.blocker_count` fails the shareable pack, while nonzero
`real_run_plan.warning_count` remains a warning to review the next-run playbook before
outreach.
`portfolio_page.html` mirrors those fields in its top-level metrics and Sharing Readiness
panel, including capture validation status/failures, query-evidence status,
counter-evidence count, and risk-flag count. It also reads local capture validation before
rendering, so a stale result card cannot hide a real-run capture blocker.
Use `run_readiness.md` before launching or sharing a run: it combines pipeline success,
dry-run/real-run mode, capture metadata, preflight status, GPU/upstream environment checks,
language training, query evidence risk flags, run audit, failure diagnostics, quality gates,
claim audit, submission packet, result card, and portfolio pack validation into two explicit
booleans: `ready_to_start_real_run` and `ready_for_external_review`. Readiness checks both
status fields and blocker/failure counts, so stale artifacts with nonzero blockers cannot
be promoted by a ready-looking status string.

You can inspect these files in one place with the Streamlit dashboard:

```bash
python -m pip install -e ".[dashboard]"
streamlit run src/nerf_llm_scene_inspector/visualization/dashboard.py
```

Set the dashboard's pipeline run directory to `results/pipeline_runs/desk_scene`.
The run review tab includes the multi-run comparison summary so repeated captures or
training attempts can be ranked before choosing a portfolio candidate. It also shows the
run quality gate, submission readiness summary, run-audit blocker count,
failure-diagnostics blocker count, capture-validation status, capture failure count,
real-run-plan status, and plan blocker/warning counts so failed, warning-level, and
send/no-send criteria are visible without opening raw JSON. The evidence audit tab exposes query
counter-evidence and risk-flag counts alongside overlays, regions, and 3D points.

If automated LERF query rendering falls back to the interactive viewer, save the viewer
outputs and convert them back into the standard query schema before annotation/evaluation:

```bash
python scripts/import_viewer_outputs.py \
  --query "mug" \
  --config runs/language_desk_scene/config.yml \
  --input results/manual_viewer/mug \
  --output results/pipeline_runs/desk_scene/queries/mug
```

For a multi-query scene report, save one folder per query slug under `results/manual_viewer/`
and repair the whole report before annotation:

```bash
python scripts/repair_scene_query_from_viewer.py \
  --report results/pipeline_runs/desk_scene/queries/mug/scene_query_report.json \
  --viewer-root results/manual_viewer \
  --require-all
```

Use `--require-all` when every expanded prompt should have manually saved viewer evidence.
Without it, missing query folders are kept as original results and recorded as warnings.
Portfolio-pack export keeps `viewer_repair_summary.json` and `viewer_import_summary.json`;
pack validation treats incomplete required repairs as share-blocking evidence issues.
The query-evidence audit separates recovery artifacts from visual evidence: viewer fallback
Markdown files and manual templates are counted as fallback artifacts, not rendered images.
A fallback-only query report fails query evidence until RGB/relevancy/overlay outputs are
imported from the viewer.

## Provenance Fields

Each pipeline run stores a `provenance` block with:

- project package version
- Python version and platform
- CLI command used for the run
- git commit, branch, dirty state, and sanitized origin remote when available
- non-fatal warnings if git metadata cannot be read

The exported portfolio pack keeps the original run files but sanitizes machine-specific paths
inside the packaged copy. The top-level `portfolio_pack_index.json` exposes a compact,
share-safe provenance excerpt and records SHA256/size digests for copied files so
`validate_portfolio_pack.py` can detect accidental edits or tampering before sharing.
The run-local `reproduction_manifest.json` now provides a similar pre-pack audit trail:
before exporting, inspect `artifact_summary` and the per-artifact `kind`, `size_bytes`,
`sha256`, and `file_count` fields to confirm that query reports, visual summaries, grids,
evaluation files, and command logs are present.
Then verify the manifest against the current run directory:

```bash
python scripts/verify_reproduction_manifest.py --run-dir results/pipeline_runs/desk_scene
python scripts/verify_reproduction_manifest.py --run-dir results/pipeline_runs/desk_scene --require-complete
```

The first command checks that artifacts recorded as present still match their saved sizes
and SHA256 digests. The stricter `--require-complete` form is intended for real portfolio
candidates where any recorded-missing expected artifact should block sharing.
When `--zip` is used, the archive is self-contained and includes the same
`portfolio_pack_index.json` with digest metadata.
The normal finalization path runs validation and re-archives the pack afterward, so the
shareable zip also includes `portfolio_pack_validation.json`. It then validates the final
zip itself and writes `results/portfolio_pack_validation.json` next to the archive.
`validate_portfolio_pack.py` accepts both zip layouts: files at the archive root, or files
inside one top-level `portfolio_pack/` directory. Pack validation fails on unresolved
query risk flags from `query_evidence_audit.json`, because those conflicts should not be
sent as clean scene-understanding evidence. Counter-evidence without risk flags remains a
warning for reviewer calibration.
It also fails on nonzero run-audit blocker counts, failure-diagnostic blocker counts, and
capture-manifest failure counts even if the corresponding status field was not refreshed.

## Finalize And Export

```bash
python scripts/create_annotation_workbench.py --annotations results/pipeline_runs/desk_scene/annotation_template.json --results results/pipeline_runs/desk_scene/queries --output results/pipeline_runs/desk_scene/evaluation/annotation_workbench
python scripts/finalize_annotations.py --run-dir results/pipeline_runs/desk_scene --filled path/to/annotations_filled.json --profile portfolio --export-pack --zip-pack
python scripts/verify_reproduction_manifest.py --run-dir results/pipeline_runs/desk_scene --require-complete
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack.zip
python scripts/check_run_quality.py --run-dir results/pipeline_runs/desk_scene --profile portfolio --pack results/portfolio_pack
python scripts/create_run_readiness.py --run-dir results/pipeline_runs/desk_scene --pack results/portfolio_pack
```

Use `export_portfolio_pack.py` directly only when debugging packaging internals. For normal
run-scoped sharing, the finalizer is preferred because it refreshes evaluation, visual QA,
reports, scorecards, submission materials, pack validation, and the final zip.
Run `audit_claims.py` with `--pack results/portfolio_pack` before outreach when you have an
exported pack; the claim audit now reads `portfolio_pack_validation.json` and fails if the
pack is not clean.
`create_run_readiness.py` treats claim-audit failures as blockers and claim-audit warnings
as review items, so dry-run caveats and pack warnings do not get confused with unsupported
external-facing claims.

Refresh the multi-run index after manual edits or copied-in run folders:

```bash
python scripts/index_runs.py --root results/pipeline_runs
python scripts/compare_runs.py --root results/pipeline_runs
python scripts/generate_project_site.py --run-index results/pipeline_runs/run_index.json
```

The index, project site, comparison reports, and experiment matrix surface result status,
submission readiness, query-evidence status, counter-evidence counts, risk-flag counts,
run-audit blocker counts, failure-diagnostics blocker counts, capture-manifest failure
counts, and real-run-plan blocker/warning counts. A real run with unresolved query risk flags is never selected as a
`portfolio_candidate`; comparison reports rank it as `needs_review`, and the experiment
matrix treats risk flags as an external-sharing blocker. Nonzero run-audit blockers,
blocked failure diagnostics, failure-diagnostics blockers, capture-manifest failures, or
real-run-plan blockers also block candidate selection. Real-run capture validation that is
missing/not `ready` also blocks candidate selection,
even if a stale status field still says ready. A real run is only ranked as a portfolio
candidate when the result card, submission packet, query-evidence audit, run audit,
failure diagnostics, real-run plan, and capture manifest are clean. The run-index `ready_runs` count is
also strict: it only counts successful non-dry-run runs with `portfolio_ready`
result/submission status, passing query evidence, clear failure diagnostics, clean run
audit, clean real-run plan, and ready capture validation.

For a small ablation-style table across variants or query sets, run:

```bash
python scripts/run_experiment_matrix.py \
  --config examples/experiment_matrix.yaml \
  --output results/experiment_matrix/real_scene_matrix \
  --real-run
```

Use `--collect-only` after manual reruns or copied-in pipeline directories to refresh the
matrix summary without launching training again. Inspect `candidate_status`,
`failure_diagnostics_status`, `failure_blocker_count`, `audit_blocker_count`,
`capture_manifest_fail_count`, `readiness_level`, `query_evidence_status`,
`query_risk_flag_count`, and
`blocking_reasons` in the CSV before choosing which real run to package or share
externally.

Share `results/portfolio_pack.zip` together with the GitHub repository link when sending a
portfolio or cold-email artifact. Do not claim benchmark superiority from a dry-run or a
single qualitative scene; report it as a reproducible research-engineering demo unless you
run a larger annotated evaluation.

## Manual Annotation Loop

After a real query run, open `evaluation/annotation_workbench/annotation_workbench.html` and inspect the overlay images. For
each query, fill or adjust:

- `target_description`: what the correct object or region is
- `acceptable_views`: view ids where the target is visible, such as `view_0000`
- `bbox_2d`: `[x1, y1, x2, y2]` in the selected rendered view
- `notes`: uncertainty, ambiguity, or qualitative-only rationale

Download the filled JSON from the workbench, then use the run-scoped finalizer. This is the
recommended path because it merges annotations, reruns evaluation and visual QA, refreshes
audits/recommendations/scorecards/quality gates, regenerates reports/result cards/portfolio
pages/reproduction bundles, and can export a fresh portfolio pack.

```bash
python scripts/finalize_annotations.py \
  --run-dir results/pipeline_runs/desk_scene \
  --filled path/to/annotations_filled.json \
  --profile real-run \
  --export-pack \
  --zip-pack
```

The lower-level tools are still available for debugging individual steps:
`merge_annotation_workbench.py`, `validate_annotations.py`, `review_annotations.py`,
`evaluate_queries.py`, `audit_run.py`, `recommend_next_steps.py`,
`audit_query_evidence.py`, `create_evidence_scorecard.py`, `check_run_quality.py`, `generate_research_report.py`,
`create_run_result_card.py`, `generate_portfolio_page.py`, `create_reproduction_bundle.py`,
`diagnose_run_failures.py`, `audit_claims.py`, and `create_submission_packet.py`.

Only rows with a filled `bbox_2d` are included in localization metrics such as
`top_k_hit_rate` and `mean_iou_2d`. Rows without bbox annotations stay in the qualitative
table as `unannotated` or `qualitative_only_no_bbox`, so missing labels do not get counted as
failed detections. If a visual prompt appears more than once because several tasks expanded
to the same backend query, the CSV preserves all rows while summary metrics use the best row
per unique query.
