# NeRF-LLM Scene Inspector Report

## Overview

This report summarizes a NeRF-LLM Scene Inspector run. The system is built on
Nerfstudio and LERF and is intended as a reproducible research engineering demo.
For real scenes, the pipeline records environment diagnostics and processed-scene
inspection artifacts under `results/pipeline_runs/<scene>/`.

## Scene

- Scene name: `desk_scene`
- Backend: `lerf/opennerf`

## Query Results

| Query | Target | Top-k Hit | Best IoU | Confidence | Warnings |
| --- | --- | --- | --- | --- | --- |
| coffee |  | False | 0.000 | 0.8862745098039215 |  |
| container |  | False | 0.000 | 0.8862745098039215 |  |
| laptop | open laptop on the desk | False | 0.194 | 0.8862745098039215 |  |
| metallic tools |  | False | 0.000 | 0.8862745098039215 |  |
| mug | white mug on the desk | False | 0.179 | 0.8862745098039215 |  |
| objects that can hold water |  | False | 0.000 | 0.8862745098039215 |  |
| safe place to put a hot cup |  | False | 0.000 | 0.8862745098039215 |  |

## Evaluation Summary

| Metric | Value |
| --- | --- |
| top_k_hit_rate | 0.0 |
| mean_iou_2d | 0.053268581137313094 |
| semantic_success_rate | 0.0 |
| average_relevancy_score | 0.8862745098039215 |
| num_evaluated_queries | 7 |

## Scene Relation Analysis

When `--analyze-relations` is enabled, the pipeline writes a deterministic relation graph
under `results/pipeline_runs/<scene>/scene_relations/`. It summarizes query-derived
entities, relation edges such as `near`, `left_of`, `likely_supports`, and
`likely_contained_in`, and whether each edge came from `3d` points or `2d_fallback`
rendered boxes. These relations are qualitative evidence, not learned physical-relation
predictions.

## Experiment Matrix

For small ablations, run `scripts/run_experiment_matrix.py` with
`examples/experiment_matrix.yaml`. The matrix report summarizes each configured pipeline run
with evidence score, prompt stability, relation-edge count, localization metrics, and links
to run-scoped artifacts. This is intended for reproducible comparison across variants, not as
a benchmark claim.

## Annotation Workbench

Run `scripts/create_annotation_workbench.py --annotations results/pipeline_runs/<scene>/annotation_template.json
--results results/pipeline_runs/<scene>/queries --output results/pipeline_runs/<scene>/evaluation/annotation_workbench`
to generate an offline HTML bbox-labeling workspace. The workbench copies query render images,
preloads candidate boxes, and exports filled annotation JSON for validation, visual review,
and evaluation. Merge the downloaded JSON back into the standard annotation schema with
`scripts/merge_annotation_workbench.py --template results/pipeline_runs/<scene>/annotation_template.json
--filled path/to/annotations_filled.json --output results/pipeline_runs/<scene>/annotations_merged.json`
before running validation or reporting metrics.
For a run-scoped refresh, use `scripts/finalize_annotations.py --run-dir
results/pipeline_runs/<scene> --filled path/to/annotations_filled.json --profile real-run`
to merge labels and regenerate evaluation, QA, scorecards, reports, result cards, and portfolio pages.

## Research Report

Run `scripts/generate_research_report.py --run-dir results/pipeline_runs/<scene>` after a
pipeline run to generate `research_report.md` and `research_report.json`. The report combines
the evidence scorecard, evaluation metrics, prompt-sensitivity diagnostics, scene-relation
analysis, reproducibility artifacts, limitations, and next steps into a paper-style project
summary suitable for portfolio review.

## Run Result Card

Run `scripts/create_run_result_card.py --run-dir results/pipeline_runs/<scene>` to generate
`run_result_card.md` and `run_result_card.json`. This one-page card gives a reviewer-facing
takeaway, shareable blurb, evidence snapshot, metrics, limitations, checks, and next actions
without claiming more than the run artifacts support.

## Submission Packet

Run `scripts/create_submission_packet.py --run-dir results/pipeline_runs/<scene> --pack
results/portfolio_pack` after validating a portfolio pack. The output records share readiness,
allowed claims, claims to avoid, recommended links, and next actions for CV or professor-outreach
use. Dry-run packets explicitly mark the run as smoke-demo evidence only.

## Real-Run Action Plan

Run `scripts/create_real_run_plan.py --run-dir results/pipeline_runs/<scene> --input
path/to/video.mp4 --type video` to generate a concrete command playbook for moving from
dry-run smoke evidence to a real CUDA/Nerfstudio/LERF scene run. The plan is an execution
checklist, not a claim that the real run has already succeeded.

## Claim Audit

Run `scripts/audit_claims.py --run-dir results/pipeline_runs/<scene> --pack
results/portfolio_pack` before sharing. The audit checks portfolio-facing text for
unsupported SOTA, novelty, benchmark-superiority, production-readiness, or robotics-policy
claims and verifies required disclaimers.

## Notes

- Metrics are lightweight portfolio metrics and depend on manual annotations.
- Dry-run results are synthetic and only validate the evaluation pipeline.
- Review `scene_data_inspection.md` before interpreting real trained outputs.
