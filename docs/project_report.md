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

## Research Report

Run `scripts/generate_research_report.py --run-dir results/pipeline_runs/<scene>` after a
pipeline run to generate `research_report.md` and `research_report.json`. The report combines
the evidence scorecard, evaluation metrics, prompt-sensitivity diagnostics, scene-relation
analysis, reproducibility artifacts, limitations, and next steps into a paper-style project
summary suitable for portfolio review.

## Submission Packet

Run `scripts/create_submission_packet.py --run-dir results/pipeline_runs/<scene> --pack
results/portfolio_pack` after validating a portfolio pack. The output records share readiness,
allowed claims, claims to avoid, recommended links, and next actions for CV or professor-outreach
use. Dry-run packets explicitly mark the run as smoke-demo evidence only.

## Notes

- Metrics are lightweight portfolio metrics and depend on manual annotations.
- Dry-run results are synthetic and only validate the evaluation pipeline.
- Review `scene_data_inspection.md` before interpreting real trained outputs.
