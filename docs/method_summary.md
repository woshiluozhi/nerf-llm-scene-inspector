# Method Summary

## What NeRF Is

A Neural Radiance Field represents a scene as a continuous function that maps 3D position and viewing direction to color and density. After optimization from posed images, the field can synthesize novel views and preserve scene geometry implicitly through volumetric density.

## What LERF Adds

LERF, Language Embedded Radiance Fields, augments a NeRF-style scene representation with language-aligned features distilled from vision-language models. Instead of only rendering RGB and depth, the trained field can render relevancy maps for text prompts such as "mug", "keyboard", or "objects for making coffee".

## Why CLIP/VLM Features Matter

CLIP and related vision-language models embed images and text into a shared feature space. Distilling those features into a 3D field allows open-vocabulary querying because the query does not need to be one of a fixed set of training labels. The output is an approximate semantic relevancy signal over rendered views and, when available, 3D samples.

## Relevance To Embodied AI And Robotics

Robots and embodied agents need persistent scene representations that connect geometry, object semantics, affordances, and language instructions. A language-embedded NeRF is useful as a research substrate because it connects real-world 3D reconstruction with natural-language object search, spatial reasoning, and scene-level question answering.

## Scene Relation Graphs

The project now includes a deterministic relation-analysis layer that converts saved `QueryResult` regions and candidate 3D points into a compact entity-relation graph. When 3D points are available, relations are marked as `3d`; otherwise the report explicitly marks image-space heuristics as `2d_fallback`. This is useful for portfolio evidence and embodied-AI-style questions such as support, containment, proximity, and left/right layout, but it is not presented as a learned physical relation model.

## Query Planning And Execution

Natural-language tasks are first converted into structured query plans with primary,
supporting, negative, and relation-hypothesis fields. The CLI and Python query engine share
the same backend-call expansion helper so a task produces the same concrete text queries
whether it is run from `query_scene.py` or imported as `SemanticQueryEngine`. By default,
primary and supporting prompts are executed, negative prompts are kept for disambiguation,
and all answers explicitly report the evidence level, limitations, and recommended follow-up
checks. When negative prompts are explicitly executed, their results are tagged in provenance
and excluded from positive answer evidence so disambiguation artifacts do not become claimed
detections.

## Experiment Matrices

The experiment-matrix runner makes the project easier to treat like a small research study rather than a single demo. A YAML file defines backend variants, query sets, prompt-sensitivity suites, relation-analysis settings, and dry-run or real-run execution. The runner launches or collects pipeline runs and writes a JSON/CSV/Markdown ablation table with evidence scores, localization metrics, prompt stability, relation-edge counts, and run links.

## Research Reports

Each pipeline run can produce `research_report.md` and `research_report.json`, a paper-style summary that pulls together the scene, backend, evidence scorecard, evaluation metrics, prompt-sensitivity diagnostics, scene-relation graph, reproducibility artifacts, limitations, and next steps. The report is intended for portfolio review and professor outreach, while still separating dry-run smoke evidence from real trained-scene claims.

Each run can also produce `run_result_card.md` and `run_result_card.json`, a one-page reviewer-facing summary. It turns the same artifacts into a primary takeaway, shareable blurb, evidence snapshot, metrics, caveats, safe claims to avoid, readiness checks, and next actions. This is meant to help a professor or recruiter quickly understand what the run demonstrates before opening the full research report.

## Annotation Workbench

The evaluation path includes an offline HTML annotation workbench generated from `annotation_template.json` and query render artifacts. It copies candidate images, preloads candidate boxes, and lets a reviewer draw `bbox_2d` labels in the browser. The downloaded JSON is merged back into a clean evaluation annotation file, validated, visually reviewed, and evaluated. In normal run-scoped usage, `finalize_annotations.py` orchestrates those lower-level steps so quantitative metrics stay tied to explicit human-reviewed labels rather than hidden assumptions.

After manual labels are exported, `finalize_annotations.py` provides the practical one-command path for refreshing a run: merge filled labels, rerun evaluation and annotation visual QA, update audit/recommendation/scorecard/quality-gate artifacts, regenerate research reports, result cards, portfolio pages, reproduction bundles, run indexes, and optionally export a fresh portfolio pack.

## Submission Packets

The submission-packet step converts run evidence into a claim-calibrated sharing checklist. It records what can safely be said in a CV bullet or professor email, what must not be claimed, whether a validated portfolio pack exists, and which warnings still need review. This keeps dry-run demos, real trained-scene results, and portfolio-ready evidence clearly separated.

## Real-Run Action Plans

Each run can also produce `real_run_plan.md` and `real_run_plan.json`, a concrete playbook for moving from smoke evidence to a real CUDA/Nerfstudio/LERF run. The plan reads the existing pipeline summary, preflight report, capture validation, quality gate, recommendations, and submission packet, then emits phased commands for capture metadata, preflight, data processing, training, annotation review, quality gates, pack validation, and sharing.

## Claim Audits

Before sharing a run externally, `claim_audit.md` scans README/docs, run reports, and optional portfolio packs for unsupported wording such as unqualified state-of-the-art claims, novel-architecture claims, benchmark superiority, production guarantees, or implemented robotics policy claims. The audit also checks that dry-run, upstream-attribution, no-new-architecture, no-SOTA, and GPU-requirement disclaimers are present.

## How This Project Differs From A Pure Reproduction

This project is built on Nerfstudio and LERF, but it adds a user-facing research engineering layer: reproducible CLI wrappers, dry-run mode, typed query artifacts, capture manifests, a local query planner, deterministic answer synthesis with evidence summaries, spatial-reasoning utilities, scene-relation graph reports, experiment-matrix summaries, research-report generation, real-run action planning, claim auditing, submission-packet generation, visualization generation, annotation QA, prompt-sensitivity diagnostics, evaluation scaffolding, evidence scorecards, project/run-level static portfolio pages, and portfolio-ready documentation. It is intended to demonstrate implementation depth and research readiness without claiming a new algorithmic contribution.
