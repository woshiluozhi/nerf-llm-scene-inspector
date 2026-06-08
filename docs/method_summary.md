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

## Experiment Matrices

The experiment-matrix runner makes the project easier to treat like a small research study rather than a single demo. A YAML file defines backend variants, query sets, prompt-sensitivity suites, relation-analysis settings, and dry-run or real-run execution. The runner launches or collects pipeline runs and writes a JSON/CSV/Markdown ablation table with evidence scores, localization metrics, prompt stability, relation-edge counts, and run links.

## Research Reports

Each pipeline run can produce `research_report.md` and `research_report.json`, a paper-style summary that pulls together the scene, backend, evidence scorecard, evaluation metrics, prompt-sensitivity diagnostics, scene-relation graph, reproducibility artifacts, limitations, and next steps. The report is intended for portfolio review and professor outreach, while still separating dry-run smoke evidence from real trained-scene claims.

## How This Project Differs From A Pure Reproduction

This project is built on Nerfstudio and LERF, but it adds a user-facing research engineering layer: reproducible CLI wrappers, dry-run mode, typed query artifacts, capture manifests, a local query planner, deterministic answer synthesis with evidence summaries, spatial-reasoning utilities, scene-relation graph reports, experiment-matrix summaries, research-report generation, visualization generation, annotation QA, prompt-sensitivity diagnostics, evaluation scaffolding, evidence scorecards, project/run-level static portfolio pages, and portfolio-ready documentation. It is intended to demonstrate implementation depth and research readiness without claiming a new algorithmic contribution.
