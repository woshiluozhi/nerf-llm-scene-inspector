.PHONY: install test lint check demo-dry-run pipeline-dry-run experiment-matrix portfolio-pack clean-generated

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev,video]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check:
	$(PYTHON) scripts/check_env.py --json
	$(PYTHON) scripts/prepare_data.py --help
	$(PYTHON) scripts/inspect_scene_data.py --help
	$(PYTHON) scripts/train_baseline_nerf.py --help
	$(PYTHON) scripts/train_language_field.py --help
	$(PYTHON) scripts/query_scene.py --help
	$(PYTHON) scripts/generate_demo_assets.py --help
	$(PYTHON) scripts/analyze_prompt_sensitivity.py --help
	$(PYTHON) scripts/analyze_scene_relations.py --help
	$(PYTHON) scripts/evaluate_queries.py --help
	$(PYTHON) scripts/compare_runs.py --help
	$(PYTHON) scripts/run_experiment_matrix.py --help
	$(PYTHON) scripts/generate_research_report.py --help
	$(PYTHON) scripts/create_run_result_card.py --help
	$(PYTHON) scripts/create_submission_packet.py --help
	$(PYTHON) scripts/create_real_run_plan.py --help
	$(PYTHON) scripts/audit_claims.py --help
	$(PYTHON) scripts/run_scene_pipeline.py --help
	$(PYTHON) scripts/run_dry_run_demo.py --help
	$(PYTHON) -m pytest

demo-dry-run:
	$(PYTHON) scripts/run_dry_run_demo.py

pipeline-dry-run:
	$(PYTHON) scripts/run_scene_pipeline.py --dry-run --query mug

experiment-matrix:
	$(PYTHON) scripts/run_experiment_matrix.py --config examples/experiment_matrix.yaml --dry-run --limit 1

portfolio-pack:
	$(PYTHON) scripts/export_portfolio_pack.py --run-dir results/pipeline_runs/desk_scene --zip
	$(PYTHON) scripts/validate_portfolio_pack.py --pack results/portfolio_pack
	$(PYTHON) scripts/create_submission_packet.py --run-dir results/pipeline_runs/desk_scene --pack results/portfolio_pack --output results/submission_packet
	$(PYTHON) scripts/create_real_run_plan.py --run-dir results/pipeline_runs/desk_scene --output results/real_run_plan --input path/to/video.mp4 --type video --submission-packet results/submission_packet/submission_packet.json
	$(PYTHON) scripts/audit_claims.py --run-dir results/pipeline_runs/desk_scene --pack results/portfolio_pack --output results/claim_audit.json --markdown-output results/claim_audit.md

clean-generated:
	rm -rf data runs outputs viewer_logs wandb
	find results -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
