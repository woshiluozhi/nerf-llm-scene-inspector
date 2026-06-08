.PHONY: install test lint check demo-dry-run pipeline-dry-run portfolio-pack clean-generated

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
	$(PYTHON) scripts/evaluate_queries.py --help
	$(PYTHON) scripts/run_scene_pipeline.py --help
	$(PYTHON) scripts/run_dry_run_demo.py --help
	$(PYTHON) -m pytest

demo-dry-run:
	$(PYTHON) scripts/run_dry_run_demo.py

pipeline-dry-run:
	$(PYTHON) scripts/run_scene_pipeline.py --dry-run --query mug

portfolio-pack:
	$(PYTHON) scripts/export_portfolio_pack.py --run-dir results/pipeline_runs/desk_scene --zip

clean-generated:
	rm -rf data runs outputs viewer_logs wandb
	find results -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
