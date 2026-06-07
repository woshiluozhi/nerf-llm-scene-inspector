# Engineering Pass Note

Baseline checks before this pass:

- `python -m pip install -e ".[dev,video,dashboard]"` succeeded.
- `pytest` passed with the original 8 tests.
- `python scripts/run_dry_run_demo.py` passed.
- Weaknesses found: no environment diagnostic script, no AGENTS.md, shallow planner coverage, single-view LERF dry-run, no strict backend mode, no qualitative evaluation report, limited portfolio card material, and CI did not run the dry-run demo.

Fix focus:

- Keep GPU/upstream dependencies optional.
- Improve CPU-safe validation and dry-run quality.
- Add tests around planner, LERF fallback, environment checks, and spatial/evaluation logic.
- Preserve honest research positioning without claiming algorithmic novelty.
