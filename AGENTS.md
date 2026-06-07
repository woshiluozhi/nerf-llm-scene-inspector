# Agent Instructions

- Use Python 3.10+ and keep code compatible with CPU-only CI.
- Use type hints and docstrings for public functions.
- Do not hardcode absolute machine-specific paths.
- Do not require GPU, Nerfstudio, LERF, OpenNeRF, COLMAP, FFmpeg, or API keys in tests.
- Treat Nerfstudio, LERF, OpenNeRF, Streamlit, Torch, and OpenAI as optional external dependencies.
- Every script that launches heavy external commands must support `--dry-run`.
- When modifying a script, run its `--help` command before finishing.
- When changing behavior, update the relevant README/docs/tests.
- Never commit API keys, tokens, checkpoints, raw datasets, large videos, generated training outputs, or heavy artifacts.
- Prefer graceful failure with actionable install commands over raw stack traces for missing external tools.
- Keep academic claims honest: this is a research engineering system built on upstream methods, not a new state-of-the-art NeRF architecture.
