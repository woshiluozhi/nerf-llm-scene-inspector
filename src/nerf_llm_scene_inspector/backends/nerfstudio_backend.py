"""Shared Nerfstudio backend helpers."""

from __future__ import annotations

from pathlib import Path

from nerf_llm_scene_inspector.utils.shell import format_command


class NerfstudioConfigMixin:
    """Small mixin for backends that load a Nerfstudio config.yml."""

    backend_name = "nerfstudio"

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.config_path: str | None = None
        self.warnings: list[str] = []

    def load(self, config_path: str) -> None:
        path = Path(config_path)
        if not self.dry_run and not path.exists():
            raise FileNotFoundError(f"Config path does not exist: {config_path}")
        self.config_path = str(path)

    def viewer_command(self) -> list[str]:
        if not self.config_path:
            raise RuntimeError("Backend config has not been loaded.")
        return ["ns-viewer", "--load-config", self.config_path]

    def viewer_command_text(self) -> str:
        return format_command(self.viewer_command())
