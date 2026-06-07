"""Subprocess helpers with dry-run support."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from nerf_llm_scene_inspector.utils.logging import get_logger


@dataclass
class CommandResult:
    """Serializable subprocess result."""

    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def format_command(command: Sequence[str]) -> str:
    """Return a shell-copyable command string."""

    return " ".join(shlex.quote(str(part)) for part in command)


def require_executable(name: str, install_hint: str | None = None) -> None:
    """Raise a helpful error if an executable is missing."""

    if shutil.which(name) is None:
        hint = f"\n\nInstall hint:\n{install_hint}" if install_hint else ""
        raise RuntimeError(f"Required executable '{name}' was not found on PATH.{hint}")


def run_command(
    command: Sequence[str],
    *,
    dry_run: bool = False,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = False,
) -> CommandResult:
    """Run a command or return a dry-run result."""

    logger = get_logger()
    command_list = [str(part) for part in command]
    logger.info("Command: %s", format_command(command_list))

    if dry_run:
        result = CommandResult(
            command=command_list,
            returncode=0,
            stdout=f"[dry-run] {format_command(command_list)}",
            dry_run=True,
        )
        _write_log(log_path, result)
        return result

    result_proc = subprocess.run(
        command_list,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(
        command=command_list,
        returncode=result_proc.returncode,
        stdout=result_proc.stdout,
        stderr=result_proc.stderr,
        dry_run=False,
    )
    _write_log(log_path, result)
    if check and not result.ok:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {format_command(command_list)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def _write_log(log_path: str | Path | None, result: CommandResult) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
