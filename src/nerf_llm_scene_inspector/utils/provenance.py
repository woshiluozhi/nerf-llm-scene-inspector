"""Reproducibility provenance for experiment and demo runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

from nerf_llm_scene_inspector import __version__
from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


@dataclass
class ReproducibilityProvenance:
    """Portable metadata that helps reproduce a pipeline run."""

    timestamp: str
    project_version: str
    python_version: str
    platform: str
    command: list[str] = field(default_factory=list)
    working_directory: str | None = None
    git_available: bool = False
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_remote: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_provenance(
    *,
    command: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
) -> ReproducibilityProvenance:
    """Collect run provenance without requiring git to be installed."""

    root = Path(repo_root) if repo_root is not None else project_root()
    warnings: list[str] = []
    git_commit = _git_text(root, "rev-parse", "HEAD", warnings=warnings)
    git_branch = _git_text(root, "branch", "--show-current", warnings=warnings)
    git_status = _git_text(root, "status", "--porcelain", warnings=warnings)
    git_remote = _git_text(root, "remote", "get-url", "origin", warnings=warnings)
    git_available = git_commit is not None

    return ReproducibilityProvenance(
        timestamp=utc_timestamp(),
        project_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        command=list(command or []),
        working_directory=str(root),
        git_available=git_available,
        git_commit=git_commit,
        git_branch=git_branch or None,
        git_dirty=bool(git_status) if git_available else None,
        git_remote=_sanitize_remote(git_remote) if git_remote else None,
        warnings=_dedupe_warnings(warnings),
    )


def _git_text(root: Path, *args: str, warnings: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        warnings.append(f"Could not run git {' '.join(args)}: {exc}")
        return None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        warnings.append(f"git {' '.join(args)} failed: {detail}")
        return None
    return proc.stdout.strip()


def _sanitize_remote(remote: str) -> str:
    if "://" not in remote:
        return remote
    parsed = urlsplit(remote)
    if "@" not in parsed.netloc:
        return remote
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        if warning not in seen:
            deduped.append(warning)
            seen.add(warning)
    return deduped
