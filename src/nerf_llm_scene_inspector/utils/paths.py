"""Path helpers used by CLI scripts and tests."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    """Return the repository root from the installed source tree."""

    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to base or the current working directory."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base or Path.cwd()) / candidate
    return candidate.resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str, max_length: int = 64) -> str:
    """Create a filesystem-friendly slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        slug = "query"
    return slug[:max_length]


def find_latest_config(search_roots: list[str | Path]) -> Path | None:
    """Find the most recently modified Nerfstudio config.yml under search roots."""

    candidates: list[Path] = []
    for root in search_roots:
        path = Path(root)
        if path.is_file() and path.name in {"config.yml", "config.yaml"}:
            candidates.append(path)
        elif path.exists():
            candidates.extend(path.rglob("config.yml"))
            candidates.extend(path.rglob("config.yaml"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_config_paths_from_text(text: str) -> list[Path]:
    """Extract config paths from Nerfstudio-style logs."""

    patterns = [
        r"(?:Config File|config(?:\.yml)? saved to|Saved config to)[:\s]+([^\r\n]+config\.ya?ml)",
        r"(--load-config\s+)([^\r\n]+config\.ya?ml)",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            group = match.group(match.lastindex or 1)
            cleaned = group.strip().strip("'\"")
            if cleaned.startswith("--load-config"):
                cleaned = cleaned.replace("--load-config", "").strip()
            paths.append(Path(cleaned))
    return paths
