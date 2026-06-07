"""Configuration loading for the scene inspector project."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BackendConfig:
    """Semantic backend configuration."""

    primary: str = "lerf"
    language_variant: str = "lerf-lite"
    opennerf_enabled: bool = False


@dataclass
class RuntimeConfig:
    """Paths and execution defaults."""

    dry_run: bool = True
    data_dir: str = "data/processed"
    runs_dir: str = "runs"
    results_dir: str = "results"


@dataclass
class QueryConfig:
    """Default query settings."""

    top_k: int = 5
    render_height: int = 512
    render_width: int = 512


@dataclass
class InspectorConfig:
    """Top-level project configuration."""

    project_name: str = "nerf-llm-scene-inspector"
    scene_name: str = "desk_scene"
    backend: BackendConfig = field(default_factory=BackendConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    query: QueryConfig = field(default_factory=QueryConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InspectorConfig":
        backend_raw = raw.get("backend") or {}
        runtime_raw = raw.get("runtime") or {}
        query_raw = raw.get("query") or {}
        return cls(
            project_name=str(raw.get("project_name", cls.project_name)),
            scene_name=str(raw.get("scene_name", cls.scene_name)),
            backend=BackendConfig(
                primary=str(backend_raw.get("primary", "lerf")),
                language_variant=str(backend_raw.get("language_variant", "lerf-lite")),
                opennerf_enabled=bool(backend_raw.get("opennerf_enabled", False)),
            ),
            runtime=RuntimeConfig(
                dry_run=bool(runtime_raw.get("dry_run", True)),
                data_dir=str(runtime_raw.get("data_dir", "data/processed")),
                runs_dir=str(runtime_raw.get("runs_dir", "runs")),
                results_dir=str(runtime_raw.get("results_dir", "results")),
            ),
            query=QueryConfig(
                top_k=int(query_raw.get("top_k", 5)),
                render_height=int(query_raw.get("render_height", 512)),
                render_width=int(query_raw.get("render_width", 512)),
            ),
        )


def load_config(path: str | Path) -> InspectorConfig:
    """Load an InspectorConfig from YAML.

    PyYAML is used when available. A small fallback parser keeps CLI help and tests
    usable in bare Python environments.
    """

    mapping = load_mapping(path)
    return InspectorConfig.from_dict(mapping)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML-like mapping from disk."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a mapping in {config_path}")
        return loaded
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the limited YAML subset used by this project.

    Supported constructs: nested mappings by two-space indentation, scalar
    values, and dash lists. This is not a general YAML parser.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    last_key_for_indent: dict[int, str] = {}

    for original_line in text.splitlines():
        if not original_line.strip() or original_line.lstrip().startswith("#"):
            continue
        indent = len(original_line) - len(original_line.lstrip(" "))
        line = original_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if line.startswith("- "):
            value = _parse_scalar(line[2:].strip())
            if not isinstance(container, list):
                parent_indent = stack[-1][0]
                parent = stack[-2][1] if len(stack) >= 2 else root
                key = last_key_for_indent.get(parent_indent)
                if isinstance(parent, dict) and key is not None:
                    new_list: list[Any] = []
                    parent[key] = new_list
                    stack[-1] = (parent_indent, new_list)
                    container = new_list
                else:
                    raise ValueError(f"List item without list parent: {line}")
            container.append(value)
            continue

        if ":" not in line:
            raise ValueError(f"Unsupported config line: {line}")
        key, value_text = line.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not isinstance(container, dict):
            raise ValueError(f"Cannot assign mapping key under list: {line}")
        if value_text == "":
            new_map: dict[str, Any] = {}
            container[key] = new_map
            last_key_for_indent[indent] = key
            stack.append((indent, new_map))
        else:
            container[key] = _parse_scalar(value_text)
            last_key_for_indent[indent] = key
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
