"""Training command wrappers for Nerfstudio, LERF, and OpenNeRF."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from nerf_llm_scene_inspector.backends.lerf_backend import LERF_INSTALL_INSTRUCTIONS
from nerf_llm_scene_inspector.backends.opennerf_backend import OPENNERF_INSTALL_INSTRUCTIONS
from nerf_llm_scene_inspector.utils.paths import (
    extract_config_paths_from_text,
    find_latest_config,
    utc_timestamp,
)
from nerf_llm_scene_inspector.utils.shell import CommandResult, require_executable, run_command


NERFSTUDIO_TRAIN_HINT = """Install Nerfstudio:
python -m pip install nerfstudio
ns-install-cli
ns-train -h"""


def train_baseline_nerf(
    data: str | Path,
    method: str,
    output: str | Path,
    *,
    max_num_iterations: int | None = None,
    dry_run: bool = False,
    command_log_path: str | Path | None = None,
) -> dict[str, object]:
    """Train a baseline Nerfstudio model such as nerfacto."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    command = ["ns-train", method, "--data", str(data)]
    if max_num_iterations is not None:
        command.extend(["--max-num-iterations", str(max_num_iterations)])
    command.extend(["--output-dir", str(output_path)])

    result = _run_train(
        command,
        output_path,
        dry_run=dry_run,
        config_name=method,
        command_log_path=command_log_path,
    )
    config_path = _resolve_config_path(result, [output_path, Path("outputs")], dry_run=dry_run)
    summary = _train_summary(
        run_type="baseline",
        method=method,
        data=data,
        output=output_path,
        command_result=result,
        config_path=config_path,
        dry_run=dry_run,
    )
    _write_train_summary(summary, "baseline", output_path.name)
    return summary


def train_language_field(
    data: str | Path,
    backend: str,
    variant: str,
    output: str | Path,
    *,
    max_num_iterations: int | None = None,
    dry_run: bool = False,
    command_log_path: str | Path | None = None,
    method_check_log_path: str | Path | None = None,
) -> dict[str, object]:
    """Train a LERF or OpenNeRF language field."""

    backend = backend.lower()
    if backend not in {"lerf", "opennerf"}:
        raise ValueError("--backend must be 'lerf' or 'opennerf'")
    method = _language_method_name(backend, variant)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        validate_ns_train_method(method, backend=backend, log_path=method_check_log_path)

    command = ["ns-train", method, "--data", str(data)]
    if max_num_iterations is not None:
        command.extend(["--max-num-iterations", str(max_num_iterations)])
    command.extend(["--output-dir", str(output_path)])
    result = _run_train(
        command,
        output_path,
        dry_run=dry_run,
        config_name=method,
        command_log_path=command_log_path,
    )
    config_path = _resolve_config_path(result, [output_path, Path("outputs")], dry_run=dry_run)
    summary = _train_summary(
        run_type="language",
        method=method,
        data=data,
        output=output_path,
        command_result=result,
        config_path=config_path,
        dry_run=dry_run,
    )
    summary["backend"] = backend
    summary["variant"] = variant
    _write_train_summary(summary, "language", output_path.name)
    return summary


def validate_ns_train_method(
    method: str,
    *,
    backend: str,
    log_path: str | Path | None = None,
) -> None:
    """Check that ns-train recognizes the requested method."""

    if shutil.which("ns-train") is None:
        hint = LERF_INSTALL_INSTRUCTIONS if backend == "lerf" else OPENNERF_INSTALL_INSTRUCTIONS
        raise RuntimeError(f"ns-train was not found on PATH.\n\n{NERFSTUDIO_TRAIN_HINT}\n\n{hint}")
    result = run_command(["ns-train", "-h"], check=False, log_path=log_path)
    if not result.ok:
        raise RuntimeError(f"Could not inspect ns-train methods:\n{result.stderr}")
    help_text = f"{result.stdout}\n{result.stderr}"
    if method not in help_text:
        hint = LERF_INSTALL_INSTRUCTIONS if backend == "lerf" else OPENNERF_INSTALL_INSTRUCTIONS
        raise RuntimeError(f"ns-train does not list method '{method}'.\n\n{hint}")


def _language_method_name(backend: str, variant: str) -> str:
    if backend == "opennerf":
        return "opennerf"
    valid = {"lerf", "lerf-lite", "lerf-big"}
    if variant not in valid:
        raise ValueError("--variant must be one of: lerf, lerf-lite, lerf-big")
    return variant


def _run_train(
    command: list[str],
    output_path: Path,
    *,
    dry_run: bool,
    config_name: str,
    command_log_path: str | Path | None = None,
) -> CommandResult:
    if dry_run:
        mock_config = output_path / "config.yml"
        mock_config.write_text(
            "\n".join(
                [
                    f"method_name: {config_name}",
                    "pipeline:",
                    "  datamanager:",
                    "    data: data/processed/mock_scene",
                ]
            ),
            encoding="utf-8",
        )
    else:
        require_executable("ns-train", NERFSTUDIO_TRAIN_HINT)
    return run_command(
        command,
        dry_run=dry_run,
        check=False,
        log_path=command_log_path or output_path / "train_command.json",
    )


def _resolve_config_path(
    result: CommandResult,
    search_roots: list[Path],
    *,
    dry_run: bool,
) -> str | None:
    if dry_run:
        latest = find_latest_config(search_roots)
        return str(latest) if latest else None
    text_paths = extract_config_paths_from_text(f"{result.stdout}\n{result.stderr}")
    for path in text_paths:
        if path.exists():
            return str(path)
    latest = find_latest_config(search_roots)
    return str(latest) if latest else None


def _train_summary(
    *,
    run_type: str,
    method: str,
    data: str | Path,
    output: Path,
    command_result: CommandResult,
    config_path: str | None,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "run_type": run_type,
        "method": method,
        "data": str(data),
        "output": str(output),
        "timestamp": utc_timestamp(),
        "command": command_result.command,
        "returncode": command_result.returncode,
        "dry_run": dry_run,
        "success": command_result.ok and config_path is not None,
        "config_path": config_path,
        "viewer_command": ["ns-viewer", "--load-config", config_path] if config_path else None,
        "stdout_tail": command_result.stdout[-2000:],
        "stderr_tail": command_result.stderr[-2000:],
    }


def _write_train_summary(summary: dict[str, object], prefix: str, run_name: str) -> Path:
    normalized_name = run_name if run_name.startswith(f"{prefix}_") else f"{prefix}_{run_name}"
    results_dir = Path("results") / normalized_name
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "train_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
