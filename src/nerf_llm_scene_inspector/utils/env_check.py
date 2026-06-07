"""CPU-safe environment diagnostics for local wrappers and optional upstream tools."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable


INSTALL_HINTS = {
    "numpy": "Install project dependencies: python -m pip install -e .",
    "PIL": "Install Pillow: python -m pip install pillow",
    "yaml": "Install PyYAML: python -m pip install pyyaml",
    "streamlit": "Install dashboard extras: python -m pip install -e .[dashboard]",
    "imageio": "Install video extras: python -m pip install -e .[video]",
    "torch": "Install a CUDA-compatible PyTorch build from https://pytorch.org/get-started/locally/",
    "nerfstudio": "Install Nerfstudio: python -m pip install nerfstudio && ns-install-cli",
    "openai": "Install optional OpenAI client: python -m pip install openai",
    "ns-process-data": "Install Nerfstudio and run ns-install-cli.",
    "ns-train": "Install Nerfstudio/LERF and run ns-install-cli.",
    "ns-viewer": "Install Nerfstudio and run ns-install-cli.",
    "colmap": "Install COLMAP, for example: conda install -c conda-forge colmap",
    "ffmpeg": "Install FFmpeg, for example: conda install -c conda-forge ffmpeg",
    "gpu": "Install CUDA-compatible PyTorch and run on an NVIDIA GPU machine.",
}


@dataclass
class CheckItem:
    """One diagnostic check result."""

    name: str
    ok: bool
    category: str
    detail: str = ""
    required: bool = False
    hint: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EnvReport:
    """Structured environment report."""

    ok: bool
    python_version: str
    platform: str
    checks: list[CheckItem] = field(default_factory=list)
    strict_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "python_version": self.python_version,
            "platform": self.platform,
            "checks": [check.to_dict() for check in self.checks],
            "strict_failures": list(self.strict_failures),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def check_import(module_name: str, *, required: bool = False) -> CheckItem:
    """Check whether a module can be imported without importing it."""

    ok = importlib.util.find_spec(module_name) is not None
    return CheckItem(
        name=module_name,
        ok=ok,
        category="python_import",
        detail="available" if ok else "missing",
        required=required,
        hint="" if ok else INSTALL_HINTS.get(module_name, ""),
    )


def check_command(command_name: str, *, required: bool = False) -> CheckItem:
    """Check whether a command exists on PATH."""

    path = shutil.which(command_name)
    return CheckItem(
        name=command_name,
        ok=path is not None,
        category="shell_command",
        detail=path or "missing",
        required=required,
        hint="" if path else INSTALL_HINTS.get(command_name, ""),
    )


def check_cuda(require_gpu: bool = False) -> CheckItem:
    """Check CUDA availability if torch is installed."""

    if importlib.util.find_spec("torch") is None:
        return CheckItem(
            name="cuda",
            ok=not require_gpu,
            category="gpu",
            detail="torch not installed",
            required=require_gpu,
            hint=INSTALL_HINTS["torch"] if require_gpu else INSTALL_HINTS["torch"],
        )
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        detail = f"available={available}"
        if available:
            detail += f", device_count={torch.cuda.device_count()}"
        return CheckItem(
            name="cuda",
            ok=available or not require_gpu,
            category="gpu",
            detail=detail,
            required=require_gpu,
            hint="" if available else INSTALL_HINTS["gpu"],
        )
    except Exception as exc:  # pragma: no cover - depends on external torch state
        return CheckItem(
            name="cuda",
            ok=not require_gpu,
            category="gpu",
            detail=f"torch import failed: {exc}",
            required=require_gpu,
            hint=INSTALL_HINTS["torch"],
        )


def check_ns_train_methods(
    expected_methods: Iterable[str] = ("nerfacto", "lerf", "lerf-lite", "lerf-big", "opennerf"),
    *,
    required: bool = False,
    timeout_seconds: int = 20,
) -> list[CheckItem]:
    """Inspect `ns-train -h` for expected method names when available."""

    if shutil.which("ns-train") is None:
        return [
            CheckItem(
                name=f"ns-train method:{method}",
                ok=False,
                category="ns_train_method",
                detail="ns-train missing",
                required=required,
                hint=INSTALL_HINTS["ns-train"],
            )
            for method in expected_methods
        ]
    try:
        proc = subprocess.run(
            ["ns-train", "-h"],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        help_text = f"{proc.stdout}\n{proc.stderr}"
    except Exception as exc:  # pragma: no cover - depends on external command behavior
        return [
            CheckItem(
                name=f"ns-train method:{method}",
                ok=False,
                category="ns_train_method",
                detail=f"failed to inspect ns-train: {exc}",
                required=required,
                hint=INSTALL_HINTS["ns-train"],
            )
            for method in expected_methods
        ]
    return [
        CheckItem(
            name=f"ns-train method:{method}",
            ok=method in help_text,
            category="ns_train_method",
            detail="listed" if method in help_text else "not listed",
            required=required,
            hint="" if method in help_text else INSTALL_HINTS["ns-train"],
        )
        for method in expected_methods
    ]


def build_env_report(require_gpu: bool = False, check_upstream: bool = False) -> EnvReport:
    """Build a full environment report without requiring optional dependencies."""

    checks: list[CheckItem] = [
        CheckItem(
            name="python>=3.10",
            ok=sys.version_info >= (3, 10),
            category="runtime",
            detail=sys.version.split()[0],
            required=True,
            hint="Use Python 3.10 or newer.",
        ),
        check_import("nerf_llm_scene_inspector", required=True),
    ]
    for module in ["numpy", "PIL", "yaml", "streamlit", "imageio", "torch", "nerfstudio", "openai"]:
        checks.append(check_import(module, required=module in {"numpy", "PIL", "yaml"}))
    for command in ["ns-process-data", "ns-train", "ns-viewer", "colmap", "ffmpeg"]:
        checks.append(check_command(command, required=check_upstream))
    checks.append(check_cuda(require_gpu=require_gpu))
    checks.extend(check_ns_train_methods(required=check_upstream))

    strict_failures = [check.name for check in checks if check.required and not check.ok]
    return EnvReport(
        ok=not strict_failures,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        checks=checks,
        strict_failures=strict_failures,
    )


def format_report_table(report: EnvReport, *, verbose: bool = False) -> str:
    """Format an environment report as a readable table."""

    lines = [
        f"Python: {report.python_version}",
        f"Platform: {report.platform}",
        f"Overall: {'PASS' if report.ok else 'WARN/FAIL'}",
        "",
        f"{'Status':<8} {'Category':<18} {'Name':<32} Detail",
        f"{'-' * 8} {'-' * 18} {'-' * 32} {'-' * 20}",
    ]
    for check in report.checks:
        if not verbose and check.ok and not check.required:
            continue
        status = "OK" if check.ok else ("FAIL" if check.required else "MISS")
        lines.append(f"{status:<8} {check.category:<18} {check.name:<32} {check.detail}")
        if verbose and check.hint and not check.ok:
            lines.append(f"{'':<8} {'hint':<18} {'':<32} {check.hint}")
    if report.strict_failures:
        lines.extend(["", "Required failures:", *[f"- {name}" for name in report.strict_failures]])
    return "\n".join(lines)
