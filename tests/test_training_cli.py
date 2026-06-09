import json
import subprocess
import sys
from pathlib import Path

import pytest

from nerf_llm_scene_inspector import training
from nerf_llm_scene_inspector.utils.shell import CommandResult


ROOT = Path(__file__).resolve().parents[1]


def test_validate_ns_train_method_uses_exact_matching(monkeypatch) -> None:
    monkeypatch.setattr(training.shutil, "which", lambda _name: "ns-train")

    def fake_run_command(command, *, check=False, log_path=None):  # noqa: ANN001, ANN202
        return CommandResult(
            command=[str(item) for item in command],
            returncode=0,
            stdout="available methods: nerfacto lerf-lite lerf-big",
        )

    monkeypatch.setattr(training, "run_command", fake_run_command)

    training.validate_ns_train_method("lerf-lite", backend="lerf")
    with pytest.raises(RuntimeError, match="does not list method 'lerf'"):
        training.validate_ns_train_method("lerf", backend="lerf")


def test_train_baseline_nerf_validates_method_before_real_training(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_validate(method, *, backend, log_path=None):  # noqa: ANN001, ANN202
        calls.append((str(method), str(backend), str(log_path)))

    def fake_run_train(command, output_path, *, dry_run, config_name, command_log_path=None):  # noqa: ANN001, ANN202
        config = Path(output_path) / "config.yml"
        config.write_text(f"method_name: {config_name}\n", encoding="utf-8")
        return CommandResult(command=[str(item) for item in command], returncode=0)

    monkeypatch.setattr(training, "validate_ns_train_method", fake_validate)
    monkeypatch.setattr(training, "_run_train", fake_run_train)

    summary = training.train_baseline_nerf(
        tmp_path / "data",
        "nerfacto",
        tmp_path / "baseline",
        method_check_log_path=tmp_path / "logs" / "baseline_method_check.json",
    )

    assert summary["success"] is True
    assert calls == [("nerfacto", "nerfstudio", str(tmp_path / "logs" / "baseline_method_check.json"))]


def test_train_language_field_cli_opennerf_variant_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "language_opennerf"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_language_field.py"),
            "--data",
            str(tmp_path / "data"),
            "--backend",
            "opennerf",
            "--variant",
            "opennerf",
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    json_start = result.stdout.find("{")
    assert json_start >= 0, result.stdout
    summary = json.loads(result.stdout[json_start:].split("\nLaunch viewer:")[0])
    assert summary["backend"] == "opennerf"
    assert summary["variant"] == "opennerf"
    assert summary["method"] == "opennerf"
    assert (output / "config.yml").exists()
