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
