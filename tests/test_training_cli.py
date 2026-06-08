import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
