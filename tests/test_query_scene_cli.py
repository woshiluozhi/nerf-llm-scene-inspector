import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_query_scene_cli_writes_answer_summary_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "query"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_scene.py"),
            "--config",
            str(tmp_path / "config.yml"),
            "--backend",
            "lerf",
            "--query",
            "mug",
            "--output",
            str(output),
            "--scene-name",
            "desk_scene",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "scene_query_report.json").read_text(encoding="utf-8"))
    assert report["scene_name"] == "desk_scene"
    assert report["answer_summary"]["support_level"] == "2d_relevancy_fallback"
    assert (output / "scene_query_report.md").exists()
    assert "Wrote markdown report" in result.stdout
