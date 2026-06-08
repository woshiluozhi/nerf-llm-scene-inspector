from pathlib import Path

from nerf_llm_scene_inspector.utils.provenance import build_provenance


def test_build_provenance_without_git_repo_is_nonfatal(tmp_path: Path) -> None:
    provenance = build_provenance(command=["run_scene_pipeline.py", "--dry-run"], repo_root=tmp_path)

    payload = provenance.to_dict()
    assert payload["project_version"] == "0.1.0"
    assert payload["command"] == ["run_scene_pipeline.py", "--dry-run"]
    assert payload["working_directory"] == str(tmp_path)
    assert payload["git_available"] is False
    assert payload["git_commit"] is None
    assert payload["warnings"]
