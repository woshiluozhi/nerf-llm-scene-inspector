from nerf_llm_scene_inspector.utils import env_check


def test_check_import_missing(monkeypatch) -> None:
    monkeypatch.setattr(env_check.importlib.util, "find_spec", lambda _name: None)
    item = env_check.check_import("streamlit")
    assert not item.ok
    assert item.category == "python_import"
    assert "Install" in item.hint


def test_check_command_missing(monkeypatch) -> None:
    monkeypatch.setattr(env_check.shutil, "which", lambda _name: None)
    item = env_check.check_command("ns-train", required=True)
    assert not item.ok
    assert item.required
    assert "Nerfstudio" in item.hint


def test_format_report_table() -> None:
    report = env_check.EnvReport(
        ok=False,
        python_version="3.11",
        platform="test",
        checks=[env_check.CheckItem("x", False, "optional", "missing", False, "install x")],
    )
    text = env_check.format_report_table(report, verbose=True)
    assert "Python: 3.11" in text
    assert "install x" in text
