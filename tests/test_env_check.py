from nerf_llm_scene_inspector.utils import env_check


def test_ns_train_method_listed_uses_exact_tokens() -> None:
    help_text = "Available methods: nerfacto, lerf-lite, lerf-big, opennerf2"

    assert env_check.ns_train_method_listed("lerf-lite", help_text)
    assert not env_check.ns_train_method_listed("lerf", help_text)
    assert not env_check.ns_train_method_listed("opennerf", help_text)


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


def test_check_ns_train_methods_uses_exact_matching(monkeypatch) -> None:
    class FakeProc:
        stdout = "methods: nerfacto lerf-lite"
        stderr = ""

    monkeypatch.setattr(env_check.shutil, "which", lambda _name: "ns-train")
    monkeypatch.setattr(env_check.subprocess, "run", lambda *args, **kwargs: FakeProc())

    checks = env_check.check_ns_train_methods(["lerf", "lerf-lite"], required=True)
    by_name = {check.name: check for check in checks}

    assert by_name["ns-train method:lerf"].ok is False
    assert by_name["ns-train method:lerf"].required is True
    assert by_name["ns-train method:lerf-lite"].ok is True


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
