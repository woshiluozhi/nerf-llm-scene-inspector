from pathlib import Path

from nerf_llm_scene_inspector.config import InspectorConfig, load_config
from nerf_llm_scene_inspector.utils.paths import resolve_path, slugify


ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    assert isinstance(config, InspectorConfig)
    assert config.project_name == "nerf-llm-scene-inspector"
    assert config.backend.primary == "lerf"
    assert config.backend.language_variant == "lerf-lite"
    assert config.query.top_k == 5


def test_path_resolution_and_slugify(tmp_path: Path) -> None:
    resolved = resolve_path("data/processed/scene", base=tmp_path)
    assert resolved == (tmp_path / "data" / "processed" / "scene").resolve()
    assert slugify("Find objects related to coffee!") == "find_objects_related_to_coffee"
