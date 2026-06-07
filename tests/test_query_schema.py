from pathlib import Path

from nerf_llm_scene_inspector.backends.base import (
    BoundingRegion,
    Candidate3DPoint,
    QueryResult,
    RenderedView,
)


def test_query_result_json_roundtrip(tmp_path: Path) -> None:
    result = QueryResult(
        query="mug",
        backend_name="lerf",
        config_path="runs/language/config.yml",
        rendered_images=[RenderedView(path="overlay.png", kind="overlay", query="mug")],
        candidate_points=[Candidate3DPoint(label="mug", x=0.0, y=1.0, z=2.0, score=0.8)],
        bounding_regions=[
            BoundingRegion(label="mug", score=0.9, bbox_2d=(10.0, 10.0, 50.0, 50.0))
        ],
        confidence=0.9,
    )
    path = tmp_path / "query_result.json"
    result.to_json(path)
    loaded = QueryResult.from_json(path)
    assert loaded.query == "mug"
    assert loaded.rendered_images[0].kind == "overlay"
    assert loaded.candidate_points[0].z == 2.0
    assert loaded.bounding_regions[0].bbox_2d == (10.0, 10.0, 50.0, 50.0)
    assert "timestamp" in loaded.provenance
