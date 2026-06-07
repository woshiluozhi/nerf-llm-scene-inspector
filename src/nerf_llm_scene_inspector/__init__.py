"""NeRF-LLM Scene Inspector package."""

from nerf_llm_scene_inspector.backends.base import (
    BoundingRegion,
    Candidate3DPoint,
    QueryResult,
    RenderedView,
    SceneQueryReport,
)

__all__ = [
    "BoundingRegion",
    "Candidate3DPoint",
    "QueryResult",
    "RenderedView",
    "SceneQueryReport",
]

__version__ = "0.1.0"
