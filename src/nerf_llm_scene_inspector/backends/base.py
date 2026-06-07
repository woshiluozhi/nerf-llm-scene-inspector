"""Backend interface and shared query dataclasses."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


CoordinateFrame = Literal["world", "camera", "image", "unknown"]


@dataclass
class BoundingRegion:
    """A 2D or 3D region associated with a semantic query."""

    label: str
    score: float | None = None
    coordinate_frame: CoordinateFrame = "image"
    bbox_2d: tuple[float, float, float, float] | None = None
    min_point_3d: tuple[float, float, float] | None = None
    max_point_3d: tuple[float, float, float] | None = None
    source_view: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BoundingRegion":
        return cls(
            label=str(raw.get("label", "")),
            score=_optional_float(raw.get("score")),
            coordinate_frame=raw.get("coordinate_frame", "image"),
            bbox_2d=_optional_tuple(raw.get("bbox_2d"), 4),
            min_point_3d=_optional_tuple(raw.get("min_point_3d"), 3),
            max_point_3d=_optional_tuple(raw.get("max_point_3d"), 3),
            source_view=raw.get("source_view"),
            notes=raw.get("notes"),
        )


@dataclass
class Candidate3DPoint:
    """Approximate 3D point returned by a semantic backend."""

    label: str
    x: float
    y: float
    z: float
    score: float | None = None
    source_view: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Candidate3DPoint":
        return cls(
            label=str(raw.get("label", "")),
            x=float(raw.get("x", 0.0)),
            y=float(raw.get("y", 0.0)),
            z=float(raw.get("z", 0.0)),
            score=_optional_float(raw.get("score")),
            source_view=raw.get("source_view"),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class RenderedView:
    """A rendered RGB image, heatmap, overlay, or diagnostic artifact."""

    path: str
    kind: str
    query: str
    caption: str | None = None
    camera_id: str | None = None
    width: int | None = None
    height: int | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RenderedView":
        return cls(
            path=str(raw.get("path", "")),
            kind=str(raw.get("kind", "artifact")),
            query=str(raw.get("query", "")),
            caption=raw.get("caption"),
            camera_id=raw.get("camera_id"),
            width=_optional_int(raw.get("width")),
            height=_optional_int(raw.get("height")),
            score=_optional_float(raw.get("score")),
        )


@dataclass
class QueryResult:
    """Structured output for one text query against a semantic field."""

    query: str
    backend_name: str
    config_path: str
    rendered_images: list[RenderedView] = field(default_factory=list)
    candidate_points: list[Candidate3DPoint] = field(default_factory=list)
    bounding_regions: list[BoundingRegion] = field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provenance.setdefault("timestamp", utc_timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "backend_name": self.backend_name,
            "config_path": self.config_path,
            "rendered_images": [view.to_dict() for view in self.rendered_images],
            "candidate_points": [point.to_dict() for point in self.candidate_points],
            "bounding_regions": [region.to_dict() for region in self.bounding_regions],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QueryResult":
        return cls(
            query=str(raw.get("query", "")),
            backend_name=str(raw.get("backend_name", "")),
            config_path=str(raw.get("config_path", "")),
            rendered_images=[
                RenderedView.from_dict(item) for item in raw.get("rendered_images", [])
            ],
            candidate_points=[
                Candidate3DPoint.from_dict(item) for item in raw.get("candidate_points", [])
            ],
            bounding_regions=[
                BoundingRegion.from_dict(item) for item in raw.get("bounding_regions", [])
            ],
            confidence=_optional_float(raw.get("confidence")),
            warnings=list(raw.get("warnings") or []),
            provenance=dict(raw.get("provenance") or {}),
        )

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def from_json(cls, path: str | Path) -> "QueryResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class SceneQueryReport:
    """Aggregated report for a natural-language scene inspection task."""

    scene_name: str
    task: str
    plan: dict[str, Any]
    query_results: list[QueryResult]
    answer: str
    warnings: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "task": self.task,
            "plan": self.plan,
            "query_results": [result.to_dict() for result in self.query_results],
            "answer": self.answer,
            "warnings": list(self.warnings),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path


class SemanticFieldBackend(ABC):
    """Abstract interface for language-queryable NeRF backends."""

    backend_name: str
    config_path: str | None

    @abstractmethod
    def load(self, config_path: str) -> None:
        """Load a trained backend config."""

    @abstractmethod
    def query_text(self, query: str, output_dir: str, top_k: int = 5) -> QueryResult:
        """Run a text query and return structured artifacts."""

    @abstractmethod
    def render_relevancy(self, query: str, output_dir: str) -> list[RenderedView]:
        """Render RGB/relevancy artifacts for a text query."""

    @abstractmethod
    def export_query_artifacts(self, result: QueryResult, output_dir: str) -> None:
        """Persist query artifacts to disk."""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_tuple(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    if len(value) != size:
        raise ValueError(f"Expected tuple of length {size}, got {value}")
    return tuple(float(item) for item in value)
