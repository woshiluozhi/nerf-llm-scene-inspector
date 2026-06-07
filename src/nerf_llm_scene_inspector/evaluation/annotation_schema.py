"""Annotation schema helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QueryAnnotation:
    """Manual annotation for one query."""

    query: str
    target_description: str = ""
    acceptable_views: list[str] = field(default_factory=list)
    bbox_2d: tuple[float, float, float, float] | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QueryAnnotation":
        bbox = raw.get("bbox_2d")
        return cls(
            query=str(raw.get("query", "")),
            target_description=str(raw.get("target_description", "")),
            acceptable_views=[str(item) for item in raw.get("acceptable_views", [])],
            bbox_2d=tuple(float(item) for item in bbox) if bbox else None,
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "target_description": self.target_description,
            "acceptable_views": self.acceptable_views,
            "bbox_2d": list(self.bbox_2d) if self.bbox_2d else None,
            "notes": self.notes,
        }


@dataclass
class AnnotationSet:
    """Scene-level query annotations."""

    scene_name: str
    queries: list[QueryAnnotation]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AnnotationSet":
        return cls(
            scene_name=str(raw.get("scene_name", "")),
            queries=[QueryAnnotation.from_dict(item) for item in raw.get("queries", [])],
        )

    def by_query(self) -> dict[str, QueryAnnotation]:
        return {annotation.query: annotation for annotation in self.queries}


def load_annotations(path: str | Path) -> AnnotationSet:
    """Load annotation JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnnotationSet.from_dict(raw)
