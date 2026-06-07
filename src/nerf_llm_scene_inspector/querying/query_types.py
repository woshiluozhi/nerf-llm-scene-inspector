"""Typed planning structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    """Structured output from a natural-language query planner."""

    task: str
    primary_visual_queries: list[str]
    supporting_visual_queries: list[str] = field(default_factory=list)
    negative_visual_queries: list[str] = field(default_factory=list)
    relation_hypotheses: list[str] = field(default_factory=list)
    recommended_backend_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer_template: str = "Likely relevant scene regions are ..."
    planner_name: str = "local_rules"
    rationale: list[str] = field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
