"""High-level semantic query orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner, Planner
from nerf_llm_scene_inspector.backends.base import QueryResult, SceneQueryReport, SemanticFieldBackend
from nerf_llm_scene_inspector.querying.answer_synthesis import synthesize_scene_answer
from nerf_llm_scene_inspector.querying.query_types import QueryPlan
from nerf_llm_scene_inspector.querying.spatial_reasoning import aggregate_multi_query_results
from nerf_llm_scene_inspector.utils.paths import slugify


@dataclass(frozen=True)
class PlannedBackendCall:
    """One concrete backend text query derived from a high-level task."""

    query: str
    backend: str = "lerf"
    purpose: str = "primary"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "backend": self.backend,
            "purpose": self.purpose,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class SemanticQueryEngine:
    """Plan a task, query a backend, and aggregate results."""

    def __init__(
        self,
        backend: SemanticFieldBackend,
        planner: Planner | None = None,
        top_k: int = 5,
        max_queries: int = 5,
        include_negative_queries: bool = False,
        scene_name: str = "unknown",
    ) -> None:
        self.backend = backend
        self.planner = planner or LocalRulePlanner()
        self.top_k = top_k
        self.max_queries = max_queries
        self.include_negative_queries = include_negative_queries
        self.scene_name = scene_name

    def run_task(self, task: str, output_dir: str | Path, *, exact_query: bool = False) -> SceneQueryReport:
        plan = self.planner.plan(task)
        result_dir = Path(output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        results: list[QueryResult] = []
        used_slugs: dict[str, int] = {}
        calls = planned_backend_calls(
            plan,
            task=task,
            exact_query=exact_query,
            include_negative=self.include_negative_queries,
            max_queries=self.max_queries,
        )
        for call in calls:
            query = call.query
            query_output = result_dir / _unique_query_slug(query, used_slugs)
            result = self.backend.query_text(query, str(query_output), top_k=self.top_k)
            result.provenance["planner_backend_call"] = call.to_dict()
            result.to_json(query_output / "query_result.json")
            results.append(result)

        aggregate = aggregate_multi_query_results(results)
        answer = synthesize_scene_answer(
            task=task,
            plan=plan.to_dict(),
            results=results,
            top_k=self.top_k,
        )
        return SceneQueryReport(
            scene_name=self.scene_name,
            task=task,
            plan=plan.to_dict(),
            query_results=results,
            answer=answer.answer,
            answer_summary=answer.to_dict(),
            warnings=plan.warnings + aggregate.warnings,
        )


def planned_backend_calls(
    plan: QueryPlan,
    *,
    task: str,
    exact_query: bool = False,
    include_negative: bool = False,
    max_queries: int | None = 5,
) -> list[PlannedBackendCall]:
    """Return the concrete backend calls that both CLI and library execution should use."""

    if exact_query:
        return [PlannedBackendCall(query=task, purpose="exact")]

    raw_calls = plan.recommended_backend_calls or _fallback_calls_from_plan(plan)
    calls: list[PlannedBackendCall] = []
    seen_queries: set[str] = set()
    for raw_call in raw_calls:
        call = _coerce_planned_call(raw_call)
        if call is None:
            continue
        if call.purpose == "negative" and not include_negative:
            continue
        key = call.query.strip().lower()
        if not key or key in seen_queries:
            continue
        seen_queries.add(key)
        calls.append(call)
    if calls and max_queries is None:
        return calls
    if calls:
        limit = max(max_queries or 1, 1)
        if len(calls) <= limit:
            return calls
        selected = calls[:limit]
        if include_negative and not any(call.purpose == "negative" for call in selected):
            negative_call = next((call for call in calls[limit:] if call.purpose == "negative"), None)
            if negative_call is not None:
                replace_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index].purpose != "primary"
                    ),
                    len(selected) - 1,
                )
                selected[replace_index] = negative_call
        return selected
    fallback_query = task or "scene"
    return [PlannedBackendCall(query=fallback_query, purpose="fallback")]


def _fallback_calls_from_plan(plan: QueryPlan) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    calls.extend({"query": query, "purpose": "primary"} for query in plan.primary_visual_queries)
    calls.extend({"query": query, "purpose": "supporting"} for query in plan.supporting_visual_queries)
    calls.extend({"query": query, "purpose": "negative"} for query in plan.negative_visual_queries)
    return calls


def _coerce_planned_call(raw_call: Any) -> PlannedBackendCall | None:
    if not isinstance(raw_call, dict):
        query = str(raw_call or "").strip()
        return PlannedBackendCall(query=query, purpose="primary") if query else None
    query = str(raw_call.get("query") or "").strip()
    if not query:
        return None
    return PlannedBackendCall(
        query=query,
        backend=str(raw_call.get("backend") or "lerf"),
        purpose=str(raw_call.get("purpose") or "primary"),
        metadata={
            str(key): value
            for key, value in raw_call.items()
            if key not in {"query", "backend", "purpose"}
        },
    )


def _unique_query_slug(query: str, used_slugs: dict[str, int]) -> str:
    base = slugify(query)
    count = used_slugs.get(base, 0) + 1
    used_slugs[base] = count
    return base if count == 1 else f"{base}_{count}"
