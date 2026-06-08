"""High-level semantic query orchestration."""

from __future__ import annotations

from pathlib import Path

from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner, Planner
from nerf_llm_scene_inspector.backends.base import QueryResult, SceneQueryReport, SemanticFieldBackend
from nerf_llm_scene_inspector.querying.spatial_reasoning import aggregate_multi_query_results
from nerf_llm_scene_inspector.utils.paths import slugify


class SemanticQueryEngine:
    """Plan a task, query a backend, and aggregate results."""

    def __init__(
        self,
        backend: SemanticFieldBackend,
        planner: Planner | None = None,
        top_k: int = 5,
        scene_name: str = "unknown",
    ) -> None:
        self.backend = backend
        self.planner = planner or LocalRulePlanner()
        self.top_k = top_k
        self.scene_name = scene_name

    def run_task(self, task: str, output_dir: str | Path) -> SceneQueryReport:
        plan = self.planner.plan(task)
        result_dir = Path(output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        results: list[QueryResult] = []
        used_slugs: dict[str, int] = {}
        for call in plan.recommended_backend_calls:
            query = str(call.get("query", ""))
            if not query:
                continue
            query_output = result_dir / _unique_query_slug(query, used_slugs)
            results.append(self.backend.query_text(query, str(query_output), top_k=self.top_k))

        aggregate = aggregate_multi_query_results(results)
        labels = [region.label for region in aggregate.bounding_regions[: self.top_k]]
        if not labels:
            labels = [result.query for result in results[: self.top_k]]
        answer = plan.final_answer_template.format(items=", ".join(labels) if labels else "pending")
        return SceneQueryReport(
            scene_name=self.scene_name,
            task=task,
            plan=plan.to_dict(),
            query_results=results,
            answer=answer,
            warnings=plan.warnings + aggregate.warnings,
        )


def _unique_query_slug(query: str, used_slugs: dict[str, int]) -> str:
    base = slugify(query)
    count = used_slugs.get(base, 0) + 1
    used_slugs[base] = count
    return base if count == 1 else f"{base}_{count}"
