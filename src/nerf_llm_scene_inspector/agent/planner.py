"""LLM-style scene query planners."""

from __future__ import annotations

import json
import os
from typing import Protocol

from nerf_llm_scene_inspector.agent.local_rules import AFFORDANCE_KEYWORDS, CONTAINER_QUERIES
from nerf_llm_scene_inspector.agent.prompt_templates import LOCAL_FINAL_TEMPLATE, PLANNER_SYSTEM_PROMPT
from nerf_llm_scene_inspector.querying.query_types import QueryPlan


class Planner(Protocol):
    """Planner protocol."""

    def plan(self, task: str) -> QueryPlan:
        """Return a structured plan for a user task."""


class LocalRulePlanner:
    """Deterministic no-API planner for open-vocabulary scene queries."""

    planner_name = "local_rules"

    def plan(self, task: str) -> QueryPlan:
        normalized = task.lower().strip()
        primary: list[str] = []
        supporting: list[str] = []
        relations: list[str] = []
        reasoning: list[str] = []

        for keyword, queries in AFFORDANCE_KEYWORDS.items():
            if keyword in normalized:
                primary.extend(queries[:4])
                supporting.extend(queries[4:])
                reasoning.append(f"Expand '{keyword}' into concrete visual categories.")

        if not primary:
            primary = _extract_query_terms(normalized)
            reasoning.append("Use noun-like phrases from the user task as direct visual queries.")

        if "hold" in normalized or "container" in normalized or "water" in normalized:
            primary = _merge_preserving_order(primary + CONTAINER_QUERIES[:4])
            supporting = _merge_preserving_order(supporting + CONTAINER_QUERIES[4:])
            relations.append("containment or concavity affordance")
            reasoning.append("Rank container-like detections by relevancy and spatial compactness.")

        if "support" in normalized or "on top" in normalized or "under" in normalized:
            relations.append("support/on-top-of heuristic")
            reasoning.append("Compare vertical ordering and near/far distance between candidate points.")

        if "safe" in normalized or "hot" in normalized:
            relations.append("stable flat surface affordance")
            reasoning.append("Prefer broad flat surfaces away from electronics and clutter.")

        if "left" in normalized or "right" in normalized or "above" in normalized or "below" in normalized:
            relations.append("image-space or camera-space directional relation")

        primary = _merge_preserving_order(primary)
        supporting = [item for item in _merge_preserving_order(supporting) if item not in primary]
        backend_calls = [
            {"backend": "lerf", "query": query, "top_k": 5, "purpose": "primary"}
            for query in primary
        ]
        backend_calls.extend(
            {"backend": "lerf", "query": query, "top_k": 5, "purpose": "supporting"}
            for query in supporting
        )

        return QueryPlan(
            task=task,
            primary_visual_queries=primary,
            supporting_visual_queries=supporting,
            relation_hypotheses=relations or ["semantic relevancy ranking"],
            recommended_backend_calls=backend_calls,
            final_answer_template=LOCAL_FINAL_TEMPLATE,
            planner_name=self.planner_name,
            warnings=[] if primary else ["No visual queries were extracted."],
        )


class LLMPlanner:
    """Optional API-backed planner with local-rule fallback."""

    planner_name = "optional_llm"

    def __init__(self, api_key_env: str = "OPENAI_API_KEY", model_env: str = "OPENAI_MODEL") -> None:
        self.api_key = os.environ.get(api_key_env)
        self.model = os.environ.get(model_env, "gpt-4.1-mini")
        self.local = LocalRulePlanner()

    def plan(self, task: str) -> QueryPlan:
        if not self.api_key:
            plan = self.local.plan(task)
            plan.warnings.append("OPENAI_API_KEY not set; used LocalRulePlanner.")
            return plan

        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError:
            plan = self.local.plan(task)
            plan.warnings.append("openai package not installed; used LocalRulePlanner.")
            return plan

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ],
                text={"format": {"type": "json_object"}},
            )
            raw_text = response.output_text
            raw = json.loads(raw_text)
            primary = list(raw.get("primary_visual_queries") or [])
            supporting = list(raw.get("supporting_visual_queries") or [])
            if not primary:
                return self.local.plan(task)
            backend_calls = list(raw.get("recommended_backend_calls") or [])
            if not backend_calls:
                backend_calls = [
                    {"backend": "lerf", "query": query, "top_k": 5, "purpose": "primary"}
                    for query in primary
                ]
            return QueryPlan(
                task=task,
                primary_visual_queries=[str(item) for item in primary],
                supporting_visual_queries=[str(item) for item in supporting],
                relation_hypotheses=[str(item) for item in raw.get("relation_hypotheses", [])],
                recommended_backend_calls=backend_calls,
                final_answer_template=str(
                    raw.get("final_answer_template")
                    or "Likely relevant scene regions are {items}."
                ),
                planner_name=self.planner_name,
            )
        except Exception as exc:  # pragma: no cover - depends on external API behavior
            plan = self.local.plan(task)
            plan.warnings.append(f"LLM planning failed; used LocalRulePlanner: {exc}")
            return plan


def get_default_planner(prefer_llm: bool = False) -> Planner:
    """Return the default planner."""

    return LLMPlanner() if prefer_llm else LocalRulePlanner()


def _extract_query_terms(text: str) -> list[str]:
    stopwords = {
        "find",
        "locate",
        "where",
        "which",
        "objects",
        "object",
        "that",
        "the",
        "are",
        "is",
        "likely",
        "related",
        "to",
        "on",
        "in",
        "of",
        "a",
        "an",
    }
    words = [word.strip(" .,?!;:") for word in text.split()]
    terms = [word for word in words if word and word not in stopwords]
    if not terms:
        return [text]
    if len(terms) >= 2:
        return [" ".join(terms[:3]), terms[0]]
    return terms


def _merge_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return merged
