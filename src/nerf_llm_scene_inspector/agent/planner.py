"""LLM-style scene query planners."""

from __future__ import annotations

import json
import os
from typing import Protocol

from nerf_llm_scene_inspector.agent.local_rules import (
    AFFORDANCE_KEYWORDS,
    CONTAINER_QUERIES,
    MATERIAL_QUERIES,
    NEGATIVE_QUERY_HINTS,
    OBJECT_QUERY_ALIASES,
    SCENE_SEMANTIC_QUERIES,
    SPATIAL_KEYWORDS,
)
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
        negative: list[str] = []
        relations: list[str] = []
        relation_anchors: list[dict[str, str]] = []
        reasoning: list[str] = []
        intent_tags: list[str] = []
        confidence = 0.55
        known_objects = _extract_known_objects(normalized)

        if _looks_like_object_search(normalized) and known_objects and not _has_rule_intent(normalized):
            primary.extend(known_objects[:3])
            intent_tags.append("object_search")
            reasoning.append("Extract concrete target objects before broader affordance expansion.")
            confidence = max(confidence, 0.70)

        for keyword, queries in AFFORDANCE_KEYWORDS.items():
            if _keyword_matches(keyword, normalized):
                primary.extend(queries[:4])
                supporting.extend(queries[4:])
                reasoning.append(f"Expand '{keyword}' into concrete visual categories.")
                intent_tags.append(f"affordance:{keyword}")
                confidence = max(confidence, 0.75)

        for keyword, queries in MATERIAL_QUERIES.items():
            if _keyword_matches(keyword, normalized):
                primary.extend(queries[:4])
                reasoning.append(f"Use material prompt expansion for '{keyword}'.")
                intent_tags.append(f"material:{keyword}")
                confidence = max(confidence, 0.72)

        for keyword, queries in SCENE_SEMANTIC_QUERIES.items():
            if _keyword_matches(keyword, normalized):
                primary.extend(queries[:4])
                supporting.extend(queries[4:])
                reasoning.append(f"Expand scene-level semantic category '{keyword}'.")
                intent_tags.append(f"scene_semantic:{keyword}")
                confidence = max(confidence, 0.78)

        for keyword, (relation, queries) in SPATIAL_KEYWORDS.items():
            if _keyword_matches(keyword, normalized):
                relations.append(relation)
                primary.extend(queries[:3])
                supporting.extend(queries[3:])
                reasoning.append(f"Add spatial relation hypothesis for '{keyword}'.")
                intent_tags.append(f"spatial:{keyword}")
                confidence = max(confidence, 0.70)

        for keyword, queries in NEGATIVE_QUERY_HINTS.items():
            if _keyword_matches(keyword, normalized):
                negative.extend(queries)
                reasoning.append(f"Track disambiguation/avoidance prompts for '{keyword}'.")
                intent_tags.append(f"negative_hint:{keyword}")

        relation_anchors.extend(_extract_relation_anchors(normalized, known_objects))
        if relation_anchors:
            for anchor in relation_anchors:
                anchor_query = anchor["anchor_query"]
                if anchor_query not in primary:
                    primary.append(anchor_query)
                context_query = anchor.get("context_query")
                if context_query:
                    supporting.append(context_query)
            reasoning.append("Separate relation anchors from candidate object queries for spatial reasoning.")

        if not primary:
            primary = _extract_query_terms(normalized)
            reasoning.append("Use noun-like phrases from the user task as direct visual queries.")

        if "hold" in normalized or "container" in normalized or "water" in normalized:
            primary = _merge_preserving_order(primary + CONTAINER_QUERIES[:4])
            supporting = _merge_preserving_order(supporting + CONTAINER_QUERIES[4:])
            relations.append("containment or concavity affordance")
            intent_tags.append("affordance:containment")
            reasoning.append("Rank container-like detections by relevancy and spatial compactness.")

        if "support" in normalized or "on top" in normalized or "under" in normalized:
            relations.append("support/on-top-of heuristic")
            intent_tags.append("relation:support")
            reasoning.append("Compare vertical ordering and near/far distance between candidate points.")

        if "safe" in normalized or "hot" in normalized:
            relations.append("stable flat surface affordance")
            intent_tags.append("affordance:stable_surface")
            reasoning.append("Prefer broad flat surfaces away from electronics and clutter.")

        if "left" in normalized or "right" in normalized or "above" in normalized or "below" in normalized:
            relations.append("image-space or camera-space directional relation")
            intent_tags.append("relation:directional")

        primary = _merge_preserving_order(primary)
        supporting = [item for item in _merge_preserving_order(supporting) if item not in primary]
        negative = [item for item in _merge_preserving_order(negative) if item not in primary]
        intent_tags = _merge_preserving_order(intent_tags)
        relations = _merge_preserving_order(relations)
        backend_calls = _build_backend_calls(
            primary=primary,
            supporting=supporting,
            negative=negative,
            intent_tags=intent_tags,
            relation_anchors=relation_anchors,
        )

        return QueryPlan(
            task=task,
            primary_visual_queries=primary,
            supporting_visual_queries=supporting,
            negative_visual_queries=negative,
            relation_hypotheses=relations or ["semantic relevancy ranking"],
            recommended_backend_calls=backend_calls,
            final_answer_template=LOCAL_FINAL_TEMPLATE,
            planner_name=self.planner_name,
            rationale=reasoning,
            confidence=confidence if primary else 0.2,
            intent_tags=intent_tags,
            relation_anchors=relation_anchors,
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
                negative_visual_queries=[str(item) for item in raw.get("negative_visual_queries", [])],
                relation_hypotheses=[str(item) for item in raw.get("relation_hypotheses", [])],
                recommended_backend_calls=backend_calls,
                final_answer_template=str(
                    raw.get("final_answer_template")
                    or "Likely relevant scene regions are {items}."
                ),
                planner_name=self.planner_name,
                rationale=[str(item) for item in raw.get("rationale", [])],
                confidence=float(raw["confidence"]) if raw.get("confidence") is not None else None,
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
        "could",
        "can",
        "please",
        "show",
        "me",
        "side",
    }
    words = [word.strip(" .,?!;:") for word in text.split()]
    terms = [word for word in words if word and word not in stopwords]
    if not terms:
        return [text]
    if len(terms) >= 2:
        return [" ".join(terms[:3]), terms[0]]
    return terms


def _looks_like_object_search(text: str) -> bool:
    object_search_prefixes = (
        "find ",
        "locate ",
        "where is ",
        "where are ",
        "show me ",
        "identify ",
    )
    return text.startswith(object_search_prefixes)


def _has_rule_intent(text: str) -> bool:
    rule_tables = (
        AFFORDANCE_KEYWORDS,
        MATERIAL_QUERIES,
        SCENE_SEMANTIC_QUERIES,
        SPATIAL_KEYWORDS,
        NEGATIVE_QUERY_HINTS,
    )
    return any(_keyword_matches(keyword, text) for table in rule_tables for keyword in table)


def _keyword_matches(keyword: str, text: str) -> bool:
    if " " in keyword or "-" in keyword:
        return keyword in text
    cleaned = "".join(character if character.isalnum() else " " for character in text.replace("-", " "))
    return f" {keyword} " in f" {cleaned} "


def _extract_known_objects(text: str) -> list[str]:
    found: list[str] = []
    cleaned = "".join(character if character.isalnum() else " " for character in text.replace("-", " "))
    padded = f" {cleaned} "
    for token, query in OBJECT_QUERY_ALIASES.items():
        if f" {token} " in padded:
            found.append(query)
    return _merge_preserving_order(found)


def _extract_relation_anchors(text: str, known_objects: list[str]) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    anchor = _best_anchor_object(text, known_objects)
    if not anchor:
        return anchors
    if "next to" in text or "beside" in text or "near " in text:
        anchors.append(
            {
                "relation": "near relation",
                "anchor_query": anchor,
                "candidate_query": "nearby object",
                "context_query": f"object near {anchor}",
                "evidence_frame": "2d_or_3d",
            }
        )
    if "support" in text or "on top" in text or "under" in text:
        anchors.append(
            {
                "relation": "support/on-top-of heuristic",
                "anchor_query": anchor,
                "candidate_query": "supporting surface",
                "context_query": f"surface supporting {anchor}",
                "evidence_frame": "2d_or_3d",
            }
        )
    if "left" in text or "right" in text or "above" in text or "below" in text:
        anchors.append(
            {
                "relation": "directional relation",
                "anchor_query": anchor,
                "candidate_query": "directional region",
                "context_query": f"objects around {anchor}",
                "evidence_frame": "image_or_camera",
            }
        )
    return anchors


def _best_anchor_object(text: str, known_objects: list[str]) -> str | None:
    if not known_objects:
        return None
    for marker in ("next to", "beside", "near", "support", "supports", "supporting", "under", "on top"):
        if marker not in text:
            continue
        marker_index = text.find(marker)
        ranked = sorted(
            known_objects,
            key=lambda item: abs(text.find(item) - marker_index) if item in text else len(text),
        )
        return ranked[0]
    return known_objects[-1]


def _build_backend_calls(
    *,
    primary: list[str],
    supporting: list[str],
    negative: list[str],
    intent_tags: list[str],
    relation_anchors: list[dict[str, str]],
) -> list[dict[str, object]]:
    anchor_by_query = {anchor["anchor_query"]: anchor for anchor in relation_anchors}
    calls: list[dict[str, object]] = []
    for query in primary:
        call = _planned_call(query, "primary", intent_tags)
        if query in anchor_by_query:
            anchor = anchor_by_query[query]
            call.update(
                {
                    "purpose": "relation_anchor",
                    "relation": anchor["relation"],
                    "candidate_query": anchor["candidate_query"],
                    "evidence_frame": anchor["evidence_frame"],
                }
            )
        calls.append(call)
    calls.extend(_planned_call(query, "supporting", intent_tags) for query in supporting)
    calls.extend(_planned_call(query, "negative", intent_tags) for query in negative)
    return calls


def _planned_call(query: str, purpose: str, intent_tags: list[str]) -> dict[str, object]:
    call: dict[str, object] = {"backend": "lerf", "query": query, "top_k": 5, "purpose": purpose}
    if intent_tags:
        call["intent_tags"] = list(intent_tags)
    return call


def _merge_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return merged
