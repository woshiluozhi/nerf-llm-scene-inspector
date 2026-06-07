"""Prompt templates for the optional LLM planner."""

PLANNER_SYSTEM_PROMPT = """You convert user scene-inspection tasks into visual queries.
Return compact JSON with primary_visual_queries, supporting_visual_queries,
negative_visual_queries, relation_hypotheses, recommended_backend_calls,
rationale, confidence, and final_answer_template.
Do not claim objects are present until backend evidence is available."""

LOCAL_FINAL_TEMPLATE = (
    "Likely relevant objects or regions are {items}. Rank them by semantic relevancy, "
    "spatial compactness, and consistency across rendered views."
)
