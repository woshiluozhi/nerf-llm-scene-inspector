from nerf_llm_scene_inspector.agent.planner import LLMPlanner, LocalRulePlanner


def test_local_planner_expands_water_container_task() -> None:
    plan = LocalRulePlanner().plan("Find objects that could hold water.")
    assert "cup" in plan.primary_visual_queries
    assert "bottle" in plan.primary_visual_queries
    assert any("containment" in item for item in plan.relation_hypotheses)
    assert plan.recommended_backend_calls


def test_local_planner_support_task() -> None:
    plan = LocalRulePlanner().plan("Find the object that supports the laptop.")
    assert "laptop" in plan.primary_visual_queries
    assert any("support" in item for item in plan.relation_hypotheses)


def test_llm_planner_falls_back_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan = LLMPlanner().plan("Which objects are likely containers?")
    assert plan.primary_visual_queries
    assert any("LocalRulePlanner" in warning for warning in plan.warnings)
