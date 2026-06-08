from nerf_llm_scene_inspector.agent.planner import LLMPlanner, LocalRulePlanner


def test_local_planner_object_search() -> None:
    plan = LocalRulePlanner().plan("find the mug")
    assert plan.primary_visual_queries[0] == "mug"
    assert "object_search" in plan.intent_tags
    assert plan.confidence is not None


def test_local_planner_laptop_search_stays_targeted() -> None:
    plan = LocalRulePlanner().plan("locate the laptop")
    assert plan.primary_visual_queries == ["laptop"]
    assert plan.supporting_visual_queries == []
    assert plan.recommended_backend_calls[0]["purpose"] == "primary"
    assert plan.recommended_backend_calls[0]["intent_tags"] == ["object_search"]


def test_local_planner_expands_water_container_task() -> None:
    plan = LocalRulePlanner().plan("Find objects that could hold water.")
    assert "cup" in plan.primary_visual_queries
    assert "bottle" in plan.primary_visual_queries
    assert any("containment" in item for item in plan.relation_hypotheses)
    assert plan.recommended_backend_calls


def test_local_planner_safe_hot_cup_has_negative_queries() -> None:
    plan = LocalRulePlanner().plan("safe place to put a hot cup")
    assert "flat surface" in plan.supporting_visual_queries or "flat surface" in plan.primary_visual_queries
    assert "electronics" in plan.negative_visual_queries
    assert any("stable" in item for item in plan.relation_hypotheses)


def test_local_planner_support_task() -> None:
    plan = LocalRulePlanner().plan("Find the object that supports the laptop.")
    assert "laptop" in plan.primary_visual_queries
    assert any("support" in item for item in plan.relation_hypotheses)
    assert plan.relation_anchors[0]["anchor_query"] == "laptop"
    assert plan.recommended_backend_calls[0]["purpose"] == "relation_anchor"


def test_local_planner_material_search() -> None:
    plan = LocalRulePlanner().plan("locate metallic tools on the desk")
    assert plan.primary_visual_queries[0] == "metallic tool"
    assert any("metal" in query for query in plan.primary_visual_queries)
    assert "desk" not in plan.primary_visual_queries[:2]
    assert "affordance:metal" not in plan.intent_tags
    assert "material:metal" not in plan.intent_tags


def test_local_planner_cutting_tools() -> None:
    plan = LocalRulePlanner().plan("tools for cutting")
    assert "scissors" in plan.primary_visual_queries
    assert "knife" in plan.primary_visual_queries


def test_local_planner_writing_objects() -> None:
    plan = LocalRulePlanner().plan("objects useful for writing")
    assert "pen" in plan.primary_visual_queries
    assert "notebook" in plan.supporting_visual_queries or "notebook" in plan.primary_visual_queries


def test_local_planner_spatial_next_to() -> None:
    plan = LocalRulePlanner().plan("object next to the mug")
    assert any("near" in item for item in plan.relation_hypotheses)
    assert plan.primary_visual_queries == ["mug"]
    assert plan.relation_anchors == [
        {
            "relation": "near relation",
            "anchor_query": "mug",
            "candidate_query": "nearby object",
            "context_query": "object near mug",
            "evidence_frame": "2d_or_3d",
        }
    ]
    assert plan.recommended_backend_calls[0]["purpose"] == "relation_anchor"
    assert plan.recommended_backend_calls[0]["candidate_query"] == "nearby object"


def test_local_planner_directional_relation_tracks_anchor_frame() -> None:
    plan = LocalRulePlanner().plan("objects on the left side of the desk")
    assert plan.relation_anchors[0]["anchor_query"] == "desk"
    assert plan.relation_anchors[0]["evidence_frame"] == "image_or_camera"
    assert "objects around desk" in plan.supporting_visual_queries


def test_local_planner_scene_level_fragile() -> None:
    plan = LocalRulePlanner().plan("fragile objects")
    assert "glass" in plan.primary_visual_queries
    assert plan.rationale


def test_llm_planner_falls_back_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan = LLMPlanner().plan("Which objects are likely containers?")
    assert plan.primary_visual_queries
    assert any("LocalRulePlanner" in warning for warning in plan.warnings)
