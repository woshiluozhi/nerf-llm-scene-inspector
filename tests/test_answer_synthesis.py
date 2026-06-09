from nerf_llm_scene_inspector.backends.base import (
    BoundingRegion,
    Candidate3DPoint,
    QueryResult,
    RenderedView,
)
from nerf_llm_scene_inspector.querying.answer_synthesis import synthesize_scene_answer


def test_synthesize_scene_answer_ranks_region_evidence() -> None:
    plan = {
        "final_answer_template": "Likely relevant scene regions are {items}.",
        "relation_hypotheses": ["containment or concavity affordance"],
        "confidence": 0.7,
    }
    results = [
        QueryResult(
            query="mug",
            backend_name="lerf",
            config_path="config.yml",
            rendered_images=[RenderedView(path="mug_overlay.png", kind="overlay", query="mug", camera_id="view_0")],
            bounding_regions=[
                BoundingRegion(
                    label="mug",
                    score=0.91,
                    bbox_2d=(10.0, 20.0, 80.0, 120.0),
                    source_view="view_0",
                )
            ],
            confidence=0.88,
        ),
        QueryResult(
            query="bottle",
            backend_name="lerf",
            config_path="config.yml",
            bounding_regions=[BoundingRegion(label="bottle", score=0.61)],
            confidence=0.6,
        ),
    ]

    answer = synthesize_scene_answer(task="Find objects that can hold water", plan=plan, results=results, top_k=2)

    assert answer.support_level == "2d_relevancy_fallback"
    assert answer.evidence[0].label == "mug"
    assert answer.evidence[0].rendered_artifacts == ["mug_overlay.png"]
    assert answer.confidence == 0.76
    assert "mug, bottle" in answer.answer
    assert answer.recommended_followups


def test_synthesize_scene_answer_handles_no_backend_results() -> None:
    answer = synthesize_scene_answer(
        task="Where is the mug?",
        plan={"final_answer_template": "Likely relevant scene regions are {items}."},
        results=[],
    )

    assert answer.support_level == "no_backend_evidence"
    assert answer.evidence == []
    assert answer.confidence is None
    assert "No backend evidence" in answer.answer
    assert "No backend query results were produced." in answer.limitations


def test_synthesize_scene_answer_uses_candidate_3d_points() -> None:
    result = QueryResult(
        query="mug",
        backend_name="lerf",
        config_path="config.yml",
        candidate_points=[
            Candidate3DPoint(
                label="mug handle",
                x=0.1,
                y=0.2,
                z=0.3,
                score=0.93,
                source_view="view_0002",
                metadata={"notes": "Back-projected heatmap centroid."},
            )
        ],
    )

    answer = synthesize_scene_answer(
        task="Where is the mug?",
        plan={"final_answer_template": "Likely relevant scene regions are {items}."},
        results=[result],
    )

    assert answer.support_level == "3d_candidate_points"
    assert answer.evidence[0].evidence_type == "3d_point"
    assert answer.evidence[0].point_3d == (0.1, 0.2, 0.3)
    assert answer.evidence[0].notes == "Back-projected heatmap centroid."


def test_synthesize_scene_answer_excludes_negative_query_evidence() -> None:
    plan = {
        "final_answer_template": "Likely relevant scene regions are {items}.",
        "negative_visual_queries": ["screen"],
    }
    positive = QueryResult(
        query="cup",
        backend_name="lerf",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(
                label="cup",
                score=0.7,
                bbox_2d=(10.0, 10.0, 80.0, 90.0),
                source_view="view_0001",
            )
        ],
        confidence=0.7,
        provenance={"planner_backend_call": {"query": "cup", "purpose": "primary"}},
    )
    negative = QueryResult(
        query="screen",
        backend_name="lerf",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(
                label="screen",
                score=0.99,
                bbox_2d=(20.0, 20.0, 90.0, 100.0),
                source_view="view_0001",
            )
        ],
        confidence=0.99,
        provenance={"planner_backend_call": {"query": "screen", "purpose": "negative"}},
    )

    answer = synthesize_scene_answer(
        task="Find something that can hold water, not a screen",
        plan=plan,
        results=[negative, positive],
    )

    assert [item.label for item in answer.evidence] == ["cup"]
    assert [item.label for item in answer.counter_evidence] == ["screen"]
    assert answer.risk_flags
    assert "Counter-evidence/avoid prompts detected: screen." in answer.answer
    assert answer.confidence == 0.5
    assert any("Negative/disambiguation query results" in item for item in answer.limitations)
    assert any("counter_evidence" in item for item in answer.limitations)
    assert any("spatial conflicts" in item for item in answer.limitations)


def test_synthesize_scene_answer_keeps_non_overlapping_counter_evidence_separate() -> None:
    plan = {
        "final_answer_template": "Likely safe regions are {items}.",
        "negative_visual_queries": ["electronics"],
    }
    positive = QueryResult(
        query="flat surface",
        backend_name="lerf",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(
                label="desk",
                score=0.8,
                bbox_2d=(10.0, 10.0, 60.0, 60.0),
                source_view="view_0001",
            )
        ],
        provenance={"planner_backend_call": {"query": "flat surface", "purpose": "primary"}},
    )
    negative = QueryResult(
        query="electronics",
        backend_name="lerf",
        config_path="config.yml",
        bounding_regions=[
            BoundingRegion(
                label="laptop",
                score=0.9,
                bbox_2d=(100.0, 100.0, 160.0, 160.0),
                source_view="view_0001",
            )
        ],
        provenance={"planner_backend_call": {"query": "electronics", "purpose": "negative"}},
    )

    answer = synthesize_scene_answer(
        task="Where is the safest place to put a hot cup?",
        plan=plan,
        results=[positive, negative],
    )

    assert [item.label for item in answer.evidence] == ["desk"]
    assert [item.label for item in answer.counter_evidence] == ["laptop"]
    assert answer.risk_flags == []
    assert answer.confidence == 0.65
    assert any("counter-evidence overlays" in item for item in answer.recommended_followups)
