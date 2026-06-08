import json
from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult, RenderedView, SemanticFieldBackend
from nerf_llm_scene_inspector.querying.query_types import QueryPlan
from nerf_llm_scene_inspector.querying.semantic_query import SemanticQueryEngine, planned_backend_calls


class FakeBackend(SemanticFieldBackend):
    backend_name = "fake"
    config_path = "fake-config.yml"

    def __init__(self) -> None:
        self.output_dirs: list[str] = []
        self.queries: list[str] = []

    def load(self, config_path: str) -> None:
        self.config_path = config_path

    def query_text(self, query: str, output_dir: str, top_k: int = 5) -> QueryResult:
        self.queries.append(query)
        self.output_dirs.append(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return QueryResult(
            query=query,
            backend_name=self.backend_name,
            config_path=self.config_path or "",
            rendered_images=[RenderedView(path=str(Path(output_dir) / "overlay.png"), kind="overlay", query=query)],
            bounding_regions=[BoundingRegion(label=query, score=0.8, bbox_2d=(1.0, 2.0, 10.0, 20.0))],
            confidence=0.75,
        )

    def render_relevancy(self, query: str, output_dir: str) -> list[RenderedView]:
        return []

    def export_query_artifacts(self, result: QueryResult, output_dir: str) -> None:
        return None


class DuplicateSlugPlanner:
    planner_name = "duplicate_slug"

    def plan(self, task: str) -> QueryPlan:
        return QueryPlan(
            task=task,
            primary_visual_queries=["Coffee mug!", "coffee mug"],
            supporting_visual_queries=[],
            negative_visual_queries=[],
            relation_hypotheses=[],
            recommended_backend_calls=[
                {"backend": "fake", "query": "Coffee mug!", "top_k": 5, "purpose": "primary"},
                {"backend": "fake", "query": "coffee mug", "top_k": 5, "purpose": "primary"},
            ],
            final_answer_template="Likely items: {items}.",
            planner_name=self.planner_name,
        )


class MixedPurposePlanner:
    planner_name = "mixed"

    def plan(self, task: str) -> QueryPlan:
        return QueryPlan(
            task=task,
            primary_visual_queries=["mug"],
            supporting_visual_queries=["cup"],
            negative_visual_queries=["screen"],
            relation_hypotheses=[],
            recommended_backend_calls=[
                {"backend": "fake", "query": "mug", "top_k": 5, "purpose": "primary"},
                {"backend": "fake", "query": "cup", "top_k": 5, "purpose": "supporting"},
                {"backend": "fake", "query": "screen", "top_k": 5, "purpose": "negative"},
            ],
            final_answer_template="Likely items: {items}.",
            planner_name=self.planner_name,
        )


def test_semantic_query_engine_records_scene_name_and_unique_query_slugs(tmp_path: Path) -> None:
    backend = FakeBackend()
    engine = SemanticQueryEngine(
        backend=backend,
        planner=DuplicateSlugPlanner(),
        scene_name="desk_scene",
    )

    report = engine.run_task("Find coffee mugs", tmp_path)

    assert report.scene_name == "desk_scene"
    assert [Path(path).name for path in backend.output_dirs] == ["coffee_mug", "coffee_mug_2"]
    assert (tmp_path / "coffee_mug").exists()
    assert (tmp_path / "coffee_mug_2").exists()
    assert report.answer_summary["support_level"] == "2d_relevancy_fallback"
    assert report.answer_summary["evidence"][0]["label"] in {"Coffee mug!", "coffee mug"}
    assert "Strongest evidence" in report.answer


def test_planned_backend_calls_exclude_negative_queries_by_default() -> None:
    plan = MixedPurposePlanner().plan("Find a cup, not a screen")

    calls = planned_backend_calls(plan, task=plan.task)

    assert [call.query for call in calls] == ["mug", "cup"]
    assert [call.purpose for call in calls] == ["primary", "supporting"]


def test_planned_backend_calls_tolerate_string_calls() -> None:
    plan = QueryPlan(
        task="Find writing tools",
        primary_visual_queries=[],
        recommended_backend_calls=["pen", {"query": "notebook", "purpose": "supporting"}],
    )

    calls = planned_backend_calls(plan, task=plan.task)

    assert [call.to_dict() for call in calls] == [
        {"query": "pen", "backend": "lerf", "purpose": "primary"},
        {"query": "notebook", "backend": "lerf", "purpose": "supporting"},
    ]


def test_planned_backend_calls_preserve_planner_metadata() -> None:
    plan = QueryPlan(
        task="object next to the mug",
        primary_visual_queries=["mug"],
        recommended_backend_calls=[
            {
                "backend": "fake",
                "query": "mug",
                "purpose": "relation_anchor",
                "top_k": 5,
                "relation": "near relation",
                "candidate_query": "nearby object",
            }
        ],
    )

    calls = planned_backend_calls(plan, task=plan.task)

    assert calls[0].to_dict() == {
        "query": "mug",
        "backend": "fake",
        "purpose": "relation_anchor",
        "metadata": {
            "top_k": 5,
            "relation": "near relation",
            "candidate_query": "nearby object",
        },
    }


def test_semantic_query_engine_can_include_negative_queries(tmp_path: Path) -> None:
    backend = FakeBackend()
    engine = SemanticQueryEngine(
        backend=backend,
        planner=MixedPurposePlanner(),
        include_negative_queries=True,
        scene_name="desk_scene",
    )

    report = engine.run_task("Find a cup, not a screen", tmp_path)

    assert backend.queries == ["mug", "cup", "screen"]
    assert len(report.query_results) == 3
    assert report.query_results[-1].provenance["planner_backend_call"]["purpose"] == "negative"
    persisted = json.loads((tmp_path / "screen" / "query_result.json").read_text(encoding="utf-8"))
    assert persisted["provenance"]["planner_backend_call"]["purpose"] == "negative"
    evidence_labels = [item["label"] for item in report.answer_summary["evidence"]]
    assert "screen" not in evidence_labels
    assert any(
        "Negative/disambiguation query results" in item
        for item in report.answer_summary["limitations"]
    )


def test_semantic_query_engine_exact_query_bypasses_planner_expansion(tmp_path: Path) -> None:
    backend = FakeBackend()
    engine = SemanticQueryEngine(
        backend=backend,
        planner=MixedPurposePlanner(),
        scene_name="desk_scene",
    )

    report = engine.run_task("Find a cup, not a screen", tmp_path, exact_query=True)

    assert backend.queries == ["Find a cup, not a screen"]
    assert report.query_results[0].query == "Find a cup, not a screen"


def test_scene_query_report_writes_markdown(tmp_path: Path) -> None:
    backend = FakeBackend()
    engine = SemanticQueryEngine(
        backend=backend,
        planner=DuplicateSlugPlanner(),
        scene_name="desk_scene",
    )

    report = engine.run_task("Find coffee mugs", tmp_path)
    output = report.to_markdown(tmp_path / "scene_query_report.md")

    markdown = output.read_text(encoding="utf-8")
    assert "# Scene Query Report" in markdown
    assert "## Answer Evidence" in markdown
    assert "Coffee mug!" in markdown


def test_scene_query_report_marks_query_purpose_in_markdown(tmp_path: Path) -> None:
    backend = FakeBackend()
    engine = SemanticQueryEngine(
        backend=backend,
        planner=MixedPurposePlanner(),
        include_negative_queries=True,
        scene_name="desk_scene",
    )

    report = engine.run_task("Find a cup, not a screen", tmp_path)
    output = report.to_markdown(tmp_path / "scene_query_report.md")

    markdown = output.read_text(encoding="utf-8")
    assert "- Negative/disambiguation visual queries: screen" in markdown
    assert "`screen` via `fake` purpose=`negative`, top_k=5" in markdown
    assert "`screen` via `fake` (purpose=`negative`, planned_backend=`fake`)" in markdown
    assert "excluded from positive answer evidence" in markdown


def test_semantic_query_engine_persists_relation_call_metadata(tmp_path: Path) -> None:
    class RelationPlanner:
        planner_name = "relation"

        def plan(self, task: str) -> QueryPlan:
            return QueryPlan(
                task=task,
                primary_visual_queries=["mug"],
                relation_hypotheses=["near relation"],
                recommended_backend_calls=[
                    {
                        "backend": "fake",
                        "query": "mug",
                        "purpose": "relation_anchor",
                        "relation": "near relation",
                        "candidate_query": "nearby object",
                    }
                ],
                planner_name=self.planner_name,
            )

    backend = FakeBackend()
    engine = SemanticQueryEngine(backend=backend, planner=RelationPlanner(), scene_name="desk_scene")

    report = engine.run_task("object next to the mug", tmp_path)

    call = report.query_results[0].provenance["planner_backend_call"]
    assert call["purpose"] == "relation_anchor"
    assert call["metadata"]["relation"] == "near relation"
    persisted = json.loads((tmp_path / "mug" / "query_result.json").read_text(encoding="utf-8"))
    assert persisted["provenance"]["planner_backend_call"]["metadata"]["candidate_query"] == "nearby object"
