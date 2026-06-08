from pathlib import Path

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult, RenderedView, SemanticFieldBackend
from nerf_llm_scene_inspector.querying.query_types import QueryPlan
from nerf_llm_scene_inspector.querying.semantic_query import SemanticQueryEngine


class FakeBackend(SemanticFieldBackend):
    backend_name = "fake"
    config_path = "fake-config.yml"

    def __init__(self) -> None:
        self.output_dirs: list[str] = []

    def load(self, config_path: str) -> None:
        self.config_path = config_path

    def query_text(self, query: str, output_dir: str, top_k: int = 5) -> QueryResult:
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
