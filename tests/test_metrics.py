from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult
from nerf_llm_scene_inspector.evaluation.metrics import (
    bbox_area,
    bbox_center_distance,
    bbox_intersection_area,
    bbox_iou,
    containment_ratio,
    qualitative_success_table,
    topk_localization_hit,
)
from nerf_llm_scene_inspector.querying.spatial_reasoning import (
    aggregate_same_label_regions,
    containment_relation,
    image_space_relation,
    rank_by_bbox_compactness,
)


def test_bbox_iou() -> None:
    assert bbox_area((0, 0, 10, 10)) == 100
    assert bbox_intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    value = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert round(value, 3) == 0.143
    assert bbox_center_distance((0, 0, 10, 10), (10, 0, 20, 10)) == 10
    assert containment_ratio((2, 2, 4, 4), (0, 0, 10, 10)) == 1.0


def test_topk_localization_hit() -> None:
    regions = [
        BoundingRegion(label="mug", score=0.9, bbox_2d=(5, 5, 15, 15), source_view="view_0001.png")
    ]
    assert topk_localization_hit(
        regions,
        (0, 0, 20, 20),
        k=1,
        iou_threshold=0.2,
        acceptable_views=["view_0001.png"],
    )
    assert not topk_localization_hit(regions, (100, 100, 120, 120), k=1)


def test_region_ranking_and_relations() -> None:
    large = BoundingRegion(label="mug", score=0.8, bbox_2d=(0, 0, 100, 100), source_view="v")
    compact = BoundingRegion(label="mug", score=0.7, bbox_2d=(10, 10, 20, 20), source_view="v")
    ranked = rank_by_bbox_compactness([large, compact])
    assert ranked[0] == compact

    merged = aggregate_same_label_regions([large, compact])
    assert len(merged) == 1
    assert merged[0].bbox_2d == (0, 0, 100, 100)

    relation = image_space_relation(compact, large)
    assert relation.evidence_type == "2d_fallback"
    contained = containment_relation(compact, large)
    assert contained is not None
    assert contained.relation == "likely-contained-in"


def test_qualitative_success_table_separates_unannotated_queries() -> None:
    results = [
        QueryResult(
            query="mug",
            backend_name="dry-run",
            config_path="config.yml",
            bounding_regions=[
                BoundingRegion(label="mug", score=0.9, bbox_2d=(0, 0, 20, 20), source_view="view_0000")
            ],
            confidence=0.9,
        ),
        QueryResult(
            query="container",
            backend_name="dry-run",
            config_path="config.yml",
            bounding_regions=[
                BoundingRegion(label="container", score=0.5, bbox_2d=(40, 40, 80, 80), source_view="view_0000")
            ],
            confidence=0.5,
        ),
    ]
    rows = qualitative_success_table(
        results,
        {
            "mug": {
                "target_description": "white mug",
                "acceptable_views": ["view_0000"],
                "bbox_2d": [0, 0, 25, 25],
            }
        },
    )

    assert rows[0]["evaluation_status"] == "evaluated"
    assert rows[0]["topk_hit"] is True
    assert rows[0]["has_bbox_annotation"] is True
    assert rows[1]["evaluation_status"] == "unannotated"
    assert rows[1]["topk_hit"] == ""
    assert rows[1]["best_iou_2d"] == ""
