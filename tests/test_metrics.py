from nerf_llm_scene_inspector.backends.base import BoundingRegion
from nerf_llm_scene_inspector.evaluation.metrics import bbox_iou, topk_localization_hit


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    value = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert round(value, 3) == 0.143


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
