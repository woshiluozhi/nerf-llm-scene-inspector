"""Local rule tables for deterministic planning."""

COFFEE_QUERIES = [
    "coffee mug",
    "mug",
    "cup",
    "coffee maker",
    "kettle",
    "spoon",
    "container",
]

HOT_CUP_SURFACE_QUERIES = [
    "table",
    "desk",
    "coaster",
    "tray",
    "flat surface",
    "empty area",
]

SUPPORT_QUERIES = [
    "laptop",
    "desk",
    "table",
    "stand",
    "shelf",
    "supporting surface",
]

METALLIC_TOOL_QUERIES = [
    "metallic tool",
    "screwdriver",
    "wrench",
    "scissors",
    "pliers",
    "metal object",
]

CONTAINER_QUERIES = [
    "cup",
    "mug",
    "bottle",
    "glass",
    "bowl",
    "box",
    "container",
]

AFFORDANCE_KEYWORDS = {
    "hold water": CONTAINER_QUERIES,
    "holding water": CONTAINER_QUERIES,
    "container": CONTAINER_QUERIES,
    "containers": CONTAINER_QUERIES,
    "coffee": COFFEE_QUERIES,
    "hot cup": HOT_CUP_SURFACE_QUERIES,
    "safest": HOT_CUP_SURFACE_QUERIES,
    "safe": HOT_CUP_SURFACE_QUERIES,
    "supports": SUPPORT_QUERIES,
    "support": SUPPORT_QUERIES,
    "laptop": SUPPORT_QUERIES,
    "metallic": METALLIC_TOOL_QUERIES,
    "metal": METALLIC_TOOL_QUERIES,
    "tools": METALLIC_TOOL_QUERIES,
}
