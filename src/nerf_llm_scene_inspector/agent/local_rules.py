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

CUTTING_TOOL_QUERIES = [
    "scissors",
    "knife",
    "box cutter",
    "cutter",
    "blade",
    "cutting tool",
]

WRITING_OBJECT_QUERIES = [
    "pen",
    "pencil",
    "marker",
    "notebook",
    "paper",
    "writing tool",
]

MATERIAL_QUERIES = {
    "metallic": ["metallic object", "metal tool", "metal surface", "silver object"],
    "metal": ["metallic object", "metal tool", "metal surface", "silver object"],
    "wooden": ["wooden object", "wood surface", "wooden desk", "wooden box"],
    "wood": ["wooden object", "wood surface", "wooden desk", "wooden box"],
    "glass": ["glass", "glass object", "transparent object", "cup"],
    "plastic": ["plastic object", "plastic container", "plastic bottle", "plastic tool"],
}

SCENE_SEMANTIC_QUERIES = {
    "coffee-making": COFFEE_QUERIES,
    "coffee making": COFFEE_QUERIES,
    "coffee": COFFEE_QUERIES,
    "office work": ["laptop", "keyboard", "mouse", "notebook", "pen", "desk"],
    "work objects": ["laptop", "keyboard", "mouse", "notebook", "pen", "desk"],
    "fragile": ["glass", "screen", "ceramic mug", "bottle", "delicate object"],
    "containers": CONTAINER_QUERIES,
    "container": CONTAINER_QUERIES,
}

SPATIAL_KEYWORDS = {
    "supporting": ("support/on-top-of heuristic", SUPPORT_QUERIES),
    "supports": ("support/on-top-of heuristic", SUPPORT_QUERIES),
    "support": ("support/on-top-of heuristic", SUPPORT_QUERIES),
    "next to": ("near relation", []),
    "beside": ("near relation", []),
    "left side": ("left/right image-space or camera-space relation", ["left side", "desk", "table"]),
    "right side": ("left/right image-space or camera-space relation", ["right side", "desk", "table"]),
    "above": ("above/below relation", []),
    "below": ("above/below relation", []),
    "under": ("above/below or support relation", SUPPORT_QUERIES),
}

AFFORDANCE_KEYWORDS = {
    "hold water": CONTAINER_QUERIES,
    "can hold water": CONTAINER_QUERIES,
    "holding water": CONTAINER_QUERIES,
    "container": CONTAINER_QUERIES,
    "containers": CONTAINER_QUERIES,
    "coffee": COFFEE_QUERIES,
    "hot cup": HOT_CUP_SURFACE_QUERIES,
    "safest": HOT_CUP_SURFACE_QUERIES,
    "safe": HOT_CUP_SURFACE_QUERIES,
    "tools for cutting": CUTTING_TOOL_QUERIES,
    "cutting": CUTTING_TOOL_QUERIES,
    "useful for writing": WRITING_OBJECT_QUERIES,
    "writing": WRITING_OBJECT_QUERIES,
    "supports": SUPPORT_QUERIES,
    "support": SUPPORT_QUERIES,
    "laptop": SUPPORT_QUERIES,
    "metallic": METALLIC_TOOL_QUERIES,
    "metal": METALLIC_TOOL_QUERIES,
    "tools": METALLIC_TOOL_QUERIES,
}

NEGATIVE_QUERY_HINTS = {
    "safe": ["electronics", "screen", "edge of table", "clutter"],
    "hot cup": ["laptop", "keyboard", "paper stack", "unstable edge"],
    "water": ["flat screen", "electronics"],
}
