"""Structured console logging helpers."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "nerf_llm_scene_inspector", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger with a compact formatter."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
