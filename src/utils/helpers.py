"""Shared utility helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2

from src.config import OUTPUT_DIR, SCREENSHOTS_DIR


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def discover_cameras(max_index: int = 5) -> list[int]:
    """Return indices of available camera devices."""
    available: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def next_json_output_path(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"scene_graph_{stamp}.json"


def next_screenshot_path(output_dir: Path = SCREENSHOTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"scene_graph_{stamp}.png"


def has_scene_graph_info(object_count: int, relation_count: int) -> bool:
    """True when the frame has at least one detected object or relation."""
    return object_count > 0 or relation_count > 0
