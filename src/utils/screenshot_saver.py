"""Periodic screenshot capture when scene graph data is present."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.inference.scene_graph import SceneGraphResult
from src.utils.helpers import has_scene_graph_info, next_screenshot_path

logger = logging.getLogger(__name__)


class ScreenshotSaver:
    """Save annotated frames at a fixed interval when detections exist."""

    def __init__(
        self,
        enabled: bool = False,
        interval_sec: float = 5.0,
        output_dir: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.interval_sec = max(0.5, interval_sec)
        self.output_dir = output_dir
        self.last_save_monotonic = 0.0
        self.saved_count = 0

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def maybe_save(self, vis: np.ndarray, result: SceneGraphResult) -> Path | None:
        if not self.enabled:
            return None
        if not has_scene_graph_info(len(result.objects), len(result.relations)):
            return None

        now = time.monotonic()
        if self.last_save_monotonic and (now - self.last_save_monotonic) < self.interval_sec:
            return None

        path = next_screenshot_path(self.output_dir) if self.output_dir else next_screenshot_path()
        if not cv2.imwrite(str(path), vis):
            logger.warning("Failed to write screenshot to %s", path)
            return None

        self.last_save_monotonic = now
        self.saved_count += 1
        logger.info(
            "Saved screenshot %s (%d objects, %d relations)",
            path,
            len(result.objects),
            len(result.relations),
        )
        return path
