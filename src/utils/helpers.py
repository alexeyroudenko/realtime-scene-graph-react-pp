"""Shared utility helpers."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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


def open_video(path: Path) -> cv2.VideoCapture:
    """Open a video file for frame-by-frame reading."""
    cap = cv2.VideoCapture(str(path))
    return cap


def pick_video_file() -> Path | None:
    """Show a native file dialog and return the selected video path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        logging.getLogger(__name__).error("tkinter is unavailable; cannot open file dialog.")
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    if sys.platform == "win32":
        root.update()

    path = filedialog.askopenfilename(
        parent=root,
        title="Select video file",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.webm *.m4v"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def read_frame(cap: cv2.VideoCapture, *, loop_video: bool) -> tuple[bool, np.ndarray | None]:
    """Read the next frame; optionally loop video files at EOF."""
    ret, frame = cap.read()
    if ret and frame is not None:
        return True, frame
    if not loop_video:
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap.read()


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
