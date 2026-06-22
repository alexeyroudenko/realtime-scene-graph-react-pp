"""Pure OpenCV real-time scene graph application."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROLS_WINDOW_HEIGHT,
    CONTROLS_WINDOW_WIDTH,
    DEFAULT_BOX_CONF,
    DEFAULT_MAX_RELATIONS,
    DEFAULT_REL_CONF,
    DEFAULT_SCREENSHOT_INTERVAL_SEC,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SCREENSHOTS_DIR,
    TRACKBAR_BOX_CONF,
    TRACKBAR_MAX_REL,
    TRACKBAR_REL_CONF,
    WINDOW_CONTROLS,
    WINDOW_INFO,
    WINDOW_MAIN,
)
from src.inference.scene_graph import SceneGraphEngine, SceneGraphResult
from src.utils.helpers import discover_cameras, next_json_output_path, open_camera, setup_logging
from src.utils.screenshot_saver import ScreenshotSaver
from src.visualization.drawer import render_info_panel, render_scene_graph

logger = logging.getLogger(__name__)


class AppState:
    """Mutable UI state shared with OpenCV trackbar callbacks."""

    def __init__(
        self,
        box_conf: float = DEFAULT_BOX_CONF,
        rel_conf: float = DEFAULT_REL_CONF,
        max_relations: int = DEFAULT_MAX_RELATIONS,
    ) -> None:
        self.box_conf = box_conf
        self.rel_conf = rel_conf
        self.max_relations = max_relations
        self.paused = False
        self.rotation_deg = 0
        self.last_result: SceneGraphResult | None = None

    def reset_defaults(self) -> tuple[int, int, int]:
        self.box_conf = DEFAULT_BOX_CONF
        self.rel_conf = DEFAULT_REL_CONF
        self.max_relations = DEFAULT_MAX_RELATIONS
        return self.trackbar_values()

    def trackbar_values(self) -> tuple[int, int, int]:
        return (
            int(round(self.box_conf * 100)),
            int(round(self.rel_conf * 100)),
            self.max_relations,
        )

    def apply_to_engine(self, engine: SceneGraphEngine) -> None:
        engine.set_thresholds(
            box_conf=self.box_conf,
            rel_conf=self.rel_conf,
            max_relations=self.max_relations,
        )


def _noop_trackbar(_value: int) -> None:
    pass


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a camera frame by 0, 90, 180, or 270 degrees."""
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _sync_window_size(
    window_name: str,
    width: int,
    height: int,
    last_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Resize an OpenCV window when content dimensions change."""
    if last_size != (width, height):
        cv2.resizeWindow(window_name, width, height)
    return width, height


def create_control_window(state: AppState) -> None:
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, CONTROLS_WINDOW_WIDTH, CONTROLS_WINDOW_HEIGHT)

    box_val, rel_val, max_val = state.trackbar_values()
    cv2.createTrackbar(TRACKBAR_BOX_CONF, WINDOW_CONTROLS, box_val, 100, _noop_trackbar)
    cv2.createTrackbar(TRACKBAR_REL_CONF, WINDOW_CONTROLS, rel_val, 100, _noop_trackbar)
    cv2.createTrackbar(TRACKBAR_MAX_REL, WINDOW_CONTROLS, max_val, 50, _noop_trackbar)


def read_trackbars(state: AppState) -> None:
    state.box_conf = cv2.getTrackbarPos(TRACKBAR_BOX_CONF, WINDOW_CONTROLS) / 100.0
    state.rel_conf = cv2.getTrackbarPos(TRACKBAR_REL_CONF, WINDOW_CONTROLS) / 100.0
    state.max_relations = max(1, cv2.getTrackbarPos(TRACKBAR_MAX_REL, WINDOW_CONTROLS))


def set_trackbars(state: AppState) -> None:
    box_val, rel_val, max_val = state.trackbar_values()
    cv2.setTrackbarPos(TRACKBAR_BOX_CONF, WINDOW_CONTROLS, box_val)
    cv2.setTrackbarPos(TRACKBAR_REL_CONF, WINDOW_CONTROLS, rel_val)
    cv2.setTrackbarPos(TRACKBAR_MAX_REL, WINDOW_CONTROLS, max_val)


def run(
    camera_index: int = 0,
    model_path: Path | None = None,
    show_info_panel: bool = True,
    auto_screenshots: bool = False,
    screenshot_interval: float = DEFAULT_SCREENSHOT_INTERVAL_SEC,
    run_seconds: float | None = None,
) -> int:
    setup_logging()
    sys.path.insert(0, str(PROJECT_ROOT))

    logger.info("Initializing REACT++ scene graph engine...")
    engine = SceneGraphEngine(model_path=model_path)
    provider = engine.session.get_providers()[0]
    state = AppState()
    state.apply_to_engine(engine)

    cameras = discover_cameras()
    if not cameras:
        logger.error("No camera devices found.")
        return 1

    current_camera = camera_index if camera_index in cameras else cameras[0]
    cap = open_camera(current_camera, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not cap.isOpened():
        logger.error("Failed to open camera %s", current_camera)
        return 1

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    create_control_window(state)
    if show_info_panel:
        cv2.namedWindow(WINDOW_INFO, cv2.WINDOW_NORMAL)

    logger.info(
        "Running on camera %s. Controls: q/ESC quit, s save JSON, p pause, r rotate, 0 reset, c camera, a auto-screenshots.",
        current_camera,
    )
    if auto_screenshots:
        logger.info(
            "Auto-screenshots enabled (every %.1fs when detections present) -> %s",
            screenshot_interval,
            SCREENSHOTS_DIR,
        )

    screenshot_saver = ScreenshotSaver(
        enabled=auto_screenshots,
        interval_sec=screenshot_interval,
        output_dir=SCREENSHOTS_DIR,
    )
    started_at = time.monotonic()

    fps = 0.0
    fps_alpha = 0.9
    empty_result = SceneGraphResult()
    raw_frame: np.ndarray | None = None
    frame: np.ndarray | None = None
    main_window_size: tuple[int, int] | None = None
    info_panel_size: tuple[int, int] | None = None

    try:
        while True:
            if not state.paused:
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    logger.warning("Failed to read frame from camera %s", current_camera)
                    break

                frame = rotate_frame(raw_frame, state.rotation_deg)

                read_trackbars(state)
                state.apply_to_engine(engine)

                loop_start = time.perf_counter()
                state.last_result = engine.predict(frame)
                elapsed = time.perf_counter() - loop_start
                instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                fps = fps_alpha * fps + (1.0 - fps_alpha) * instant_fps
            elif raw_frame is None:
                break

            result = state.last_result or empty_result
            vis = render_scene_graph(frame, result, fps=fps, paused=state.paused, provider=provider)
            screenshot_saver.maybe_save(vis, result)
            vis_h, vis_w = vis.shape[:2]
            main_window_size = _sync_window_size(WINDOW_MAIN, vis_w, vis_h, main_window_size)
            cv2.imshow(WINDOW_MAIN, vis)

            if show_info_panel:
                panel = render_info_panel(result)
                panel_h, panel_w = panel.shape[:2]
                info_panel_size = _sync_window_size(WINDOW_INFO, panel_w, panel_h, info_panel_size)
                cv2.imshow(WINDOW_INFO, panel)

            key = cv2.waitKey(1 if not state.paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                state.paused = not state.paused
            elif key == ord("r"):
                state.rotation_deg = (state.rotation_deg + 90) % 360
                if raw_frame is not None:
                    frame = rotate_frame(raw_frame, state.rotation_deg)
                    read_trackbars(state)
                    state.apply_to_engine(engine)
                    state.last_result = engine.predict(frame)
                main_window_size = None
                logger.info("Camera rotation: %d°", state.rotation_deg)
            elif key == ord("0"):
                state.reset_defaults()
                set_trackbars(state)
                state.apply_to_engine(engine)
            elif key == ord("s"):
                if state.last_result is not None:
                    out_path = next_json_output_path(OUTPUT_DIR)
                    state.last_result.save_json(out_path)
                    logger.info("Saved scene graph to %s", out_path)
            elif key == ord("c"):
                if len(cameras) > 1:
                    pos = cameras.index(current_camera)
                    current_camera = cameras[(pos + 1) % len(cameras)]
                    cap.release()
                    cap = open_camera(current_camera, CAMERA_WIDTH, CAMERA_HEIGHT)
                    main_window_size = None
                    logger.info("Switched to camera %s", current_camera)
                else:
                    logger.info("Only one camera available.")
            elif key == ord("a"):
                enabled = screenshot_saver.toggle()
                logger.info("Auto-screenshots %s", "enabled" if enabled else "disabled")

            if run_seconds is not None and (time.monotonic() - started_at) >= run_seconds:
                logger.info(
                    "Run time limit reached (%.0fs). Saved %d screenshot(s).",
                    run_seconds,
                    screenshot_saver.saved_count,
                )
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time scene graph generation with pure OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Initial webcam index.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a local REACT++ ONNX model (auto-downloads if omitted).",
    )
    parser.add_argument(
        "--no-info-panel",
        action="store_true",
        help="Disable the secondary info window.",
    )
    parser.add_argument(
        "--auto-screenshots",
        action="store_true",
        help="Periodically save annotated screenshots when objects or relations are detected.",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=DEFAULT_SCREENSHOT_INTERVAL_SEC,
        help="Minimum seconds between auto-screenshots (default: 5).",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Auto-quit after N seconds (useful for timed runs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        camera_index=args.camera,
        model_path=args.model,
        show_info_panel=not args.no_info_panel,
        auto_screenshots=args.auto_screenshots,
        screenshot_interval=args.screenshot_interval,
        run_seconds=args.run_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
