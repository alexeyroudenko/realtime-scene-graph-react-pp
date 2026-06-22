"""Visualization helpers for OpenCV rendering."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from src.config import (
    INFO_PANEL_LINE_HEIGHT,
    INFO_PANEL_MAX_LINES,
    INFO_PANEL_MIN_HEIGHT,
    INFO_PANEL_WIDTH,
)
from src.inference.scene_graph import SceneGraphResult


def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per class via golden-ratio hue spacing."""
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_bounding_box(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> tuple[int, int]:
    """Draw a styled bounding box and return its center point."""
    left, top, right, bottom = [int(v) for v in bbox]
    cv2.rectangle(image, (left, top), (right, bottom), color, 1)

    length = min(15, int((right - left) * 0.2), int((bottom - top) * 0.2))
    corners = [
        ((left, top), (1, 1)),
        ((right, top), (-1, 1)),
        ((left, bottom), (1, -1)),
        ((right, bottom), (-1, -1)),
    ]
    for (x, y), (dx, dy) in corners:
        cv2.line(image, (x, y), (x + dx * length, y), color, 3)
        cv2.line(image, (x, y), (x, y + dy * length), color, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, 0.45, 1)
    label_y = top - 6
    if label_y - text_h < 0:
        cv2.rectangle(image, (left, top), (left + text_w + 6, top + text_h + 6), color, -1)
        cv2.putText(image, label, (left + 3, top + text_h + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (left, label_y - text_h - 4), (left + text_w + 6, label_y + 2), color, -1)
        cv2.putText(image, label, (left + 3, label_y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return (left + right) // 2, (top + bottom) // 2


def draw_relation(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    predicate: str,
    score: float,
) -> None:
    """Draw an arrowed relation line with predicate label."""
    dist = float(np.hypot(start[0] - end[0], start[1] - end[1]))
    if dist <= 40:
        return

    alpha = 20.0 / (dist + 1e-6)
    p1 = (
        int(start[0] * (1 - alpha) + end[0] * alpha),
        int(start[1] * (1 - alpha) + end[1] * alpha),
    )
    p2 = (
        int(end[0] * (1 - alpha) + start[0] * alpha),
        int(end[1] * (1 - alpha) + start[1] * alpha),
    )

    cv2.line(image, p1, p2, (255, 128, 0), 2, cv2.LINE_AA)
    cv2.line(image, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    for delta in (0.5, -0.5):
        tip = (
            int(p2[0] + 8 * np.cos(angle + delta)),
            int(p2[1] + 8 * np.sin(angle + delta)),
        )
        cv2.line(image, p2, tip, (255, 255, 255), 1, cv2.LINE_AA)

    mid = (
        int(start[0] * 0.65 + end[0] * 0.35),
        int(start[1] * 0.65 + end[1] * 0.35),
    )
    label = f"{predicate} ({score:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
    cv2.rectangle(image, (mid[0] - 2, mid[1] - th - 2), (mid[0] + tw + 2, mid[1] + 2), (20, 20, 20), -1)
    cv2.putText(image, label, (mid[0], mid[1] - 1), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def render_scene_graph(
    frame: np.ndarray,
    result: SceneGraphResult,
    fps: float | None = None,
    paused: bool = False,
    provider: str = "CPU",
) -> np.ndarray:
    """Draw boxes, relations, and HUD overlay on the live frame."""
    canvas = frame.copy()
    centers: list[tuple[int, int]] = []

    for obj in result.objects:
        label = f"{obj.index}: {obj.class_name} {obj.confidence:.2f}"
        center = draw_bounding_box(canvas, obj.bbox, label, class_color(obj.class_id))
        centers.append(center)

    for rel in result.relations:
        if rel.subject_index >= len(centers) or rel.object_index >= len(centers):
            continue
        draw_relation(
            canvas,
            centers[rel.subject_index],
            centers[rel.object_index],
            rel.predicate,
            rel.triplet_score,
        )

    _draw_hud(canvas, result, fps, paused, provider)
    return canvas


def render_info_panel(
    result: SceneGraphResult,
    width: int = INFO_PANEL_WIDTH,
    line_height: int = INFO_PANEL_LINE_HEIGHT,
) -> np.ndarray:
    """Render a separate panel listing objects and relations."""
    object_lines = [f"[{obj.index}] {obj.class_name} ({obj.confidence:.2f})" for obj in result.objects]
    relation_lines = [
        (
            f"{rel.subject_index}->{rel.object_index}: "
            f"{result.objects[rel.subject_index].class_name} "
            f"- {rel.predicate} - "
            f"{result.objects[rel.object_index].class_name} "
            f"({rel.triplet_score:.2f})"
        )
        for rel in result.relations
        if rel.subject_index < len(result.objects) and rel.object_index < len(result.objects)
    ]

    sections = [
        ("Objects", object_lines or ["(none)"]),
        ("Relations", relation_lines or ["(none)"]),
    ]

    height = 40
    for _, lines in sections:
        height += 28 + len(lines) * line_height
    height = max(height, INFO_PANEL_MIN_HEIGHT)

    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28

    for title, lines in sections:
        cv2.putText(panel, title, (12, y), font, 0.6, (120, 220, 255), 1, cv2.LINE_AA)
        y += 24
        for line in lines[:INFO_PANEL_MAX_LINES]:
            cv2.putText(panel, line[:58], (16, y), font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            y += line_height
        y += 8

    return panel


def _draw_hud(
    image: np.ndarray,
    result: SceneGraphResult,
    fps: float | None,
    paused: bool,
    provider: str,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, (0.3 * image.shape[1]) / 500)
    lines = [
        f"Provider: {provider}",
        f"Inference: {result.inference_ms:.1f} ms",
        f"Objects: {len(result.objects)}  Relations: {len(result.relations)}",
    ]
    if fps is not None:
        lines.insert(0, f"FPS: {fps:.1f}")
    if paused:
        lines.append("PAUSED")

    lines.extend(["q/ESC quit | s save | p pause | r reset | c camera"])

    for i, text in enumerate(lines):
        pos = (12, 24 + i * int(28 * scale))
        cv2.putText(image, text, pos, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, pos, font, scale, (80, 255, 80), 1, cv2.LINE_AA)
