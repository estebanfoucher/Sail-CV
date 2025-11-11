from typing import Any

import cv2
import numpy as np

from models import Track


def get_color_for_class(
    class_id: int, class_info: dict[int, dict[str, Any]]
) -> tuple[int, int, int]:
    """
    Get the color for a given class ID from class_info.
    Args:
        class_id (int): The class ID.
        class_info (dict): Mapping from class_id to {"name", "color"}.
    Returns:
        tuple: (B, G, R) color.
    """
    return tuple(class_info[class_id]["color"])


def draw_single_track(
    frame: np.ndarray,
    track: Track,
    class_info: dict[int, dict[str, Any]],
    show_confidence: bool = False,
    show_class_name: bool = False,
) -> None:
    """
    Draw a single track (bounding box, track ID, class name if enabled) on the frame.
    Args:
        frame (np.ndarray): The image to draw on (BGR format).
        track (Track): Track Pydantic model with detection, track_id, frame_id.
        class_info (dict): Mapping from class_id to {"name", "color"}.
        show_confidence (bool): Whether to display confidence score.
        show_class_name (bool): Whether to display class name.
    """
    xyxy = track.detection.bbox.xyxy
    class_id = track.detection.class_id
    color = get_color_for_class(class_id, class_info)
    text = ""
    track_id = track.track_id
    conf = track.detection.confidence
    label = class_info[class_id]["name"] if show_class_name else ""

    if show_class_name:
        text += f"{label}"
    if track_id is not None:
        if text:
            text += f" ID:{track_id}"
        else:
            text = f"ID:{track_id}"
    if show_confidence and conf is not None:
        text += f" {conf:.2f}"

    x1, y1, x2, y2 = int(xyxy.x1), int(xyxy.y1), int(xyxy.x2), int(xyxy.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    class_info: dict[int, dict[str, Any]],
    show_confidence: bool = False,
    show_class_name: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, and (optionally) class names on a frame using the original class color.
    Args:
        frame (np.ndarray): The image to draw on (BGR format).
        tracks (List[Track]): List of Track Pydantic models.
        class_info (dict): Mapping from class_id to {"name", "color"}.
        show_confidence (bool): Whether to display confidence score on the box.
        show_class_name (bool): Whether to display class name (default: False).
    Returns:
        np.ndarray: The frame with drawings.
    """
    for track in tracks:
        draw_single_track(
            frame,
            track,
            class_info,
            show_confidence=show_confidence,
            show_class_name=show_class_name,
        )
    return frame
