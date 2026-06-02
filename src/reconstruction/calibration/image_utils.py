#!/usr/bin/env python3
"""
Image processing utilities for handling EXIF orientation and other image operations.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose


def apply_exif_transpose_cv2(frame: np.ndarray) -> np.ndarray:
    """
    Apply EXIF transpose to OpenCV frame to handle orientation metadata.

    Args:
        frame: OpenCV frame (BGR numpy array)

    Returns:
        Corrected frame with proper orientation
    """
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Apply EXIF transpose
    corrected_pil = exif_transpose(pil_image)

    # Convert back to RGB numpy array
    corrected_rgb = np.array(corrected_pil)

    # Convert RGB back to BGR for OpenCV
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)

    return corrected_bgr


def ensure_landscape_orientation(frame: np.ndarray) -> np.ndarray:
    """
    Apply EXIF transpose (for in-memory frames) and rotate portrait to landscape.

    Matches VideoReader orientation handling used during scene calibration.
    """
    corrected_frame = apply_exif_transpose_cv2(frame)
    height, width = corrected_frame.shape[:2]
    if width >= height:
        return corrected_frame
    return cv2.rotate(corrected_frame, cv2.ROTATE_90_CLOCKWISE)


def load_orientation_corrected_image(path: Path | str) -> np.ndarray:
    """
    Load an image file with EXIF orientation applied, then ensure landscape.

    Opens via PIL so EXIF orientation tags are respected (unlike cv2.imread).
    """
    pil_image = exif_transpose(Image.open(path)).convert("RGB")
    frame_rgb = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return ensure_landscape_orientation(frame_bgr)


def apply_exif_transpose_to_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Apply EXIF transpose to a list of OpenCV frames.

    Args:
        frames: List of OpenCV frames (BGR numpy arrays)

    Returns:
        List of corrected frames with proper orientation
    """
    return [apply_exif_transpose_cv2(frame) for frame in frames]


def save_frame_with_exif_handling(frame: np.ndarray, output_path: str) -> bool:
    """
    Save a frame with EXIF orientation handling.

    Args:
        frame: OpenCV frame (BGR numpy array)
        output_path: Path to save the frame

    Returns:
        True if successful, False otherwise
    """
    try:
        corrected_frame = apply_exif_transpose_cv2(frame)
        cv2.imwrite(output_path, corrected_frame)
        return True
    except Exception as e:
        print(f"Error saving frame with EXIF handling: {e}")
        return False
