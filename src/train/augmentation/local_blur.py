"""Motion blur applied only inside YOLO bbox ROIs (labels unchanged)."""

from __future__ import annotations

import random

import cv2
import numpy as np


def _yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    padding_frac: float,
) -> tuple[int, int, int, int]:
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    return x1, y1, x2, y2


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    """Line kernel along angle_deg (0 = horizontal motion)."""
    size = max(size, 3)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    rad = np.deg2rad(angle_deg)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for i in range(size):
        x = round(c + (i - c) * dx)
        y = round(c + (i - c) * dy)
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        kernel[y, x] = 1.0
    s = kernel.sum()
    if s < 1e-6:
        kernel[c, :] = 1.0
        s = kernel.sum()
    return kernel / s


def apply_motion_blur_to_bboxes(
    image_rgb: np.ndarray,
    bboxes_yolo: list[list[float]],
    *,
    rng: random.Random | None = None,
    p: float = 0.45,
    max_boxes_per_image: int = 4,
    padding_frac: float = 0.12,
    kernel_size: int = 15,
) -> np.ndarray:
    """
    For each bbox (stochastic), blur only pixels inside the (padded) ROI.

    Args:
        image_rgb: HxWx3 uint8 RGB
        bboxes_yolo: list of [xc, yc, w, h] normalized
        rng: random source (default: global random)
        p: probability each box is selected for blur (capped by max_boxes)
        max_boxes_per_image: max boxes to blur per image
    """
    if rng is None:
        rng = random.Random()
    if not bboxes_yolo or image_rgb.size == 0:
        return image_rgb

    h, w = image_rgb.shape[:2]
    out = image_rgb.copy()
    indices = list(range(len(bboxes_yolo)))
    rng.shuffle(indices)
    blurred = 0
    for i in indices:
        if blurred >= max_boxes_per_image:
            break
        if rng.random() > p:
            continue
        box = bboxes_yolo[i]
        xc, yc, bw, bh = box
        x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, bw, bh, w, h, padding_frac)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue
        ks = min(kernel_size, max(3, min(x2 - x1, y2 - y1) // 2 * 2 + 1))
        angle = rng.uniform(0, 180)
        k = _motion_blur_kernel(ks, angle)
        blurred_roi = cv2.filter2D(roi, -1, k)
        out[y1:y2, x1:x2] = blurred_roi
        blurred += 1
    return out
