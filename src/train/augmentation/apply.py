"""Run Albumentations compose + optional bbox motion blur."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

from .local_blur import apply_motion_blur_to_bboxes

if TYPE_CHECKING:
    import albumentations as A


def _sanitize_yolo_boxes(
    bboxes: list[list[float]],
    class_labels: list[int],
    *,
    eps: float = 1e-6,
) -> tuple[list[list[float]], list[int]]:
    """
    Clamp YOLO boxes to valid range and drop degenerate entries.

    This avoids occasional floating-point overflows at image borders
    (e.g. tiny negative y_min) that Albumentations refuses.
    """
    out_b: list[list[float]] = []
    out_c: list[int] = []
    for box, cls in zip(bboxes, class_labels, strict=True):
        xc, yc, bw, bh = map(float, box[:4])
        x1 = xc - bw / 2.0
        y1 = yc - bh / 2.0
        x2 = xc + bw / 2.0
        y2 = yc + bh / 2.0

        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        x2 = min(1.0, max(0.0, x2))
        y2 = min(1.0, max(0.0, y2))

        if x2 - x1 <= eps or y2 - y1 <= eps:
            continue

        bw2 = x2 - x1
        bh2 = y2 - y1
        xc2 = (x1 + x2) / 2.0
        yc2 = (y1 + y2) / 2.0

        out_b.append([xc2, yc2, bw2, bh2])
        out_c.append(int(cls))
    return out_b, out_c


def augment_yolo_sample(
    image_rgb: np.ndarray,
    bboxes: list[list[float]],
    class_labels: list[int],
    compose: A.Compose,
    *,
    rng: random.Random | None = None,
    bbox_motion_blur: bool = True,
    motion_blur_p: float = 0.45,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """
    Apply Albumentations then optional ROI motion blur.

    Returns:
        augmented RGB image, bboxes (YOLO), class_labels (aligned, may be shorter if filtered)
    """
    if rng is None:
        rng = random.Random()
    bboxes_in, class_in = _sanitize_yolo_boxes(bboxes, class_labels)
    if not bboxes_in:
        out = compose(image=image_rgb, bboxes=[], class_labels=[])
        img = out["image"]
        if bbox_motion_blur:
            img = apply_motion_blur_to_bboxes(
                img, [], rng=rng, p=0.0, max_boxes_per_image=0
            )
        return img, [], []

    out = compose(
        image=image_rgb,
        bboxes=[list(b) for b in bboxes_in],
        class_labels=list(class_in),
    )
    img = out["image"]
    b_out = [list(b) for b in out["bboxes"]]
    c_out = list(out["class_labels"])

    if bbox_motion_blur and b_out:
        img = apply_motion_blur_to_bboxes(
            img,
            b_out,
            rng=rng,
            p=motion_blur_p,
            max_boxes_per_image=4,
        )
    return img, b_out, c_out
