"""Run Albumentations compose + optional bbox motion blur."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

from .local_blur import apply_motion_blur_to_bboxes

if TYPE_CHECKING:
    import albumentations as A


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
    if not bboxes:
        out = compose(image=image_rgb, bboxes=[], class_labels=[])
        img = out["image"]
        if bbox_motion_blur:
            img = apply_motion_blur_to_bboxes(
                img, [], rng=rng, p=0.0, max_boxes_per_image=0
            )
        return img, [], []

    out = compose(
        image=image_rgb,
        bboxes=[list(b) for b in bboxes],
        class_labels=list(class_labels),
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
