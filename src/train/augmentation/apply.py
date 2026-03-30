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
    min_bbox_side_px: int = 10,
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

    h, w = image_rgb.shape[:2]
    original_min_sides: list[float] = [min(bw * w, bh * h) for (_, _, bw, bh) in bboxes]

    out = compose(
        image=image_rgb,
        bboxes=[list(b) for b in bboxes],
        class_labels=list(class_labels),
    )
    img = out["image"]
    b_out = [list(b) for b in out["bboxes"]]
    c_out = list(out["class_labels"])

    # Enforce: for bboxes that were >= `min_bbox_side_px` before augmentation,
    # avoid shrinking them below `min_bbox_side_px` afterwards.
    # Native small boxes (originally < min_bbox_side_px) are preserved.
    # NOTE: Albumentations preserves order when we keep min_visibility/min_area low.
    if b_out and min_bbox_side_px > 0:
        kept_b: list[list[float]] = []
        kept_c: list[int] = []
        for i, (box, cls) in enumerate(zip(b_out, c_out, strict=True)):
            bw, bh = box[2], box[3]
            new_min_side = min(bw * w, bh * h)
            # Albumentations preserves bbox order when we keep min_area/min_visibility low.
            old_min_side = original_min_sides[i]
            if old_min_side >= min_bbox_side_px and new_min_side < min_bbox_side_px:
                continue
            kept_b.append(box)
            kept_c.append(cls)
        b_out, c_out = kept_b, kept_c

    if bbox_motion_blur and b_out:
        img = apply_motion_blur_to_bboxes(
            img,
            b_out,
            rng=rng,
            p=motion_blur_p,
            max_boxes_per_image=4,
        )
    return img, b_out, c_out
