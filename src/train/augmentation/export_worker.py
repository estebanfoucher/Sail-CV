"""Multiprocessing worker for parallel flat YOLO export (one Compose per process)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .apply import augment_yolo_sample
from .pipelines import build_augmentation_pipeline, seed_augmentation_globals
from .yolo_io import (
    load_image_rgb,
    parse_yolo_label_file,
    save_image_rgb,
    write_yolo_label_file,
)

# List singleton so worker processes can cache Compose without `global` (multiprocessing).
_compose_cache: list[Any] = []


def export_flat_one_image(args: tuple) -> str:
    """
    Export one source image: original JPG + label, then `repeats` augmented pairs.

    Tuple fields:
        src_image, labels_dir, export_img, export_lbl (str paths),
        idx (int), repeats (int), base_seed (int | None), no_bbox_blur (bool), preset (str).
    """
    (
        src_image_s,
        labels_dir_s,
        export_img_s,
        export_lbl_s,
        idx,
        repeats,
        base_seed,
        no_bbox_blur,
        preset,
    ) = args
    img_path = Path(src_image_s)
    labels_dir = Path(labels_dir_s)
    export_img = Path(export_img_s)
    export_lbl = Path(export_lbl_s)
    stem = img_path.stem
    label_path = labels_dir / f"{stem}.txt"

    image_rgb = load_image_rgb(img_path)
    bboxes, class_labels = parse_yolo_label_file(label_path)

    save_image_rgb(export_img / f"{stem}.jpg", image_rgb)
    write_yolo_label_file(export_lbl / f"{stem}.txt", bboxes, class_labels)

    if not _compose_cache:
        _compose_cache.append(build_augmentation_pipeline(seed=None, preset=preset))
    compose = _compose_cache[0]

    for r in range(repeats):
        sub_seed = None
        if base_seed is not None:
            sub_seed = base_seed + idx * 10_007 + r * 1_003
        if sub_seed is not None:
            compose.set_random_seed(sub_seed)
            seed_augmentation_globals(sub_seed)
        rng = random.Random(sub_seed) if sub_seed is not None else random.Random()
        aug, bb, cl = augment_yolo_sample(
            image_rgb,
            bboxes,
            class_labels,
            compose,
            rng=rng,
            bbox_motion_blur=not no_bbox_blur,
        )
        save_image_rgb(export_img / f"{stem}_aug{r}.jpg", aug)
        write_yolo_label_file(export_lbl / f"{stem}_aug{r}.txt", bb, cl)

    return stem
