"""Smoke tests for train.augmentation (requires `train` extra / albumentations)."""

from __future__ import annotations

import random
from pathlib import Path  # noqa: TC003

import pytest

pytest.importorskip("albumentations")

import numpy as np

from train.augmentation.apply import augment_yolo_sample
from train.augmentation.local_blur import apply_motion_blur_to_bboxes
from train.augmentation.pipelines import build_augmentation_pipeline
from train.augmentation.yolo_io import (
    parse_yolo_label_file,
    write_yolo_label_file,
)


def test_build_pipeline_and_augment_with_boxes(tmp_path: Path) -> None:
    compose = build_augmentation_pipeline(seed=42)
    img = np.zeros((120, 160, 3), dtype=np.uint8) + 128
    bboxes = [[0.5, 0.5, 0.25, 0.3]]
    classes = [1]
    rng = random.Random(42)
    aug, bb, cl = augment_yolo_sample(
        img,
        bboxes,
        classes,
        compose,
        rng=rng,
        bbox_motion_blur=True,
    )
    assert aug.shape[2] == 3
    assert aug.dtype == np.uint8
    assert len(bb) == len(cl)
    lbl = tmp_path / "t.txt"
    write_yolo_label_file(lbl, bb, cl)
    bb2, cl2 = parse_yolo_label_file(lbl)
    assert len(bb2) == len(cl2)
    assert cl2 == cl
    for a, b in zip(bb, bb2, strict=True):
        assert np.allclose(a, b, rtol=0, atol=1e-5)


def test_bbox_motion_blur_changes_roi() -> None:
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[:, :] = (200, 180, 160)
    # bright box center
    img[30:50, 40:60] = (250, 250, 250)
    bboxes = [[0.5, 0.5, 0.2, 0.25]]
    rng = random.Random(0)
    out = apply_motion_blur_to_bboxes(
        img,
        bboxes,
        rng=rng,
        p=1.0,
        max_boxes_per_image=1,
        kernel_size=21,
    )
    inner_before = img[35:45, 45:55].astype(np.float32).mean()
    inner_after = out[35:45, 45:55].astype(np.float32).mean()
    assert abs(inner_after - inner_before) > 1.0


def test_augment_empty_labels() -> None:
    compose = build_augmentation_pipeline(seed=0)
    img = np.ones((64, 64, 3), dtype=np.uint8) * 99
    aug, bb, cl = augment_yolo_sample(img, [], [], compose, bbox_motion_blur=False)
    assert aug.ndim == 3
    assert bb == []
    assert cl == []
