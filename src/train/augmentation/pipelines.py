"""Albumentation Compose pipelines for YOLO (geom + global weather + color)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import albumentations as A

import albumentations as A


def _set_deterministic(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def seed_augmentation_globals(seed: int | None) -> None:
    """Sync random + NumPy RNG before applying a Compose (use with a single reused pipeline)."""
    _set_deterministic(seed)


def _albumentations_major() -> int:
    return int(A.__version__.split(".", maxsplit=1)[0])


def build_augmentation_pipeline(
    *,
    seed: int | None = None,
    preset: str = "sail_default",
) -> A.Compose:
    """
    Full-frame Albumentations chain: geometry (updates bboxes) then weather/color.

    Bbox-local motion blur is applied separately in apply.py.
    """
    _set_deterministic(seed)
    if preset != "sail_default":
        raise ValueError(f"Unknown preset: {preset!r} (only 'sail_default')")

    maj = _albumentations_major()

    # Shadow ROI biased toward frame center so polygons often cross object ROIs.
    # Slightly reduced vs the earlier default to avoid overly harsh shadows.
    if maj >= 2:
        shadow = A.RandomShadow(
            shadow_roi=(0.05, 0.08, 0.95, 0.92),
            num_shadows_limit=(1, 3),
            shadow_dimension=6,
            p=0.45,
        )
    else:
        shadow = A.RandomShadow(
            shadow_roi=(0.05, 0.08, 0.95, 0.92),
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=6,
            p=0.45,
        )

    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_area=0.0,
        min_visibility=0.0,
    )

    if maj >= 2:
        gauss_noise = A.GaussNoise(
            std_range=(0.032, 0.19),
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1.0,
            p=0.2,
        )
        coarse_dropout = A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(24, 24),
            hole_width_range=(24, 24),
            fill=0,
            p=0.2,
        )
        random_fog = A.RandomFog(
            alpha_coef=0.1,
            fog_coef_range=(0.1, 0.35),
            p=0.28,
        )
        random_rain = A.RandomRain(
            slant_range=(-8, 8),
            drop_length=18,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.75,
            rain_type="default",
            p=0.25,
        )
        random_sun_flare = A.RandomSunFlare(
            flare_roi=(0.1, 0.1, 0.9, 0.6),
            angle_range=(0.3, 0.7),
            num_flare_circles_range=(2, 5),
            src_radius=180,
            src_color=(255, 255, 255),
            p=0.15,
        )
        random_snow = A.RandomSnow(
            brightness_coeff=2.0,
            snow_point_range=(0.1, 0.3),
            p=0.15,
        )
    else:
        gauss_noise = A.GaussNoise(var_limit=(8.0, 48.0), p=0.2)
        coarse_dropout = A.CoarseDropout(
            max_holes=4,
            max_height=24,
            max_width=24,
            min_holes=1,
            fill=0,
            p=0.2,
        )
        random_fog = A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.35,
            alpha_coef=0.1,
            p=0.28,
        )
        random_rain = A.RandomRain(
            slant_lower=-8,
            slant_upper=8,
            drop_length=18,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.75,
            rain_type="default",
            p=0.25,
        )
        random_sun_flare = A.RandomSunFlare(
            flare_roi=(0.1, 0.1, 0.9, 0.6),
            angle_lower=0.3,
            angle_upper=0.7,
            num_flare_circles_lower=2,
            num_flare_circles_upper=5,
            src_radius=180,
            src_color=(255, 255, 255),
            p=0.15,
        )
        random_snow = A.RandomSnow(
            brightness_coeff=2.0,
            snow_point_lower=0.1,
            snow_point_upper=0.3,
            p=0.15,
        )

    transforms: list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.12),
        A.Rotate(
            limit=18,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.45,
        ),
        A.Affine(
            # Slightly restricted scale to avoid shrinking objects too much.
            scale=(0.90, 1.10),
            translate_percent=0.04,
            rotate=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.35,
        ),
        # Slightly restricted random scaling.
        A.RandomScale(scale_limit=0.08, p=0.35),
        A.HueSaturationValue(
            hue_shift_limit=12,
            sat_shift_limit=22,
            val_shift_limit=18,
            p=0.45,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.18,
            contrast_limit=0.18,
            p=0.45,
        ),
        A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.25),
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.35), p=0.2),
        gauss_noise,
        coarse_dropout,
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.AdvancedBlur(blur_limit=(3, 7), p=0.12),
        shadow,
        random_fog,
        random_rain,
        random_sun_flare,
        random_snow,
        A.RandomGravel(number_of_patches=8, p=0.12),
        A.ToGray(p=0.08),
    ]

    return A.Compose(transforms, bbox_params=bbox_params, seed=seed)
