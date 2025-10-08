#!/usr/bin/env python3
"""
Image processing utilities for MASt3R
Provides resizing functions based on dust3r.utils.image logic
"""

import cv2
import numpy as np
import PIL.Image
from dust3r.utils.image import ImgNorm, _resize_pil_image
from loguru import logger
from PIL.ImageOps import exif_transpose


def resize_image(img, size, square_ok=False, patch_size=16):
    """
    Resize image using dust3r preprocessing logic without tensor conversion.

    Args:
        img: PIL image
        size: Target size for resizing (224 for crop, other for long edge)
        square_ok: Whether square images are acceptable
        patch_size: Patch size for alignment (default 16)

    Returns:
        PIL.Image: Resized and cropped PIL image
    """

    W1, H1 = img.size

    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to specified size
        img = _resize_pil_image(img, size)

    W, H = img.size
    cx, cy = W // 2, H // 2

    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    return img


def load_image(image_path, size=None):
    """Load and preprocess image from path for preprocess_image"""
    logger.debug(f"Loading image: {image_path}")
    assert size is not None, "Size must be provided"
    image = exif_transpose(PIL.Image.open(image_path)).convert("RGB")
    return image


def convert_image(image: np.ndarray):
    """Convert image output from VideoReader to PIL Image input of preprocess_image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(image_rgb)
    return image_pil


def preprocess_image(image, size=None, square_ok=False, patch_size=16, idx=0):
    """Preprocess image for MASt3R inference."""

    # input image loaded as such img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
    W1, H1 = image.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(image, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(image, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    W2, H2 = img.size
    logger.debug(f" - preprocessed {image} with resolution {W1}x{H1} --> {W2}x{H2}")
    img = {
        "img": ImgNorm(img)[None],
        "true_shape": np.int32([img.size[::-1]]),
        "idx": idx,
        "instance": str(idx),
    }

    return img
