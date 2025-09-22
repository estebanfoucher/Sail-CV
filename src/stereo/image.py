#!/usr/bin/env python3
"""
Image processing utilities for MASt3R
Provides resizing functions based on dust3r.utils.image logic
"""

import PIL.Image
from dust3r.utils.image import _resize_pil_image
from PIL.ImageOps import exif_transpose


def resize_image(image_path, size, square_ok=False, patch_size=16):
    """
    Resize image using dust3r preprocessing logic without tensor conversion.

    Args:
        image_path: Path to the image file
        size: Target size for resizing (224 for crop, other for long edge)
        square_ok: Whether square images are acceptable
        patch_size: Patch size for alignment (default 16)

    Returns:
        PIL.Image: Resized and cropped PIL image
    """
    # Load image with EXIF handling
    img = exif_transpose(PIL.Image.open(image_path)).convert("RGB")

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
