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


def crop_to_match_resolution(
    img: np.ndarray, target_H: int, target_W: int
) -> np.ndarray:
    """
    Crop image to match target resolution by shrinking (conserving aspect ratio) then cropping.
    Used for matching different camera resolutions during calibration.

    Strategy:
    1. Calculate scale factors for both dimensions
    2. Use the LARGER scale factor (max) capped at 1.0 (never enlarge) to preserve aspect ratio
    3. Resize image with this scale factor
    4. Center crop to match target dimensions (removes top/bottom or left/right bands)

    This ensures after shrinking, both dimensions are >= target, so we can crop both.
    Example: img (2880, 3840) -> target (2160, 3840)
      - scale_H = 0.75, scale_W = 1.0
      - Using max(0.75, 1.0) = 1.0: no shrink needed, just crop height from 2880 to 2160
      - Using min(0.75, 1.0) = 0.75: would shrink to (2160, 2880), then can't crop width from 2880 to 3840!

    Args:
        img: numpy array (H, W, 3) in RGB format
        target_H: Target height (should be <= img height)
        target_W: Target width (should be <= img width)

    Returns:
        numpy array (target_H, target_W, 3) cropped to target resolution
    """
    H, W = img.shape[:2]

    # If already at target resolution, return as is
    if target_H == H and target_W == W:
        return img.copy()

    # Calculate scale factors for both dimensions
    scale_H = target_H / H
    scale_W = target_W / W

    # Use the LARGER scale factor (max) capped at 1.0 to shrink while conserving aspect ratio
    # This ensures after shrinking, both dimensions are >= target, so we can crop both
    # Example: img (2880, 3840) -> target (2160, 3840)
    #   scale_H = 0.75, scale_W = 1.0
    #   Using max(0.75, 1.0) = 1.0: no shrink needed, just crop height from 2880 to 2160
    #   Using min(0.75, 1.0) = 0.75: shrink to (2160, 2880), then can't crop width from 2880 to 3840!
    # Never enlarge (cap at 1.0)
    scale = min(max(scale_H, scale_W), 1.0)

    # Resize with aspect ratio preserved
    new_H = round(H * scale)
    new_W = round(W * scale)
    img_resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # Center crop to match target dimensions
    # After resize, both dimensions should be >= target (if we didn't enlarge)
    # Calculate crop offsets
    crop_y = (new_H - target_H) // 2
    crop_x = (new_W - target_W) // 2

    # Crop: remove top/bottom bands if height needs adjustment, left/right if width needs adjustment
    img_cropped = img_resized[crop_y : crop_y + target_H, crop_x : crop_x + target_W]

    return img_cropped


def resize_image_numpy(
    img: np.ndarray, size: int, square_ok: bool = False, patch_size: int = 16
) -> np.ndarray:
    """
    Resize numpy array image using same logic as resize_image().

    Args:
        img: numpy array (H, W, 3) in RGB format
        size: Target size for resizing (224 for crop, other for long edge)
        square_ok: Whether square images are acceptable
        patch_size: Patch size for alignment (default 16)

    Returns:
        numpy array (H', W', 3) resized and cropped
    """
    H1, W1 = img.shape[:2]

    if size == 224:
        # resize short side to 224 (then crop)
        scale_factor = size / min(W1, H1)
        W_resized = round(W1 * scale_factor)
        H_resized = round(H1 * scale_factor)
    else:
        # resize long edge to specified size
        long_edge = max(W1, H1)
        scale_factor = size / long_edge
        W_resized = round(W1 * scale_factor)
        H_resized = round(H1 * scale_factor)

    # Resize using OpenCV
    img_resized = cv2.resize(
        img, (W_resized, H_resized), interpolation=cv2.INTER_LINEAR
    )

    # Center crop with patch_size alignment
    cx, cy = W_resized // 2, H_resized // 2

    if size == 224:
        half = min(cx, cy)
        crop_x1, crop_y1 = int(cx - half), int(cy - half)
        crop_x2, crop_y2 = int(cx + half), int(cy + half)
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and W_resized == H_resized:
            halfh = 3 * halfw / 4
        crop_x1, crop_y1 = int(cx - halfw), int(cy - halfh)
        crop_x2, crop_y2 = int(cx + halfw), int(cy + halfh)

    # Crop using numpy slicing
    img_cropped = img_resized[crop_y1:crop_y2, crop_x1:crop_x2]

    return img_cropped


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
