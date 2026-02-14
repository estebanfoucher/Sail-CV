"""Utility functions for crop extraction and validation."""

import numpy as np

from models.bounding_box import BoundingBox
from models.image import Image


def extract_crop_from_bbox(image: Image, bbox: BoundingBox) -> np.ndarray:
    """
    Extract a crop from an image based on a bounding box.

    The crop is extracted respecting image boundaries. If the bounding box
    extends beyond the image, the crop will be clipped to the image size.

    Args:
        image: Image object containing the image data
        bbox: BoundingBox defining the crop region

    Returns:
        Numpy array of the cropped region with shape (H, W, 3) where H and W
        are the height and width of the crop region.
    """
    img_array = image.image
    img_height, img_width = img_array.shape[:2]

    # Get bounding box coordinates
    x1 = max(0, bbox.xyxy.x1)
    y1 = max(0, bbox.xyxy.y1)
    x2 = min(img_width, bbox.xyxy.x2)
    y2 = min(img_height, bbox.xyxy.y2)

    # Extract crop
    crop = img_array[y1:y2, x1:x2]

    return crop


def validate_crop_coordinates(image: Image, bbox: BoundingBox) -> bool:
    """
    Validate that a bounding box is within image bounds.

    Args:
        image: Image object containing the image data
        bbox: BoundingBox to validate

    Returns:
        True if the bounding box is valid (has positive area and is within
        or partially within image bounds), False otherwise.
    """
    img_array = image.image
    img_height, img_width = img_array.shape[:2]

    # Check if bbox has valid coordinates
    return not (
        bbox.xyxy.x2 <= bbox.xyxy.x1
        or bbox.xyxy.y2 <= bbox.xyxy.y1
        or (
            bbox.xyxy.x2 <= 0
            or bbox.xyxy.x1 >= img_width
            or bbox.xyxy.y2 <= 0
            or bbox.xyxy.y1 >= img_height
        )
    )
