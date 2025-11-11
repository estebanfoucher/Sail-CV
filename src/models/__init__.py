"""
Pydantic models for the detection pipeline.

This module contains all Pydantic models used for type safety and validation:
- XYXY, BoundingBox: Bounding box representations
- Image: Image representation with color space
- Detection, ModelSpecs: Detection pipeline models
"""

from .bounding_box import XYXY, BoundingBox
from .detector import Detection, ModelSpecs
from .image import Image

__all__ = [
    "XYXY",
    "BoundingBox",
    "Detection",
    "Image",
    "ModelSpecs",
]
