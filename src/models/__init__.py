"""
Pydantic models for the detection and tracking pipeline.

This module contains all Pydantic models used for type safety and validation:
- XYXY, BoundingBox: Bounding box representations
- Image: Image representation with color space
- Detection, ModelSpecs: Detection pipeline models
- Track, TrackerConfig: Tracking pipeline models
- Layout, LayoutPosition: Layout-based tracking models
- CropModule: Base class for crop analysis modules
- MaskDetector: Base class for mask detection modules
- BackgroundDetector: Base class for background detection modules
"""

from .background_detector import BackgroundDetector
from .bounding_box import XYXY, BoundingBox
from .crop_module import CropModule
from .detector import Detection, ModelSpecs
from .image import Image
from .layout import Layout, LayoutPosition
from .mask_detector import MaskDetector
from .track import Track, TrackerConfig

__all__ = [
    "BackgroundDetector",
    "XYXY",
    "BoundingBox",
    "CropModule",
    "Detection",
    "Image",
    "Layout",
    "LayoutPosition",
    "MaskDetector",
    "ModelSpecs",
    "Track",
    "TrackerConfig",
]
