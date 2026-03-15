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
- TelltalePoint, SailGeometry, CameraConfig, Sail3DConfig: 3D sail tracking models
"""

from .background_detector import BackgroundDetector
from .bounding_box import XYXY, BoundingBox
from .classifier import ClassifierConfig
from .crop_module import CropModule
from .detector import Detection, ModelSpecs
from .image import Image
from .layout import Layout, LayoutPosition
from .mask_detector import MaskDetector
from .pipeline_config import (
    ArrowSenseConfig,
    BackgroundDetectorConfig,
    CropModuleConfig,
    DetectorConfig,
    LayoutTrackerConfig,
    OutputConfig,
    PipelineConfig,
    VisualizationConfig,
)
from .sail_3d import CameraConfig, Sail3DConfig, SailGeometry, TelltalePoint
from .track import Track, TrackerConfig

__all__ = [
    "XYXY",
    "ArrowSenseConfig",
    "BackgroundDetector",
    "BackgroundDetectorConfig",
    "BoundingBox",
    "CameraConfig",
    "ClassifierConfig",
    "CropModule",
    "CropModuleConfig",
    "Detection",
    "DetectorConfig",
    "Image",
    "Layout",
    "LayoutPosition",
    "LayoutTrackerConfig",
    "MaskDetector",
    "ModelSpecs",
    "OutputConfig",
    "PipelineConfig",
    "Sail3DConfig",
    "SailGeometry",
    "TelltalePoint",
    "Track",
    "TrackerConfig",
    "VisualizationConfig",
]
