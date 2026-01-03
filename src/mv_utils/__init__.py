"""
MVS (Multi-View Stereo) Utilities Library

This library provides utilities for multi-view stereo calibration and processing.
"""

# Import base modules first
from .extrinsics_calibration import (
    CharucoDetector,
    StereoTagDetector,
    calibrate_stereo_many,
    get_summary,
)
from .intrinsics_calibration import IntrinsicCalibration

# Import higher-level modules that depend on base modules
from .scene import Scene
from .stereo_data_folder_structure import (
    load_scene_folder_structure,
    load_stereo_data_folder_structure,
)
from .utils import load_parameters
from .video_utils import Video, get_unique_video_name

__version__ = "1.0.0"
__all__ = [
    "CharucoDetector",
    "IntrinsicCalibration",
    "Scene",
    "StereoTagDetector",
    "Video",
    "calibrate_stereo_many",
    "get_summary",
    "get_unique_video_name",
    "load_parameters",
    "load_scene_folder_structure",
    "load_stereo_data_folder_structure",
]
