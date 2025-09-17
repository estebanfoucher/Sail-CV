"""
MVS (Multi-View Stereo) Utilities Library

This library provides utilities for multi-view stereo calibration and processing.
"""
# Import base modules first
from .stereo_data_folder_structure import load_stereo_data_folder_structure, load_scene_folder_structure
from .utils import load_parameters
from .video_utils import Video, get_unique_video_name
from .intrinsics_calibration import IntrinsicCalibration
from .extrinsics_calibration import StereoTagDetector, calibrate_stereo_many, get_summary

# Import higher-level modules that depend on base modules
from .scene import Scene

__version__ = "1.0.0"
__all__ = [
    "Scene",
    "IntrinsicCalibration",
    "StereoTagDetector",
    "calibrate_stereo_many",
    "get_summary",
    "Video",
    "get_unique_video_name",
    "load_stereo_data_folder_structure",
    "load_scene_folder_structure",
    "load_parameters"
]
