"""
MVS (Multi-View Stereo) Utilities Library

This library provides utilities for multi-view stereo calibration and processing.
"""

from ._log import logger

# Import base modules first
from .extrinsics_calibration import (
    CharucoDetector,
    StereoTagDetector,
    calibrate_stereo_many,
    get_summary,
)
from .intrinsics_calibration import IntrinsicCalibration
from .utils import load_parameters

__version__ = "1.0.0"
__all__ = [
    "CharucoDetector",
    "IntrinsicCalibration",
    "StereoTagDetector",
    "calibrate_stereo_many",
    "get_summary",
    "load_parameters",
]

# Higher-level modules pull in heavy/optional deps (e.g. pydantic via video.py).
# Keep them optional so the core calibration API works in lean environments
# (such as the Jetson's pip-less system Python).
try:
    from .scene import Scene
    from .stereo_data_folder_structure import (
        load_scene_folder_structure,
        load_stereo_data_folder_structure,
    )
    from .video_utils import Video, get_unique_video_name

    __all__ += [
        "Scene",
        "Video",
        "get_unique_video_name",
        "load_scene_folder_structure",
        "load_stereo_data_folder_structure",
    ]
except ImportError as _e:  # pragma: no cover - depends on runtime env
    logger.debug(f"mv_utils higher-level modules unavailable: {_e}")
