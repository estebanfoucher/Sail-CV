"""
Input validation utilities for MVS web application
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Import existing classes from src
from video import VideoReader
from calibration.utils import validate_calibration

# Configure logger
logger = logging.getLogger(__name__)


def validate_video_file(file_path: str) -> tuple[bool, str]:
    """
    Validate that the uploaded file is a valid MP4 video using existing VideoReader class

    Args:
        file_path: Path to the uploaded file

    Returns:
        tuple: (is_valid, error_message)
    """
    logger.debug(f"Validating video file: {file_path}")

    try:
        if not os.path.exists(file_path):
            logger.warning(f"Video file does not exist: {file_path}")
            return False, f"File does not exist: {file_path}"

        # Check file extension
        if not file_path.lower().endswith('.mp4'):
            logger.warning(f"Video file has wrong extension: {os.path.splitext(file_path)[1]}")
            return False, f"File must be MP4 format, got: {os.path.splitext(file_path)[1]}"

        # Use existing VideoReader class to validate
        try:
            logger.debug(f"Opening video file with VideoReader: {file_path}")
            reader = VideoReader.open_video_file(file_path)
            if reader is None:
                logger.error(f"VideoReader returned None for: {file_path}")
                return False, "Could not open video file with VideoReader"
        except Exception as e:
            logger.error(f"Error opening video file {file_path}: {str(e)}")
            return False, f"Error opening video file: {str(e)}"

        # Check if we can read at least one frame
        try:
            logger.debug(f"Reading first frame from: {file_path}")
            ret, frame = reader.read()
            reader.release()

            if not ret:
                logger.error(f"Could not read frames from: {file_path}")
                return False, "Could not read any frames from video"

            if frame is None:
                logger.error(f"Frame is None from: {file_path}")
                return False, "Video frame is empty or corrupted"

            logger.info(f"Video file validated successfully: {file_path}")
            return True, "Video file is valid"

        except Exception as e:
            logger.error(f"Error reading video frames from {file_path}: {str(e)}")
            reader.release()
            return False, f"Error reading video frames: {str(e)}"

    except Exception as e:
        logger.error(f"Unexpected error validating video {file_path}: {str(e)}", exc_info=True)
        return False, f"Unexpected error validating video: {str(e)}"


def validate_calibration_file(file_path: str) -> tuple[bool, str]:
    """
    Validate that the uploaded file is a valid calibration JSON using existing validation

    Args:
        file_path: Path to the uploaded calibration file

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"

        # Check file extension
        if not file_path.lower().endswith('.json'):
            return False, f"File must be JSON format, got: {os.path.splitext(file_path)[1]}"

        # Try to load and parse JSON
        try:
            with open(file_path, 'r') as f:
                calib_data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}"
        except Exception as e:
            return False, f"Error reading JSON file: {str(e)}"

        # Use existing validation function from mv_utils
        is_valid, error_msg = validate_calibration(calib_data)
        return is_valid, error_msg

    except Exception as e:
        return False, f"Unexpected error validating calibration file: {str(e)}"


def get_video_info(file_path: str) -> Dict[str, Any]:
    """
    Get video information using existing VideoReader class

    Args:
        file_path: Path to the video file

    Returns:
        dict: Video information or empty dict if invalid
    """
    try:
        # Use existing VideoReader class
        reader = VideoReader.open_video_file(file_path)
        if reader is None:
            return {}

        # Get video specs from existing class
        specs = reader.specs

        return {
            'width': specs.width,
            'height': specs.height,
            'fps': specs.fps,
            'frame_count': specs.frame_count,
            'duration': specs.duration_seconds or (specs.frame_count / specs.fps if specs.fps > 0 else 0),
            'resolution': f"{specs.width}x{specs.height}"
        }

    except Exception:
        return {}


def validate_video_compatibility(video_1_path: str, video_2_path: str) -> Tuple[bool, str]:
    """
    Validate that two videos are compatible for stereo processing

    Args:
        video_1_path: Path to first video
        video_2_path: Path to second video

    Returns:
        tuple: (is_compatible, message)
    """
    try:
        # First validate individual videos
        valid_1, error_1 = validate_video_file(video_1_path)
        if not valid_1:
            return False, f"Video 1 validation failed: {error_1}"

        valid_2, error_2 = validate_video_file(video_2_path)
        if not valid_2:
            return False, f"Video 2 validation failed: {error_2}"

        # Get video information
        info_1 = get_video_info(video_1_path)
        info_2 = get_video_info(video_2_path)

        if not info_1 or not info_2:
            return False, "Could not read video information for compatibility check"

        # Check resolution compatibility
        if info_1['resolution'] != info_2['resolution']:
            return False, f"Resolution mismatch: Video 1 has {info_1['resolution']}, Video 2 has {info_2['resolution']}"

        # Check frame count compatibility (allow some tolerance)
        frame_diff = abs(info_1['frame_count'] - info_2['frame_count'])
        if frame_diff > 10:  # Allow up to 10 frame difference
            return False, f"Frame count mismatch: Video 1 has {info_1['frame_count']} frames, Video 2 has {info_2['frame_count']} frames (difference: {frame_diff})"

        # Check FPS compatibility
        fps_diff = abs(info_1['fps'] - info_2['fps'])
        if fps_diff > 0.1:  # Allow small FPS difference
            return False, f"FPS mismatch: Video 1 has {info_1['fps']:.2f} FPS, Video 2 has {info_2['fps']:.2f} FPS (difference: {fps_diff:.2f})"

        return True, f"Videos are compatible: {info_1['resolution']}, {info_1['fps']:.2f} FPS, {info_1['frame_count']} frames"

    except Exception as e:
        return False, f"Error validating video compatibility: {str(e)}"


def get_calibration_info(file_path: str) -> Dict[str, Any]:
    """
    Get calibration file information using existing validation

    Args:
        file_path: Path to the calibration file

    Returns:
        dict: Calibration information or empty dict if invalid
    """
    try:
        with open(file_path, 'r') as f:
            calib_data = json.load(f)

        # Use existing validation function
        is_valid = validate_calibration(calib_data)

        if not is_valid:
            return {}

        # Get calibration details
        info = {
            'is_valid': True,
            'has_camera_matrix1': 'camera_matrix1' in calib_data,
            'has_camera_matrix2': 'camera_matrix2' in calib_data,
            'has_dist_coeffs1': 'dist_coeffs1' in calib_data,
            'has_dist_coeffs2': 'dist_coeffs2' in calib_data,
            'has_rotation_matrix': 'rotation_matrix' in calib_data,
            'has_translation_vector': 'translation_vector' in calib_data,
            'success': calib_data.get('success', False),
            'reprojection_error': calib_data.get('reprojection_error', 0.0),
            'num_correspondences': calib_data.get('num_correspondences', 0),
            'num_pairs': calib_data.get('num_pairs', 0)
        }

        return info

    except Exception:
        return {}
