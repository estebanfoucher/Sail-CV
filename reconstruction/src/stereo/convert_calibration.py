#!/usr/bin/env python3
"""
Calibration parameter converter for MAST3R model compatibility.
This script converts camera calibration parameters from original resolution
to the resolution used by MAST3R's image preprocessing pipeline.
"""

import numpy as np
from loguru import logger


def calculate_resize_and_crop_params(original_size, target_size=512, patch_size=16):
    """
    Calculate resize and crop parameters for MASt3R preprocessing.

    Args:
        original_size: (width, height) of original image
        target_size: Target long edge size (default 512)
        patch_size: Patch size for alignment (default 16)

    Returns:
        Dictionary with transformation parameters
    """
    W1, H1 = original_size

    # Calculate resize factor (resize long edge to target_size)
    S = max(W1, H1)
    scale_factor = target_size / S

    # Calculate new size after resize
    W_resized = round(W1 * scale_factor)
    H_resized = round(H1 * scale_factor)

    # Calculate crop parameters (center crop with patch_size alignment)
    cx, cy = W_resized // 2, H_resized // 2

    # Align to patch_size multiples
    halfw = ((2 * cx) // patch_size) * patch_size / 2
    halfh = ((2 * cy) // patch_size) * patch_size / 2

    # For 16:9 aspect ratio, adjust height
    if W_resized == H_resized:  # Square case
        halfh = 3 * halfw / 4

    # Final crop coordinates
    crop_x1 = int(cx - halfw)
    crop_y1 = int(cy - halfh)
    crop_x2 = int(cx + halfw)
    crop_y2 = int(cy + halfh)

    # Final dimensions
    W_final = crop_x2 - crop_x1
    H_final = crop_y2 - crop_y1

    return {
        "scale_factor": scale_factor,
        "resized_size": (W_resized, H_resized),
        "crop_offset": (crop_x1, crop_y1),
        "final_size": (W_final, H_final),
        "crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2),
    }


def transform_camera_matrix(camera_matrix, transform_params):
    """
    Transform camera intrinsics matrix for new resolution.

    Args:
        camera_matrix: 3x3 camera matrix
        transform_params: Parameters from calculate_resize_and_crop_params

    Returns:
        Transformed 3x3 camera matrix
    """
    K = camera_matrix.copy()
    scale_factor = transform_params["scale_factor"]
    crop_offset = transform_params["crop_offset"]

    # Scale focal lengths
    K[0, 0] *= scale_factor  # fx
    K[1, 1] *= scale_factor  # fy

    # Adjust principal point: scale then offset for crop
    K[0, 2] = K[0, 2] * scale_factor - crop_offset[0]  # cx
    K[1, 2] = K[1, 2] * scale_factor - crop_offset[1]  # cy

    return K


def convert_calibration_parameters(
    calibration_data, original_size=None, target_size=512, patch_size=16
):
    """
    Convert calibration parameters for MASt3R compatibility.

    Args:
        calibration_data: Original calibration data from JSON
        original_size: (width, height) of original images (optional, uses image_size from calibration_data if not provided)
        target_size: Target long edge size for MASt3R
        patch_size: Patch size for alignment

    Returns:
        Converted calibration data
    """
    # Use image_size from calibration_data if original_size not provided
    if original_size is None:
        if "image_size" in calibration_data:
            original_size = tuple(calibration_data["image_size"])
            logger.debug(f"Using image_size from calibration_data: {original_size}")
        else:
            raise ValueError(
                "original_size must be provided or calibration_data must contain 'image_size'"
            )

    logger.debug(f"Converting calibration from {original_size} to MAST3R format")

    # Calculate transformation parameters
    transform_params = calculate_resize_and_crop_params(
        original_size, target_size, patch_size
    )

    # Convert camera matrices
    camera_matrix1 = np.array(calibration_data["camera_matrix1"])
    camera_matrix2 = np.array(calibration_data["camera_matrix2"])

    new_camera_matrix1 = transform_camera_matrix(camera_matrix1, transform_params)
    new_camera_matrix2 = transform_camera_matrix(camera_matrix2, transform_params)

    # Create new calibration data
    new_calibration = calibration_data.copy()
    new_calibration["camera_matrix1"] = new_camera_matrix1.tolist()
    new_calibration["camera_matrix2"] = new_camera_matrix2.tolist()
    new_calibration["image_size"] = list(transform_params["final_size"])

    # Add transformation info for reference
    new_calibration["_transform_info"] = {
        "original_size": list(original_size),
        "target_size": target_size,
        "patch_size": patch_size,
        "scale_factor": transform_params["scale_factor"],
        "crop_offset": list(transform_params["crop_offset"]),
    }

    return new_calibration
