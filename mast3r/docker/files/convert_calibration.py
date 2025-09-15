#!/usr/bin/env python3
"""
Calibration parameter converter for MAST3R model compatibility.
This script converts camera calibration parameters from original resolution
to the resolution used by MAST3R's image preprocessing pipeline.

Usage:
    python convert_calibration.py <input_calibration.json> <output_calibration.json>
"""

import json
import numpy as np
import sys
import os
from loguru import logger


def calculate_resize_and_crop_params(original_size, target_size=512, patch_size=16):
    """
    Calculate resize and crop parameters for MAST3R preprocessing.
    
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
    W_resized = int(round(W1 * scale_factor))
    H_resized = int(round(H1 * scale_factor))
    
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
        'scale_factor': scale_factor,
        'resized_size': (W_resized, H_resized),
        'crop_offset': (crop_x1, crop_y1),
        'final_size': (W_final, H_final),
        'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2)
    }


def transform_camera_matrix(camera_matrix, transform_params):
    """
    Transform camera matrix (intrinsics) for new resolution.
    
    Args:
        camera_matrix: 3x3 camera matrix
        transform_params: Parameters from calculate_resize_and_crop_params
    
    Returns:
        Transformed 3x3 camera matrix
    """
    K = camera_matrix.copy()
    scale_factor = transform_params['scale_factor']
    crop_offset = transform_params['crop_offset']
    
    # Scale focal lengths
    K[0, 0] *= scale_factor  # fx
    K[1, 1] *= scale_factor  # fy
    
    # Adjust principal point: scale then offset for crop
    K[0, 2] = K[0, 2] * scale_factor - crop_offset[0]  # cx
    K[1, 2] = K[1, 2] * scale_factor - crop_offset[1]  # cy
    
    return K


def convert_calibration_parameters(calibration_data, original_size, target_size=512, patch_size=16):
    """
    Convert calibration parameters for MAST3R compatibility.
    
    Args:
        calibration_data: Original calibration data from JSON
        original_size: (width, height) of original images
        target_size: Target long edge size for MAST3R
        patch_size: Patch size for alignment
    
    Returns:
        Converted calibration data
    """
    logger.info(f"Converting calibration from {original_size} to MAST3R format")
    
    # Calculate transformation parameters
    transform_params = calculate_resize_and_crop_params(
        original_size, target_size, patch_size
    )
    
    logger.info(f"Transform params: {transform_params}")
    
    # Convert camera matrices
    camera_matrix1 = np.array(calibration_data['camera_matrix1'])
    camera_matrix2 = np.array(calibration_data['camera_matrix2'])
    
    new_camera_matrix1 = transform_camera_matrix(camera_matrix1, transform_params)
    new_camera_matrix2 = transform_camera_matrix(camera_matrix2, transform_params)
    
    # Create new calibration data
    new_calibration = calibration_data.copy()
    new_calibration['camera_matrix1'] = new_camera_matrix1.tolist()
    new_calibration['camera_matrix2'] = new_camera_matrix2.tolist()
    new_calibration['image_size'] = list(transform_params['final_size'])
    
    # Add transformation info for reference
    new_calibration['_transform_info'] = {
        'original_size': list(original_size),
        'target_size': target_size,
        'patch_size': patch_size,
        'scale_factor': transform_params['scale_factor'],
        'crop_offset': list(transform_params['crop_offset'])
    }
    
    logger.info(f"Converted image size: {new_calibration['image_size']}")
    logger.info(f"New camera matrix 1:\n{new_camera_matrix1}")
    logger.info(f"New camera matrix 2:\n{new_camera_matrix2}")
    
    return new_calibration


def main():
    """Main function to convert calibration parameters."""
    if len(sys.argv) != 3:
        print("Usage: python convert_calibration.py <input_calibration.json> <output_calibration.json>")
        print("Example: python convert_calibration.py /mast3r/tmp/mast3r_test/extrinsics_calibration.json /mast3r/tmp/mast3r_output/extrinsics_calibration_512x288.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input calibration file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Load original calibration
        with open(input_path, 'r') as f:
            calibration_data = json.load(f)
        
        # Get original size from calibration data
        original_size = tuple(calibration_data['image_size'])  # (width, height)
        
        # Convert for MAST3R (512x288 for 16:9 aspect ratio)
        converted_calibration = convert_calibration_parameters(
            calibration_data, 
            original_size, 
            target_size=512, 
            patch_size=16
        )
        
        # Save converted calibration
        with open(output_path, 'w') as f:
            json.dump(converted_calibration, f, indent=2)
        
        logger.info(f"Successfully converted calibration:")
        logger.info(f"  Input:  {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Original size: {original_size}")
        logger.info(f"  Converted size: {converted_calibration['image_size']}")
        logger.info(f"  Scale factor: {converted_calibration['_transform_info']['scale_factor']:.4f}")
        
    except Exception as e:
        logger.error(f"Error converting calibration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
