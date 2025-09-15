#!/usr/bin/env python3
"""
MASt3R Utility Functions

This module contains utility functions for saving results, camera verification,
and various export formats. These functions are separated from the main inference
logic for better code organization and reusability.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, filename: str) -> None:
    """
    Save point cloud as PLY file.
    
    Args:
        points: 3D points array of shape (H, W, 3) or (N, 3)
        colors: Color array of shape (H, W, 3) or (N, 3) with values in [0, 1]
        filename: Output PLY file path
    """
    # Flatten the points and colors
    points_flat = points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    
    # Filter out invalid points (NaN, inf)
    valid_mask = np.isfinite(points_flat).all(axis=1)
    points_valid = points_flat[valid_mask]
    colors_valid = colors_flat[valid_mask]
    
    # Convert colors to 0-255 range
    colors_valid = (colors_valid * 255).astype(np.uint8)
    
    # Write PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_valid)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points_valid)):
            f.write(f"{points_valid[i, 0]:.6f} {points_valid[i, 1]:.6f} {points_valid[i, 2]:.6f} "
                   f"{colors_valid[i, 0]} {colors_valid[i, 1]} {colors_valid[i, 2]}\n")
    
    logger.info(f"Point cloud saved to {filename} with {len(points_valid)} points")


def save_pixel_pairs(matches_im0: np.ndarray, matches_im1: np.ndarray, filename: str) -> None:
    """
    Save pixel pairs (dense matching results) in a structured format.
    
    Args:
        matches_im0: Pixel coordinates in first image, shape (N, 2)
        matches_im1: Pixel coordinates in second image, shape (N, 2)
        filename: Output JSON file path
    """
    # Create structured data for pixel pairs
    pixel_pairs = []
    for i in range(len(matches_im0)):
        pair = {
            'id': i,
            'image1_pixel': [float(matches_im0[i, 0]), float(matches_im0[i, 1])],
            'image2_pixel': [float(matches_im1[i, 0]), float(matches_im1[i, 1])]
        }
        pixel_pairs.append(pair)
    
    # Save as JSON
    with open(filename, 'w') as f:
        json.dump({
            'pixel_pairs': pixel_pairs,
            'num_pairs': len(pixel_pairs),
            'format': 'dense_matching_results',
            'description': 'Pixel correspondences between two images from MASt3R dense matching'
        }, f, indent=2)
    
    logger.info(f"Pixel pairs saved to {filename} with {len(pixel_pairs)} pairs")


def verify_camera_data(cams2world_1: np.ndarray, cams2world_2: np.ndarray, 
                      image1_path: str, image2_path: str) -> bool:
    """
    Verify camera data format and print summary.
    
    Args:
        cams2world_1: Camera-to-world matrix for first camera
        cams2world_2: Camera-to-world matrix for second camera
        image1_path: Path to first image
        image2_path: Path to second image
        
    Returns:
        bool: True if camera data is valid, False otherwise
    """
    logger.info("=== CAMERA DATA VERIFICATION ===")
    
    # Extract camera names
    cam1_name = os.path.splitext(os.path.basename(image1_path))[0]
    cam2_name = os.path.splitext(os.path.basename(image2_path))[0]
    
    # Extract positions and rotations
    pos1 = cams2world_1[:3, 3]
    pos2 = cams2world_2[:3, 3]
    R1 = cams2world_1[:3, :3]
    R2 = cams2world_2[:3, :3]
    
    # Calculate distances and angles
    distance = np.linalg.norm(pos2 - pos1)
    
    # Check if matrices are valid rotation matrices
    det1 = np.linalg.det(R1)
    det2 = np.linalg.det(R2)
    ortho1 = np.allclose(R1 @ R1.T, np.eye(3), atol=1e-6)
    ortho2 = np.allclose(R2 @ R2.T, np.eye(3), atol=1e-6)
    
    logger.info(f"Camera 1 ({cam1_name}):")
    logger.info(f"  Position: [{pos1[0]:.3f}, {pos1[1]:.3f}, {pos1[2]:.3f}]")
    logger.info(f"  Rotation matrix determinant: {det1:.6f} (should be ±1)")
    logger.info(f"  Orthogonal: {ortho1}")
    
    logger.info(f"Camera 2 ({cam2_name}):")
    logger.info(f"  Position: [{pos2[0]:.3f}, {pos2[1]:.3f}, {pos2[2]:.3f}]")
    logger.info(f"  Rotation matrix determinant: {det2:.6f} (should be ±1)")
    logger.info(f"  Orthogonal: {ortho2}")
    
    logger.info(f"Camera baseline distance: {distance:.3f} units")
    
    # Validation results
    valid = (abs(det1 - 1.0) < 1e-6 and abs(det2 - 1.0) < 1e-6 and ortho1 and ortho2)
    logger.info(f"Data validation: {'PASS' if valid else 'FAIL'}")
    
    return valid


def create_project_structure(output_dir: str, image1_path: str, image2_path: str) -> Dict[str, str]:
    """
    Create organized project directory structure.
    
    Args:
        output_dir: Base output directory
        image1_path: Path to first input image
        image2_path: Path to second input image
        
    Returns:
        Dict containing paths to created directories
    """
    output_path = Path(output_dir)
    
    # Create main directories
    dirs = {
        'point_clouds': output_path / 'point_clouds',
        'cameras': output_path / 'cameras',
        'matches': output_path / 'matches',
        'images': output_path / 'images'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy input images to project
    import shutil
    shutil.copy2(image1_path, dirs['images'] / Path(image1_path).name)
    shutil.copy2(image2_path, dirs['images'] / Path(image2_path).name)
    
    logger.info(f"Project structure created in {output_dir}")
    return {k: str(v) for k, v in dirs.items()}


