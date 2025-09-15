#!/usr/bin/env python3
"""
MASt3R Post-Processing Module
Handles camera pose estimation and data processing
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Import utility functions to avoid duplication
from mast3r_utils import verify_camera_data


def estimate_camera_poses_from_3d_points(pts3d_1, pts3d_2, matches_im0, matches_im1, view1, view2):
    """
    Estimate camera poses from 3D points and matches using a simple approach
    """
    print("Estimating camera poses from 3D points and matches...")
    
    # Get image dimensions
    H1, W1 = view1['true_shape'][0]
    H2, W2 = view2['true_shape'][0]
    
    # Sample some matches to estimate poses
    n_matches = min(100, len(matches_im0))
    sample_indices = np.linspace(0, len(matches_im0)-1, n_matches, dtype=int)
    
    # Get 3D points corresponding to matches
    pts3d_1_matched = []
    pts3d_2_matched = []
    
    for idx in sample_indices:
        x1, y1 = matches_im0[idx]
        x2, y2 = matches_im1[idx]
        
        # Convert to integer indices for 3D point lookup
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        
        # Check bounds
        if 0 <= x1 < W1 and 0 <= y1 < H1 and 0 <= x2 < W2 and 0 <= y2 < H2:
            # Get 3D points (note: pts3d_2 is already in the coordinate system of image 1)
            pt1 = pts3d_1[y1, x1]
            pt2 = pts3d_2[y2, x2]
            
            # Check if points are valid (not NaN or inf)
            if np.isfinite(pt1).all() and np.isfinite(pt2).all():
                pts3d_1_matched.append(pt1)
                pts3d_2_matched.append(pt2)
    
    if len(pts3d_1_matched) < 10:
        print("Warning: Not enough valid 3D point matches, using identity poses")
        return np.eye(4), np.eye(4)
    
    pts3d_1_matched = np.array(pts3d_1_matched)
    pts3d_2_matched = np.array(pts3d_2_matched)
    
    # Camera 1 is at origin (identity pose)
    cams2world_1 = np.eye(4)
    
    # Estimate camera 2 pose using Procrustes analysis
    # Center the points
    centroid_1 = np.mean(pts3d_1_matched, axis=0)
    centroid_2 = np.mean(pts3d_2_matched, axis=0)
    
    centered_1 = pts3d_1_matched - centroid_1
    centered_2 = pts3d_2_matched - centroid_2
    
    # Compute rotation using SVD
    H = centered_2.T @ centered_1
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    
    # Ensure proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Compute translation
    t = centroid_1 - R @ centroid_2
    
    # Create camera 2 pose matrix
    cams2world_2 = np.eye(4)
    cams2world_2[:3, :3] = R
    cams2world_2[:3, 3] = t
    
    print(f"Estimated camera poses:")
    print(f"Camera 1 (identity): {cams2world_1}")
    print(f"Camera 2: {cams2world_2}")
    
    return cams2world_1, cams2world_2

def extract_camera_poses(raw_data):
    """Extract or estimate camera poses from raw inference data"""
    pred1 = raw_data['pred1']
    pred2 = raw_data['pred2']
    
    try:
        # Check for various possible camera pose keys
        camera_keys = ['cams2world', 'cam2world', 'pose', 'camera_pose']
        cams2world_1 = None
        cams2world_2 = None
        
        for key in camera_keys:
            if key in pred1:
                cams2world_1 = pred1[key].squeeze(0).detach().cpu().numpy()
                print(f"Found camera pose for image 1 in key: {key}")
                break
                
        for key in camera_keys:
            if key in pred2:
                cams2world_2 = pred2[key].squeeze(0).detach().cpu().numpy()
                print(f"Found camera pose for image 2 in key: {key}")
                break
        
        # If no camera poses found, estimate them from 3D points and matches
        if cams2world_1 is None or cams2world_2 is None:
            print("No camera poses found in predictions, estimating from 3D points and matches...")
            cams2world_1, cams2world_2 = estimate_camera_poses_from_3d_points(
                raw_data['pts3d_1'], raw_data['pts3d_2'], 
                raw_data['matches_im0'], raw_data['matches_im1'], 
                raw_data['view1'], raw_data['view2']
            )
            
    except Exception as e:
        # If camera poses are not available, estimate them
        print(f"Error extracting camera poses: {e}, estimating from 3D points...")
        cams2world_1, cams2world_2 = estimate_camera_poses_from_3d_points(
            raw_data['pts3d_1'], raw_data['pts3d_2'], 
            raw_data['matches_im0'], raw_data['matches_im1'], 
            raw_data['view1'], raw_data['view2']
        )
    
    return cams2world_1, cams2world_2

def process_point_clouds(raw_data):
    """Process and color point clouds"""
    pts3d_1 = raw_data['pts3d_1']
    pts3d_2 = raw_data['pts3d_2']
    img1_colors = raw_data['img1_colors']
    img2_colors = raw_data['img2_colors']
    
    # Resize colors to match point cloud resolution
    H1, W1 = pts3d_1.shape[:2]
    H2, W2 = pts3d_2.shape[:2]
    
    img1_colors_resized = cv2.resize(img1_colors, (W1, H1))
    img2_colors_resized = cv2.resize(img2_colors, (W2, H2))
    
    return {
        'pts3d_1': pts3d_1,
        'pts3d_2': pts3d_2,
        'colors_1': img1_colors_resized,
        'colors_2': img2_colors_resized
    }

def process_mast3r_results(raw_data, image1_path, image2_path):
    """
    Process raw MASt3R inference results
    
    Args:
        raw_data: Raw inference results from MASt3R
        image1_path: Path to first image
        image2_path: Path to second image
    
    Returns:
        dict: Processed results with cameras and point clouds
    """
    print("\n=== POST-PROCESSING MASt3R RESULTS ===")
    
    # Extract camera poses
    cams2world_1, cams2world_2 = extract_camera_poses(raw_data)
    
    # Verify camera data
    camera_data_valid = verify_camera_data(cams2world_1, cams2world_2, image1_path, image2_path)
    
    # Process point clouds
    processed_clouds = process_point_clouds(raw_data)
    
    # Create default camera intrinsics (can be overridden later)
    # Assuming 1920x1080 images with reasonable defaults
    default_intrinsics = np.array([
        [1000.0, 0.0, 960.0],    # fx, 0, cx
        [0.0, 1000.0, 540.0],    # 0, fy, cy  
        [0.0, 0.0, 1.0]          # 0, 0, 1
    ], dtype=np.float32)
    
    # Format camera poses for MAST3R input compatibility
    camera_poses_formatted = {
        'cameras': [
            {
                'name': 'frame_1',
                'position': cams2world_1[:3, 3].tolist(),
                'rotation_matrix': cams2world_1[:3, :3].tolist(),
                'matrix_world': cams2world_1.tolist(),
                'intrinsics': default_intrinsics.tolist(),
                'image_size': [1920, 1080]  # width, height
            },
            {
                'name': 'frame_2', 
                'position': cams2world_2[:3, 3].tolist(),
                'rotation_matrix': cams2world_2[:3, :3].tolist(),
                'matrix_world': cams2world_2.tolist(),
                'intrinsics': default_intrinsics.tolist(),
                'image_size': [1920, 1080]  # width, height
            }
        ]
    }
    
    return {
        'cameras': {
            'cams2world_1': cams2world_1,
            'cams2world_2': cams2world_2,
            'valid': camera_data_valid,
            'formatted': camera_poses_formatted
        },
        'point_clouds': processed_clouds,
        'matches': {
            'matches_im0': raw_data['matches_im0'],
            'matches_im1': raw_data['matches_im1'],
            'num_matches': raw_data['num_matches']
        },
        'views': {
            'view1': raw_data['view1'],
            'view2': raw_data['view2']
        }
    }
