#!/usr/bin/env python3
"""
Triangulation script for MAST3R matches using OpenCV triangulatePoints.

This script reads the matches from MAST3R output and the converted calibration
parameters to triangulate 3D points and export them as a PLY file.

Usage:
    python triangulate_matches.py <matches_file> <calibration_file> <output_ply> [d]
    python triangulate_matches.py /mast3r/tmp/mast3r_output/matches /mast3r/tmp/mast3r_output/extrinsics_calibration_512x288.json /mast3r/tmp/mast3r_output/triangulated_points.ply 5.0
    # Alternatively, set env TRIANGULATE_D to apply bounds filtering on x,y,z in [-d, d]
    # Default: d=10.0 if not provided
"""

import json
import numpy as np
import cv2
import sys
import os
from loguru import logger


def load_calibration_data(calibration_file):
    """
    Load camera calibration data from JSON file.
    
    Args:
        calibration_file: Path to calibration JSON file
    
    Returns:
        Dictionary with camera matrices, distortion coefficients, and extrinsics
    """
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    # Extract camera matrices
    K1 = np.array(calib_data['camera_matrix1'], dtype=np.float32)
    K2 = np.array(calib_data['camera_matrix2'], dtype=np.float32)
    
    # Extract distortion coefficients
    dist1 = np.array(calib_data['dist_coeffs1'][0], dtype=np.float32)
    dist2 = np.array(calib_data['dist_coeffs2'][0], dtype=np.float32)
    
    # Extract extrinsics
    R = np.array(calib_data['rotation_matrix'], dtype=np.float32)
    t = np.array(calib_data['translation_vector'], dtype=np.float32).flatten()
    
    # Get image size
    image_size = tuple(calib_data['image_size'])
    
    logger.info(f"Loaded calibration data:")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Camera 1 matrix:\n{K1}")
    logger.info(f"  Camera 2 matrix:\n{K2}")
    logger.info(f"  Rotation matrix:\n{R}")
    logger.info(f"  Translation vector: {t}")
    
    return {
        'K1': K1, 'K2': K2,
        'dist1': dist1, 'dist2': dist2,
        'R': R, 't': t,
        'image_size': image_size
    }


def load_matches(matches_file):
    """
    Load pixel correspondences from MAST3R matches.
    
    Args:
        matches_file: File containing matches JSON files
    
    Returns:
        Tuple of (points1, points2) as numpy arrays
    """
    pixel_pairs_file = matches_file
    
    if not os.path.exists(pixel_pairs_file):
        raise FileNotFoundError(f"Pixel pairs file not found: {pixel_pairs_file}")
    
    with open(pixel_pairs_file, 'r') as f:
        matches_data = json.load(f)
    
    pixel_pairs = matches_data['pixel_pairs']
    
    # Extract points from pixel pairs
    points1 = []
    points2 = []
    
    for pair in pixel_pairs:
        # Skip invalid points (0,0) which might be padding
        if pair['image1_pixel'] == [0.0, 0.0] and pair['image2_pixel'] == [0.0, 0.0]:
            continue
            
        points1.append(pair['image1_pixel'])
        points2.append(pair['image2_pixel'])
    
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)
    
    logger.info(f"Loaded {len(points1)} valid point correspondences")
    logger.info(f"Points1 shape: {points1.shape}")
    logger.info(f"Points2 shape: {points2.shape}")
    
    return points1, points2


def triangulate_points(points1, points2, calib_data):
    """
    Triangulate 3D points from 2D correspondences using OpenCV.
    
    Args:
        points1: 2D points in image 1 (Nx2)
        points2: 2D points in image 2 (Nx2)
        calib_data: Camera calibration data
    
    Returns:
        3D points (Nx3) in homogeneous coordinates
    """
    K1, K2 = calib_data['K1'], calib_data['K2']
    dist1, dist2 = calib_data['dist1'], calib_data['dist2']
    R, t = calib_data['R'], calib_data['t']
    
    # Undistort points
    points1_undist = cv2.undistortPoints(points1.reshape(-1, 1, 2), K1, dist1, P=K1)
    points2_undist = cv2.undistortPoints(points2.reshape(-1, 1, 2), K2, dist2, P=K2)
    
    # Reshape back to Nx2
    points1_undist = points1_undist.reshape(-1, 2)
    points2_undist = points2_undist.reshape(-1, 2)
    
    # Create projection matrices for triangulation
    # Camera 1 is at origin (identity rotation, zero translation)
    # P1 = K1 [I|0] where I is 3x3 identity, 0 is 3x1 zero vector
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    # Camera 2 uses the world-to-camera transforms from stereo calibration
    # P2 = K2 [R|t] where R and t are from cv2.stereoCalibrate
    # This assumes camera1's frame is "world" for projection purposes
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, points1_undist.T, points2_undist.T)
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T  # Transpose to get Nx3
    
    logger.info(f"Triangulated {len(points_3d)} 3D points")
    logger.info(f"3D points shape: {points_3d.shape}")
    logger.info(f"3D points range: X[{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}], "
                f"Y[{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}], "
                f"Z[{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")
    
    return points_3d


def extract_colors_from_image(points1, image_path):
    """
    Extract colors from image for given 2D points.
    
    Args:
        points1: 2D points in image 1 (Nx2)
        image_path: Path to the image file
    
    Returns:
        colors: RGB colors (Nx3) for each point
    """
    import cv2
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    colors = []
    for point in points1:
        x, y = point
        
        # Clamp coordinates to image bounds
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        
        # Extract color at the point
        color = image_rgb[y, x]  # Note: OpenCV uses (y, x) indexing
        colors.append(color)
    
    logger.info(f"Extracted colors for {len(colors)} points from {image_path}")
    return np.array(colors)


def write_ply_file(points_3d, output_file, colors=None):
    """
    Write 3D points to PLY file.
    
    Args:
        points_3d: 3D points (Nx3)
        output_file: Output PLY file path
        colors: Optional colors (Nx3) in RGB format
    """
    num_points = len(points_3d)
    
    with open(output_file, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Write points
        for i in range(num_points):
            x, y, z = points_3d[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    logger.info(f"Saved {num_points} points to PLY file: {output_file}")


def main():
    """Main function to triangulate matches and export PLY."""
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage: python triangulate_matches.py <matches_file> <calibration_file> <output_ply> <resized_frame_path> [d]")
        print("Example: python triangulate_matches.py /mast3r/tmp/mast3r_output/matches /mast3r/tmp/mast3r_output/extrinsics_calibration_512x288.json /mast3r/tmp/mast3r_output/triangulated_points.ply /mast3r/tmp/mast3r_output/resized_frame/frame_1.png 5.0")
        sys.exit(1)
    
    matches_file = sys.argv[1]
    calibration_file = sys.argv[2]
    output_ply = sys.argv[3]
    resized_frame_path = sys.argv[4]
    d = 10.0  # default bounds
    if len(sys.argv) == 6:
        try:
            d = float(sys.argv[5])
        except ValueError:
            logger.error(f"Invalid d value: {sys.argv[5]}")
            sys.exit(1)
    else:
        d_env = os.environ.get('TRIANGULATE_D')
        if d_env is not None:
            try:
                d = float(d_env)
            except ValueError:
                logger.warning(f"Ignoring invalid TRIANGULATE_D env value: {d_env}")
    
    # Check if input files exist
    if not os.path.exists(matches_file):
        logger.error(f"Matches file not found: {matches_file}")
        sys.exit(1)
    
    if not os.path.exists(calibration_file):
        logger.error(f"Calibration file not found: {calibration_file}")
        sys.exit(1)
    
    if not os.path.exists(resized_frame_path):
        logger.error(f"Resized frame file not found: {resized_frame_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_ply)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Load calibration data
        logger.info("Loading calibration data...")
        calib_data = load_calibration_data(calibration_file)
        
        # Load matches
        logger.info("Loading matches...")
        points1, points2 = load_matches(matches_file)
        
        if len(points1) == 0:
            logger.error("No valid matches found!")
            sys.exit(1)
        
        # Triangulate 3D points
        logger.info("Triangulating 3D points...")
        points_3d = triangulate_points(points1, points2, calib_data)
        
        logger.info(f"Triangulated {len(points_3d)} 3D points")
        
        if len(points_3d) == 0:
            logger.error("No valid 3D points after filtering!")
            sys.exit(1)
        
        # Bounds filtering (vectorized): keep points with x,y,z in [-d, d]
        logger.info(f"Applying XYZ bounds filter with d={d}")
        mask = np.all(np.abs(points_3d) <= d, axis=1)
        kept = int(mask.sum())
        logger.info(f"Keeping {kept}/{len(points_3d)} points within [-{d}, {d}] on all axes")
        if kept == 0:
            logger.error("All points filtered out by bounds; aborting.")
            sys.exit(1)
        points_3d = points_3d[mask]
        # Keep points1 synchronized for color extraction
        points1 = points1[mask]
        
        # Extract colors from resized frame_1
        logger.info("Extracting colors from resized frame_1...")
        colors = extract_colors_from_image(points1, resized_frame_path)
        if colors is not None:
            logger.info(f"Successfully extracted colors for {len(colors)} points")
        else:
            logger.error("Failed to extract colors from resized frame")
            sys.exit(1)

        
        # Write PLY file with colors
        logger.info("Writing colored PLY file...")
        write_ply_file(points_3d, output_ply, colors)
        
        logger.info("Triangulation completed successfully!")
        logger.info(f"Output PLY file: {output_ply}")
        logger.info(f"Total 3D points: {len(points_3d)}")
        
    except Exception as e:
        logger.error(f"Error during triangulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
