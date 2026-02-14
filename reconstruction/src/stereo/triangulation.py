import json

import cv2
import numpy as np
from loguru import logger


def load_calibration_data(calibration_file):
    """
    Load stereo camera calibration data from JSON file.

    Args:
        calibration_file: Path to calibration JSON file

    Returns:
        Dictionary with camera matrices, distortion coefficients, and extrinsics
    """
    with open(calibration_file) as f:
        calib_data = json.load(f)

    # Extract camera matrices
    K1 = np.array(calib_data["camera_matrix1"], dtype=np.float32)
    K2 = np.array(calib_data["camera_matrix2"], dtype=np.float32)

    # Extract distortion coefficients
    dist1 = np.array(calib_data["dist_coeffs1"][0], dtype=np.float32)
    dist2 = np.array(calib_data["dist_coeffs2"][0], dtype=np.float32)

    # Extract extrinsics
    R = np.array(calib_data["rotation_matrix"], dtype=np.float32)
    t = np.array(calib_data["translation_vector"], dtype=np.float32).flatten()

    # Get image size
    image_size = tuple(calib_data["image_size"])

    logger.info("Loaded calibration data:")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Camera 1 matrix:\n{K1}")
    logger.info(f"  Camera 2 matrix:\n{K2}")
    logger.info(f"  Rotation matrix:\n{R}")
    logger.info(f"  Translation vector: {t}")

    return {
        "K1": K1,
        "K2": K2,
        "dist1": dist1,
        "dist2": dist2,
        "R": R,
        "t": t,
        "image_size": image_size,
    }


def triangulate_points(points1, points2, calib_data):
    """
    Triangulate 3D points from 2D correspondences using OpenCV.

    Args:
        points1: 2D points in image 1 (Nx2)
        points2: 2D points in image 2 (Nx2)
        calib_data: Camera calibration data from stereo calibration

    Returns:
        3D points (Nx3) in world coordinates
    """

    # Convert points to numpy arrays and ensure correct data type
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    K1, K2 = (
        np.array(calib_data["camera_matrix1"]),
        np.array(calib_data["camera_matrix2"]),
    )
    dist1, dist2 = (
        np.array(calib_data["dist_coeffs1"]),
        np.array(calib_data["dist_coeffs2"]),
    )
    R, t = (
        np.array(calib_data["rotation_matrix"]),
        np.array(calib_data["translation_vector"]),
    )

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

    logger.debug(f"Triangulated {len(points_3d)} 3D points")
    logger.debug(f"3D points shape: {points_3d.shape}")
    logger.debug(
        f"3D points range: X[{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}], "
        f"Y[{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}], "
        f"Z[{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]"
    )

    return points_3d


def extract_colors_from_image(points1, image):
    """
    Extract RGB colors from image at given 2D point locations.

    Args:
        points1: 2D points in image 1 (Nx2)
        image: RGB image in [0, 255] range

    Returns:
        RGB colors (Nx3) for each point in [0, 255] range
    """

    height, width = image.shape[:2]

    colors = []
    for point in points1:
        x, y = point

        # Clamp coordinates to image bounds
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))

        # Extract color at the point
        color = image[y, x]
        colors.append(color)

    colors_array = np.array(colors)

    logger.debug(
        f"Extracted colors, range: [{colors_array.min()}, {colors_array.max()}]"
    )
    return colors_array


def filter_pairs_with_mask(points1, points2, mask, camera=1):
    """
    Filter point pairs based on segmentation mask.

    Args:
        points1: 2D points in image 1 (Nx2)
        points2: 2D points in image 2 (Nx2)
        mask: Segmentation mask (HxW) with values > 0 indicating valid regions
        camera: Camera number (1 or 2) to determine which points to filter

    Returns:
        tuple: (filtered_points1, filtered_points2) containing only points inside mask
    """
    if mask is None:
        logger.warning("No mask provided, returning all points")
        return points1, points2

    # Select which points to check based on camera argument
    if camera == 1:
        points_to_check = points1
    elif camera == 2:
        points_to_check = points2
    else:
        raise ValueError("Camera argument must be 1 or 2")

    # Convert points to integers
    points_int = points_to_check.astype(int)

    # Ensure coordinates are within mask bounds
    points_int[:, 0] = np.clip(points_int[:, 0], 0, mask.shape[1] - 1)
    points_int[:, 1] = np.clip(points_int[:, 1], 0, mask.shape[0] - 1)

    # Check which points are inside the mask
    mask_values = mask[points_int[:, 1], points_int[:, 0]]
    inside_mask = mask_values > 0

    # Filter both point sets
    filtered_points1 = points1[inside_mask]
    filtered_points2 = points2[inside_mask]

    logger.debug(
        f"Filtered pairs (camera {camera}): {len(filtered_points1)} out of {len(points1)} points inside mask"
    )

    return filtered_points1, filtered_points2
