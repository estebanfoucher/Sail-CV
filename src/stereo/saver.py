#!/usr/bin/env python3


import json

import numpy as np
from loguru import logger


def save_point_cloud_ply(
    points: np.ndarray, colors: np.ndarray, filename: str, bound_distance: float = 20
) -> None:
    """
    Save point cloud as PLY file.

    Args:
        points: 3D points array of shape (H, W, 3) or (N, 3)
        colors: Color array of shape (H, W, 3) or (N, 3) RGB [0, 255] range
        filename: Output PLY file path
        bound_distance: Keep only points with distance < bound_distance from origin
    """
    # Flatten the points and colors
    points_flat = points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)

    # Filter out invalid points (NaN, inf)
    valid_mask = np.isfinite(points_flat).all(axis=1)

    distance_mask = np.linalg.norm(points_flat, axis=1) < bound_distance
    valid_mask = valid_mask & distance_mask

    points_valid = points_flat[valid_mask]
    colors_valid = colors_flat[valid_mask]

    # Write PLY file
    with open(filename, "w") as f:
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
            f.write(
                f"{points_valid[i, 0]:.6f} {points_valid[i, 1]:.6f} {points_valid[i, 2]:.6f} "
                f"{colors_valid[i, 0]} {colors_valid[i, 1]} {colors_valid[i, 2]}\n"
            )

    logger.debug(
        f"Point cloud saved to {filename} with {len(points_valid)} points (filtered by distance < {bound_distance})"
    )


def save_pixel_pairs(
    matches_im0: np.ndarray, matches_im1: np.ndarray, filename: str
) -> None:
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
            "id": i,
            "image1_pixel": [float(matches_im0[i, 0]), float(matches_im0[i, 1])],
            "image2_pixel": [float(matches_im1[i, 0]), float(matches_im1[i, 1])],
        }
        pixel_pairs.append(pair)

    # Save as JSON
    with open(filename, "w") as f:
        json.dump(
            {
                "pixel_pairs": pixel_pairs,
                "num_pairs": len(pixel_pairs),
                "format": "dense_matching_results",
                "description": "Pixel correspondences between two images from MASt3R dense matching",
            },
            f,
            indent=2,
        )

    logger.debug(f"Pixel pairs saved to {filename}")
