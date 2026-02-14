#!/usr/bin/env python3


import json
from pathlib import Path

import cv2
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


def save_point_cloud_obj(
    points: np.ndarray, colors: np.ndarray, filename: str, bound_distance: float = 20
) -> None:
    """
    Save point cloud as OBJ file.

    Args:
        points: 3D points array of shape (H, W, 3) or (N, 3)
        colors: Color array of shape (H, W, 3) or (N, 3) RGB [0, 255] range
        filename: Output OBJ file path
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

    # Write OBJ file
    with open(filename, "w") as f:
        f.write("# OBJ file converted from point cloud\n")
        f.write("# Point cloud representation\n")
        f.write(f"# {len(points_valid)} points\n")

        # Write vertices with colors (OBJ format supports vertex colors)
        for point, color in zip(points_valid, colors_valid, strict=False):
            # Normalize colors to [0, 1] range for OBJ format
            color_normalized = color / 255.0
            f.write(
                f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color_normalized[0]:.3f} {color_normalized[1]:.3f} {color_normalized[2]:.3f}\n"
            )

        # Write point references (each point as a vertex)
        for i in range(len(points_valid)):
            f.write(f"p {i + 1}\n")

    logger.debug(
        f"Point cloud saved to {filename} with {len(points_valid)} points (filtered by distance < {bound_distance})"
    )


def render_match_correspondences(
    image_1,
    image_2,
    matches_im0: np.ndarray,
    matches_im1: np.ndarray,
    output_path: str,
    density_factor: int = 32,
) -> None:
    """
    Render correspondence matches between two images side-by-side with lines connecting matching points.

    Args:
        image_1: PIL Image or numpy array (resized image, typically 512x512)
        image_2: PIL Image or numpy array (resized image, typically 512x512)
        matches_im0: numpy array of shape (N, 2) - pixel coordinates in image_1
        matches_im1: numpy array of shape (N, 2) - pixel coordinates in image_2
        output_path: Path to save the rendered image
        density_factor: Factor to reduce match density (default 8, shows 1/8th of matches)
    """
    # Convert PIL images to numpy arrays if needed
    img1_array = np.array(image_1) if hasattr(image_1, "size") else image_1.copy()
    img2_array = np.array(image_2) if hasattr(image_2, "size") else image_2.copy()

    # Ensure images are RGB (3 channels)
    if len(img1_array.shape) == 2:
        img1_array = cv2.cvtColor(img1_array, cv2.COLOR_GRAY2RGB)
    elif img1_array.shape[2] == 4:
        img1_array = cv2.cvtColor(img1_array, cv2.COLOR_RGBA2RGB)

    if len(img2_array.shape) == 2:
        img2_array = cv2.cvtColor(img2_array, cv2.COLOR_GRAY2RGB)
    elif img2_array.shape[2] == 4:
        img2_array = cv2.cvtColor(img2_array, cv2.COLOR_RGBA2RGB)

    # Get dimensions
    h1, w1 = img1_array.shape[:2]
    h2, w2 = img2_array.shape[:2]

    # Create side-by-side combined image
    separator_width = 5
    combined_width = w1 + w2 + separator_width
    combined_height = max(h1, h2)
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Place images side by side
    combined_image[:h1, :w1] = img1_array
    combined_image[:h2, w1 + separator_width : w1 + separator_width + w2] = img2_array

    # Draw separator line
    cv2.line(
        combined_image,
        (w1, 0),
        (w1, combined_height),
        (255, 255, 255),
        separator_width,
    )

    # Draw correspondences
    original_num_matches = len(matches_im0) if len(matches_im0) > 0 else 0
    rendered_num_matches = 0

    if len(matches_im0) > 0 and len(matches_im1) > 0:
        # Subsample matches to reduce density
        matches_im0_subsampled = matches_im0[::density_factor]
        matches_im1_subsampled = matches_im1[::density_factor]
        rendered_num_matches = len(matches_im0_subsampled)

        # Generate colors for matches (use HSV color space for better color distribution)
        num_matches = len(matches_im0_subsampled)
        colors = []
        for i in range(num_matches):
            hue = int(180 * i / max(num_matches, 1)) % 180
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))

        # Draw lines and points for each match
        line_thickness = 1
        point_radius = 2

        for i in range(num_matches):
            color = colors[i]

            # Point in image 1 (left side)
            pt1 = tuple(matches_im0_subsampled[i].astype(int))

            # Point in image 2 (right side, offset by width of image 1 + separator)
            pt2 = (
                int(matches_im1_subsampled[i][0]) + w1 + separator_width,
                int(matches_im1_subsampled[i][1]),
            )

            # Draw line connecting the points
            cv2.line(combined_image, pt1, pt2, color, line_thickness)

            # Draw points
            cv2.circle(combined_image, pt1, point_radius, color, -1)
            cv2.circle(combined_image, pt2, point_radius, color, -1)

    # Save the combined image
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    logger.debug(
        f"Match correspondences rendered to {output_path} with {rendered_num_matches} matches "
        f"(subsampled from {original_num_matches} by factor {density_factor})"
    )
