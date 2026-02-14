#!/usr/bin/env python3
"""
Utility functions for intrinsics conversion tests.
Handles visualization, image generation, and result saving.
"""

import os
import json
import numpy as np
import PIL.Image
import PIL.ImageDraw
from datetime import datetime


def save_json_results(output_path, data):
    """Save test results as JSON file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_synthetic_checkerboard(width, height, square_size=120):
    """
    Generate a synthetic checkerboard image with known corner positions.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        square_size: Size of each checkerboard square in pixels

    Returns:
        tuple: (PIL.Image, list of corner coordinates)
    """
    rows = height // square_size
    cols = width // square_size

    # Create image array
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill checkerboard pattern
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y_start = i * square_size
                y_end = min((i + 1) * square_size, height)
                x_start = j * square_size
                x_end = min((j + 1) * square_size, width)
                img_array[y_start:y_end, x_start:x_end] = 255

    # Generate corner coordinates (intersections of checkerboard)
    corners = []
    for i in range(1, rows):
        for j in range(1, cols):
            x = j * square_size
            y = i * square_size
            corners.append((x, y))

    # Convert to PIL Image
    pil_image = PIL.Image.fromarray(img_array)

    return pil_image, corners


def generate_3d_test_points_random(
    num_points=45, x_range=(-2000, 2000), y_range=(-1500, 1500), z_range=(500, 2000)
):
    """
    Generate random 3D test points without using any camera matrix.
    This provides unbiased validation of the intrinsics conversion.

    Args:
        num_points: Number of 3D points to generate
        x_range: (min_x, max_x) range in mm
        y_range: (min_y, max_y) range in mm
        z_range: (min_z, max_z) range in mm

    Returns:
        Nx3 array of 3D points
    """
    np.random.seed(42)  # For reproducible results

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    # Generate random points
    X = np.random.uniform(x_min, x_max, num_points)
    Y = np.random.uniform(y_min, y_max, num_points)
    Z = np.random.uniform(z_min, z_max, num_points)

    points_3d = np.column_stack([X, Y, Z])

    return points_3d


def create_projection_visualization(
    points_2d_ground_truth, points_2d_predicted, image_size
):
    """
    Create visualization comparing ground truth and predicted projections.

    Args:
        points_2d_ground_truth: Nx2 array of ground truth 2D points
        points_2d_predicted: Nx2 array of predicted 2D points
        image_size: (width, height) tuple

    Returns:
        PIL.Image: Visualization image
    """
    width, height = image_size
    viz_img = PIL.Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = PIL.ImageDraw.Draw(viz_img)

    # Draw all valid points
    for gt, pred in zip(points_2d_ground_truth, points_2d_predicted):
        # Ground truth in green (larger circle)
        draw.ellipse(
            [gt[0] - 3, gt[1] - 3, gt[0] + 3, gt[1] + 3],
            fill=(0, 255, 0),
            outline=(0, 128, 0),
        )
        # Predicted in red (smaller circle, should overlap)
        draw.ellipse(
            [pred[0] - 2, pred[1] - 2, pred[0] + 2, pred[1] + 2],
            fill=(255, 0, 0),
            outline=(128, 0, 0),
        )

    return viz_img


def create_corner_visualization(image, corners, image_size):
    """
    Create visualization of checkerboard with corners marked.

    Args:
        image: PIL.Image to annotate
        corners: Nx2 array of corner coordinates
        image_size: (width, height) tuple

    Returns:
        PIL.Image: Annotated image
    """
    viz_img = image.copy()
    draw = PIL.ImageDraw.Draw(viz_img)

    # Draw all valid corners
    for corner in corners:
        x, y = corner[0], corner[1]
        # Draw cross at corner
        draw.line([x - 5, y, x + 5, y], fill=(255, 0, 0), width=2)
        draw.line([x, y - 5, x, y + 5], fill=(255, 0, 0), width=2)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))

    return viz_img


def format_projections_for_json(
    points_2d_ground_truth, points_2d_predicted, pixel_errors
):
    """
    Format all projection data for JSON export.

    Args:
        points_2d_ground_truth: Nx2 array of ground truth points
        points_2d_predicted: Nx2 array of predicted points
        pixel_errors: N array of pixel errors

    Returns:
        list: List of dictionaries with projection data
    """
    projections = []
    for i in range(len(points_2d_ground_truth)):
        gt = points_2d_ground_truth[i]
        pred = points_2d_predicted[i]
        error = pixel_errors[i]
        projections.append(
            {
                "point_index": int(i),
                "ground_truth": [float(gt[0]), float(gt[1])],
                "predicted": [float(pred[0]), float(pred[1])],
                "error_px": float(error),
            }
        )
    return projections


def format_corners_for_json(corners_original, corners_transformed, valid_mask):
    """
    Format corner tracking data for JSON export.

    Args:
        corners_original: Nx2 array of original corners
        corners_transformed: Nx2 array of transformed corners
        valid_mask: N boolean array of valid corners

    Returns:
        list: List of dictionaries with corner data
    """
    corners_data = []
    valid_indices = np.where(valid_mask)[0]
    corners_valid = corners_transformed[valid_mask]

    for i, idx in enumerate(valid_indices):
        orig = corners_original[idx]
        trans = corners_valid[i]
        corners_data.append(
            {
                "corner_index": int(i),
                "original_index": int(idx),
                "original": [float(orig[0]), float(orig[1])],
                "resized": [float(trans[0]), float(trans[1])],
            }
        )

    return corners_data


def create_accuracy_test_results(
    test_name,
    camera_name,
    original_size,
    target_size,
    patch_size,
    camera_matrix_original,
    camera_matrix_resized,
    transform_params,
    points_2d_ground_truth_valid,
    points_2d_predicted_valid,
    pixel_errors,
    total_points_generated,
    point_generation_method="grid",
):
    """
    Create comprehensive accuracy test results dictionary.

    Args:
        point_generation_method: "grid" or "random" to indicate how points were generated

    Returns:
        dict: Structured test results
    """
    max_error = np.max(pixel_errors)
    mean_error = np.mean(pixel_errors)
    median_error = np.median(pixel_errors)

    # Get all projections (not just samples)
    all_projections = format_projections_for_json(
        points_2d_ground_truth_valid, points_2d_predicted_valid, pixel_errors
    )

    results = {
        "test_name": test_name,
        "camera": camera_name,
        "point_generation_method": point_generation_method,
        "timestamp": datetime.now().isoformat(),
        "original_size": list(original_size),
        "target_size": target_size,
        "patch_size": patch_size,
        "camera_matrix_original": camera_matrix_original.tolist(),
        "camera_matrix_resized": camera_matrix_resized.tolist(),
        "transform_params": {
            "scale_factor": float(transform_params["scale_factor"]),
            "resized_size": list(transform_params["resized_size"]),
            "crop_offset": list(transform_params["crop_offset"]),
            "final_size": list(transform_params["final_size"]),
        },
        "test_points": {
            "total_generated": int(total_points_generated),
            "valid_in_resized": int(len(points_2d_ground_truth_valid)),
        },
        "errors": {
            "max_error_px": float(max_error),
            "mean_error_px": float(mean_error),
            "median_error_px": float(median_error),
            "threshold_px": 0.5,
            "passed": bool(float(max_error) < 0.5),
        },
        "all_projections": all_projections,  # All points, not just samples
    }

    return results


def print_sample_projections(
    points_2d_ground_truth, points_2d_predicted, pixel_errors, num_samples=5
):
    """Print sample projections to console for verification."""
    print(f"\nSample projections (first {num_samples} valid points):")
    for i in range(min(num_samples, len(points_2d_ground_truth))):
        gt = points_2d_ground_truth[i]
        pred = points_2d_predicted[i]
        error = pixel_errors[i]
        print(
            f"  Point {i}: GT=({gt[0]:.2f}, {gt[1]:.2f}), "
            f"Pred=({pred[0]:.2f}, {pred[1]:.2f}), Error={error:.4f}px"
        )


def print_sample_corners(
    corners_original, corners_transformed, valid_mask, num_samples=5
):
    """Print sample corner transformations to console."""
    print(f"\nSample corner transformations (first {num_samples}):")
    corners_valid = corners_transformed[valid_mask]

    for i in range(min(num_samples, len(corners_valid))):
        original_idx = np.where(valid_mask)[0][i]
        orig = corners_original[original_idx]
        trans = corners_valid[i]
        print(
            f"  Corner {i}: Original=({orig[0]:.1f}, {orig[1]:.1f}) "
            f"→ Resized=({trans[0]:.2f}, {trans[1]:.2f})"
        )
