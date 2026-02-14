#!/usr/bin/env python3
"""
Test suite for validating camera intrinsics conversion during image resize/crop.
This test ensures that K_resized correctly matches the transformation applied to images.

Results are saved to output_tests/ directory as JSON and PNG files for inspection.
"""

import os
import json
import numpy as np
import pytest
import PIL.Image
from pathlib import Path

# Project root (pythonpath configured in pyproject.toml)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from stereo.image import resize_image
from stereo.convert_calibration import (
    calculate_resize_and_crop_params,
    transform_camera_matrix,
)
from .test_intrinsics_conversion_utils import (
    save_json_results,
    generate_synthetic_checkerboard,
    generate_3d_test_points_random,
    create_projection_visualization,
    create_corner_visualization,
    format_projections_for_json,
    format_corners_for_json,
    create_accuracy_test_results,
    print_sample_projections,
    print_sample_corners,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def output_dir():
    """Create and return output directory for test results"""
    output_path = os.path.join(project_root, "output_tests", "intrinsics_conversion")
    os.makedirs(output_path, exist_ok=True)
    return output_path


@pytest.fixture
def original_image_size():
    """Original image size (width, height)"""
    return (1920, 1080)


@pytest.fixture
def target_size():
    """Target size for MASt3R preprocessing"""
    return 512


@pytest.fixture
def patch_size():
    """Patch size for alignment"""
    return 16


@pytest.fixture
def intrinsics_1_1():
    """Load intrinsics for camera 1_1"""
    intrinsics_path = os.path.join(
        project_root, "assets", "reconstruction", "intrinsics", "intrinsics_1_1.json"
    )
    with open(intrinsics_path, "r") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"])


@pytest.fixture
def intrinsics_1_2():
    """Load intrinsics for camera 1_2"""
    intrinsics_path = os.path.join(
        project_root, "assets", "reconstruction", "intrinsics", "intrinsics_1_2.json"
    )
    with open(intrinsics_path, "r") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"])


@pytest.fixture
def synthetic_checkerboard(original_image_size):
    """Generate a synthetic checkerboard image with known corner positions."""
    width, height = original_image_size
    return generate_synthetic_checkerboard(width, height)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def project_3d_to_2d(points_3d, camera_matrix):
    """
    Project 3D points to 2D image coordinates using camera matrix.

    Args:
        points_3d: Nx3 array of 3D points
        camera_matrix: 3x3 camera intrinsics matrix

    Returns:
        Nx2 array of 2D pixel coordinates
    """
    # Homogeneous projection: [x, y, z] -> [u*z, v*z, z]
    points_2d_homogeneous = camera_matrix @ points_3d.T  # 3xN

    # Normalize by z coordinate
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

    return points_2d.T  # Nx2


def apply_resize_crop_transform(points_2d, transform_params):
    """
    Apply the same resize and crop transformation that is applied to images.

    Args:
        points_2d: Nx2 array of 2D coordinates in original space
        transform_params: Dictionary from calculate_resize_and_crop_params

    Returns:
        Nx2 array of 2D coordinates in resized/cropped space
    """
    scale_factor = transform_params["scale_factor"]
    crop_offset = transform_params["crop_offset"]

    # Scale coordinates
    points_scaled = points_2d * scale_factor

    # Apply crop offset
    points_cropped = points_scaled - np.array(crop_offset)

    return points_cropped


def generate_3d_test_points(
    camera_matrix, original_size, z_depth=1000.0, grid_spacing=200
):
    """
    Generate 3D test points that project into the original image space.

    Args:
        camera_matrix: 3x3 camera intrinsics
        original_size: (width, height) of original image
        z_depth: Depth of the plane in mm
        grid_spacing: Spacing between points in pixels

    Returns:
        Nx3 array of 3D points
    """
    width, height = original_size

    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Generate grid of pixel coordinates
    pixel_coords = []
    for u in range(grid_spacing, width, grid_spacing):
        for v in range(grid_spacing, height, grid_spacing):
            pixel_coords.append([u, v])

    pixel_coords = np.array(pixel_coords)

    # Back-project to 3D at depth z_depth
    # Formula: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    X = (pixel_coords[:, 0] - cx) * z_depth / fx
    Y = (pixel_coords[:, 1] - cy) * z_depth / fy
    Z = np.full(len(pixel_coords), z_depth)

    points_3d = np.column_stack([X, Y, Z])

    return points_3d


# ============================================================================
# TESTS
# ============================================================================


def test_resize_image_dimensions(
    synthetic_checkerboard, target_size, patch_size, output_dir
):
    """Test that resize_image produces expected dimensions."""
    pil_image, _ = synthetic_checkerboard

    # Apply resize
    resized_image = resize_image(pil_image, size=target_size, patch_size=patch_size)

    # Get dimensions
    width, height = resized_image.size

    # Verify dimensions are multiples of patch_size
    assert width % patch_size == 0, f"Width {width} should be multiple of {patch_size}"
    assert height % patch_size == 0, (
        f"Height {height} should be multiple of {patch_size}"
    )

    # Save results
    results = {
        "test_name": "test_resize_image_dimensions",
        "target_size": target_size,
        "patch_size": patch_size,
        "output_width": width,
        "output_height": height,
        "width_divisible_by_patch": width % patch_size == 0,
        "height_divisible_by_patch": height % patch_size == 0,
    }

    # Save JSON
    save_json_results(os.path.join(output_dir, "test_resize_dimensions.json"), results)

    # Save images
    pil_image.save(os.path.join(output_dir, "checkerboard_original.png"))
    resized_image.save(os.path.join(output_dir, "checkerboard_resized.png"))

    print(f"✅ Resized image dimensions: {width}x{height}")
    print(f"📁 Saved results to: {output_dir}")


def test_transform_params_match_resize(
    synthetic_checkerboard, original_image_size, target_size, patch_size, output_dir
):
    """
    Test that calculate_resize_and_crop_params produces parameters
    that match the actual resize_image output.
    """
    pil_image, _ = synthetic_checkerboard

    # Calculate transformation parameters
    transform_params = calculate_resize_and_crop_params(
        original_image_size, target_size, patch_size
    )

    # Apply actual resize
    resized_image = resize_image(pil_image, size=target_size, patch_size=patch_size)

    # Compare dimensions
    actual_size = resized_image.size  # (width, height)
    expected_size = transform_params["final_size"]  # (width, height)

    assert actual_size == expected_size, (
        f"Actual size {actual_size} should match expected {expected_size}"
    )

    # Save results
    results = {
        "test_name": "test_transform_params_match_resize",
        "original_size": list(original_image_size),
        "target_size": target_size,
        "patch_size": patch_size,
        "transform_params": {
            "scale_factor": transform_params["scale_factor"],
            "resized_size": list(transform_params["resized_size"]),
            "crop_offset": list(transform_params["crop_offset"]),
            "final_size": list(transform_params["final_size"]),
            "crop_coords": list(transform_params["crop_coords"]),
        },
        "actual_output_size": list(actual_size),
        "sizes_match": actual_size == expected_size,
    }

    save_json_results(os.path.join(output_dir, "test_transform_params.json"), results)

    print(f"✅ Transform params match resize: {actual_size}")
    print(f"📁 Saved results to: {output_dir}")


@pytest.mark.parametrize("intrinsics_fixture", ["intrinsics_1_1", "intrinsics_1_2"])
@pytest.mark.parametrize("point_method", ["grid", "random"])
def test_intrinsics_conversion_accuracy(
    intrinsics_fixture,
    point_method,
    original_image_size,
    target_size,
    patch_size,
    output_dir,
    request,
):
    """
    Main test: Validate that K_resized correctly transforms 3D->2D projections.

    Tests both:
    1. Grid-based points (biased - uses camera matrix for back-projection)
    2. Random points (unbiased - independent of camera matrix)

    This test:
    1. Generates 3D test points (grid or random)
    2. Projects them using K_original → gets pixel coords in original space
    3. Applies resize/crop transform → gets "ground truth" coords in resized space
    4. Projects them using K_resized → gets predicted coords in resized space
    5. Compares ground truth vs predicted (should be pixel-perfect)
    """
    # Get intrinsics from fixture
    camera_matrix = request.getfixturevalue(intrinsics_fixture)

    print(f"\n{'=' * 60}")
    print(f"Testing {intrinsics_fixture} with {point_method} points")
    print(f"{'=' * 60}")

    # Calculate transformation parameters
    transform_params = calculate_resize_and_crop_params(
        original_image_size, target_size, patch_size
    )

    print(f"Transform params:")
    print(f"  Scale factor: {transform_params['scale_factor']:.6f}")
    print(f"  Resized size: {transform_params['resized_size']}")
    print(f"  Crop offset: {transform_params['crop_offset']}")
    print(f"  Final size: {transform_params['final_size']}")

    # Generate 3D test points based on method
    if point_method == "grid":
        points_3d = generate_3d_test_points(
            camera_matrix, original_image_size, z_depth=1000.0, grid_spacing=200
        )
    else:  # random
        points_3d = generate_3d_test_points_random(
            num_points=45,
            x_range=(-2000, 2000),
            y_range=(-1500, 1500),
            z_range=(500, 2000),
        )

    print(f"\nGenerated {len(points_3d)} 3D test points using {point_method} method")

    # Method 1: Project with K_original, then apply transform (GROUND TRUTH)
    points_2d_original = project_3d_to_2d(points_3d, camera_matrix)
    points_2d_ground_truth = apply_resize_crop_transform(
        points_2d_original, transform_params
    )

    # Method 2: Project with K_resized (PREDICTED)
    camera_matrix_resized = transform_camera_matrix(camera_matrix, transform_params)
    points_2d_predicted = project_3d_to_2d(points_3d, camera_matrix_resized)

    # Filter points that fall within the resized image bounds
    final_width, final_height = transform_params["final_size"]

    valid_mask = (
        (points_2d_ground_truth[:, 0] >= 0)
        & (points_2d_ground_truth[:, 0] < final_width)
        & (points_2d_ground_truth[:, 1] >= 0)
        & (points_2d_ground_truth[:, 1] < final_height)
    )

    points_2d_ground_truth_valid = points_2d_ground_truth[valid_mask]
    points_2d_predicted_valid = points_2d_predicted[valid_mask]

    print(f"Valid points in resized image: {len(points_2d_ground_truth_valid)}")

    # Calculate pixel errors
    pixel_errors = np.linalg.norm(
        points_2d_ground_truth_valid - points_2d_predicted_valid, axis=1
    )

    max_error = np.max(pixel_errors)
    mean_error = np.mean(pixel_errors)
    median_error = np.median(pixel_errors)

    print(f"\nProjection errors:")
    print(f"  Max error: {max_error:.6f} px")
    print(f"  Mean error: {mean_error:.6f} px")
    print(f"  Median error: {median_error:.6f} px")

    # Show sample projections
    print_sample_projections(
        points_2d_ground_truth_valid, points_2d_predicted_valid, pixel_errors
    )

    # Create comprehensive results with ALL projections
    results = create_accuracy_test_results(
        "test_intrinsics_conversion_accuracy",
        intrinsics_fixture,
        original_image_size,
        target_size,
        patch_size,
        camera_matrix,
        camera_matrix_resized,
        transform_params,
        points_2d_ground_truth_valid,
        points_2d_predicted_valid,
        pixel_errors,
        len(points_3d),
        point_method,
    )

    # Save detailed results as JSON
    output_file = os.path.join(
        output_dir, f"test_accuracy_{intrinsics_fixture}_{point_method}.json"
    )
    save_json_results(output_file, results)

    # Create visualization image
    viz_img = create_projection_visualization(
        points_2d_ground_truth_valid,
        points_2d_predicted_valid,
        (final_width, final_height),
    )

    viz_file = os.path.join(
        output_dir, f"projections_{intrinsics_fixture}_{point_method}.png"
    )
    viz_img.save(viz_file)

    print(f"📁 Saved results to: {output_file}")
    print(f"📁 Saved visualization to: {viz_file}")

    # Assert pixel-perfect match (< 0.5 px error)
    assert max_error < 0.5, (
        f"Max projection error {max_error:.6f} px exceeds threshold of 0.5 px"
    )

    print(
        f"\n✅ PASSED: Intrinsics conversion is accurate for {intrinsics_fixture} with {point_method} points"
    )


def test_corner_tracking_through_resize(
    synthetic_checkerboard,
    intrinsics_1_1,
    original_image_size,
    target_size,
    patch_size,
    output_dir,
):
    """
    Test that checkerboard corners can be tracked through the resize operation.
    This validates that the transformation is consistent.
    """
    pil_image, corners_original = synthetic_checkerboard

    # Calculate transformation parameters
    transform_params = calculate_resize_and_crop_params(
        original_image_size, target_size, patch_size
    )

    # Apply resize to image
    resized_image = resize_image(pil_image, size=target_size, patch_size=patch_size)

    # Transform corners using transformation parameters
    corners_array = np.array(corners_original)
    corners_transformed = apply_resize_crop_transform(corners_array, transform_params)

    # Filter corners that are within resized image bounds
    final_width, final_height = transform_params["final_size"]
    valid_mask = (
        (corners_transformed[:, 0] >= 0)
        & (corners_transformed[:, 0] < final_width)
        & (corners_transformed[:, 1] >= 0)
        & (corners_transformed[:, 1] < final_height)
    )

    corners_valid = corners_transformed[valid_mask]

    print(f"\nCheckerboard corner tracking:")
    print(f"  Original corners: {len(corners_original)}")
    print(f"  Valid corners in resized image: {len(corners_valid)}")
    print(f"  Resized image size: {final_width}x{final_height}")

    # We should have some corners visible
    assert len(corners_valid) > 0, "No corners remain visible after resize"

    # Show sample corners
    print_sample_corners(corners_array, corners_transformed, valid_mask)

    # Format all corners for JSON
    all_corners = format_corners_for_json(
        corners_array, corners_transformed, valid_mask
    )

    # Save results
    results = {
        "test_name": "test_corner_tracking_through_resize",
        "original_image_size": list(original_image_size),
        "resized_image_size": [final_width, final_height],
        "corners": {
            "total_original": int(len(corners_original)),
            "valid_in_resized": int(len(corners_valid)),
        },
        "all_corners": all_corners,  # All corners, not just samples
    }

    output_file = os.path.join(output_dir, "test_corner_tracking.json")
    save_json_results(output_file, results)

    # Create visualization with corners marked
    resized_viz = create_corner_visualization(
        resized_image, corners_valid, (final_width, final_height)
    )

    viz_file = os.path.join(output_dir, "checkerboard_resized_with_corners.png")
    resized_viz.save(viz_file)

    print(f"📁 Saved results to: {output_file}")
    print(f"📁 Saved visualization to: {viz_file}")
    print(f"✅ Corner tracking test passed")


# ============================================================================
# SUMMARY TEST
# ============================================================================


def test_complete_pipeline_validation(
    intrinsics_1_1,
    intrinsics_1_2,
    original_image_size,
    target_size,
    patch_size,
    output_dir,
):
    """
    Comprehensive validation of the complete pipeline.
    This is the final fixture-ready test that confirms our conversion is correct.
    """
    print(f"\n{'=' * 60}")
    print("COMPLETE PIPELINE VALIDATION")
    print(f"{'=' * 60}")

    all_passed = True
    results = {}

    for name, camera_matrix in [
        ("intrinsics_1_1", intrinsics_1_1),
        ("intrinsics_1_2", intrinsics_1_2),
    ]:
        print(f"\nTesting {name}...")

        # Calculate transformation
        transform_params = calculate_resize_and_crop_params(
            original_image_size, target_size, patch_size
        )

        # Generate test points
        points_3d = generate_3d_test_points(
            camera_matrix, original_image_size, z_depth=1000.0, grid_spacing=200
        )

        # Ground truth path
        points_2d_original = project_3d_to_2d(points_3d, camera_matrix)
        points_2d_ground_truth = apply_resize_crop_transform(
            points_2d_original, transform_params
        )

        # Predicted path
        camera_matrix_resized = transform_camera_matrix(camera_matrix, transform_params)
        points_2d_predicted = project_3d_to_2d(points_3d, camera_matrix_resized)

        # Filter valid points
        final_width, final_height = transform_params["final_size"]
        valid_mask = (
            (points_2d_ground_truth[:, 0] >= 0)
            & (points_2d_ground_truth[:, 0] < final_width)
            & (points_2d_ground_truth[:, 1] >= 0)
            & (points_2d_ground_truth[:, 1] < final_height)
        )

        points_2d_ground_truth_valid = points_2d_ground_truth[valid_mask]
        points_2d_predicted_valid = points_2d_predicted[valid_mask]

        # Calculate errors
        pixel_errors = np.linalg.norm(
            points_2d_ground_truth_valid - points_2d_predicted_valid, axis=1
        )

        max_error = np.max(pixel_errors)
        mean_error = np.mean(pixel_errors)
        median_error = np.median(pixel_errors)

        results[name] = {
            "max_error": float(max_error),
            "mean_error": float(mean_error),
            "median_error": float(median_error),
            "num_points": int(len(points_2d_ground_truth_valid)),
            "passed": bool(float(max_error) < 0.5),
        }

        passed = max_error < 0.5
        all_passed = all_passed and passed

        status = "✅ PASSED" if passed else "❌ FAILED"
        print(
            f"  {status} - Max error: {max_error:.6f} px, Mean error: {mean_error:.6f} px"
        )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Max error: {result['max_error']:.6f} px")
        print(f"  Mean error: {result['mean_error']:.6f} px")
        print(f"  Valid points: {result['num_points']}")

    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED - Intrinsics conversion is validated!")
    else:
        print(f"\n❌ SOME TESTS FAILED - Review conversion logic")

    # Save comprehensive summary
    summary = {
        "test_name": "test_complete_pipeline_validation",
        "configuration": {
            "original_size": list(original_image_size),
            "target_size": target_size,
            "patch_size": patch_size,
        },
        "results": results,
        "overall": {
            "all_passed": bool(all_passed),
            "threshold_px": 0.5,
        },
    }

    output_file = os.path.join(output_dir, "complete_validation_summary.json")
    save_json_results(output_file, summary)

    print(f"\n📁 Saved comprehensive summary to: {output_file}")

    assert all_passed, "Complete pipeline validation failed"
