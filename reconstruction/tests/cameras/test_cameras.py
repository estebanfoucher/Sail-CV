"""
Test file for camera visualization with coherent coordinate transformations.

This file tests the stereo calibration format using extrinsics_calibration_512x288.json.
The stereo calibration format uses world-to-camera transforms that are automatically converted
to camera-to-world transforms for proper visualization.

Coordinate System Conventions:
- Stereo calibration: X_camera2 = R @ X_camera1 + T (world-to-camera)
- Camera class: P_world = R @ P_camera + T (camera-to-world)
- The conversion function handles the transformation between these conventions.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from cameras import (
    Camera,
    convert_world_to_camera_to_camera_to_world,
    create_cameras_from_stereo_calibration,
    export_cameras_to_cloudcompare,
)

# Get the test assets directory relative to this test file
TEST_ASSETS_DIR = Path(__file__).parent / "test_assets"
ROOT_DIR = Path(__file__).parent.parent.parent

OUTPUT_DIR = ROOT_DIR / "output_tests" / "cameras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cameras_from_stereo_calibration(
    calibration_path: str, image_1_path: str, image_2_path: str
):
    """
    Load cameras from stereo calibration JSON format.

    Args:
        calibration_path: Path to stereo calibration JSON file (e.g., extrinsics_calibration_512x288.json)
        image_1_path: Path to first camera image
        image_2_path: Path to second camera image

    Returns:
        Tuple of (camera1, camera2) Camera objects
    """
    with open(calibration_path) as f:
        calibration_data = json.load(f)

    return create_cameras_from_stereo_calibration(
        calibration_data, image_1_path, image_2_path
    )


def create_modified_cameras_from_stereo_calibration(
    calibration_path: str,
    image_1_path: str,
    image_2_path: str,
    translation_vector=None,
    rotation_matrix=None,
):
    """
    Create cameras with modified translation and rotation from stereo calibration data.

    Args:
        calibration_path: Path to stereo calibration JSON file
        image_1_path: Path to first camera image
        image_2_path: Path to second camera image
        translation_vector: New translation vector for camera2 (if None, uses original from calibration)
        rotation_matrix: New rotation matrix for camera2 (if None, uses original from calibration)

    Returns:
        Tuple of (camera1, camera2) Camera objects
    """
    with open(calibration_path) as f:
        calibration_data = json.load(f)

    # Extract original calibration data
    camera_matrix1 = np.array(calibration_data["camera_matrix1"])
    camera_matrix2 = np.array(calibration_data["camera_matrix2"])
    original_rotation_matrix = np.array(calibration_data["rotation_matrix"])
    original_translation_vector = np.array(calibration_data["translation_vector"])
    image_size = tuple(calibration_data["image_size"])

    # Use provided values or defaults
    if rotation_matrix is not None:
        rotation_matrix = np.array(rotation_matrix)
    else:
        rotation_matrix = original_rotation_matrix

    if translation_vector is not None:
        translation_vector = np.array(translation_vector).reshape(3, 1)
    else:
        translation_vector = original_translation_vector

    # Create camera1 (reference camera at origin)
    camera1 = Camera(
        name="camera_1",
        position=[0.0, 0.0, 0.0],
        rotation_matrix=np.eye(3),
        intrinsics=camera_matrix1,
        image_size=image_size,
        image_path=image_1_path,
    )

    # Create camera2 with modified transforms
    # Convert world-to-camera transforms to camera-to-world transforms
    camera2_rotation_world, camera2_position_world = (
        convert_world_to_camera_to_camera_to_world(rotation_matrix, translation_vector)
    )

    camera2 = Camera(
        name="camera_2",
        position=camera2_position_world,
        rotation_matrix=camera2_rotation_world,
        intrinsics=camera_matrix2,
        image_size=image_size,
        image_path=image_2_path,
    )

    return camera1, camera2


def test_stereo_calibration_normal():
    """Test using the actual stereo calibration data from extrinsics_calibration_512x288.json"""
    calibration_path = str(TEST_ASSETS_DIR / "extrinsics_calibration_512x288.json")
    image_1_path = str(TEST_ASSETS_DIR / "images" / "frame_1.png")
    image_2_path = str(TEST_ASSETS_DIR / "images" / "frame_2.png")
    output_dir = str(OUTPUT_DIR / "stereo_calibration_normal")

    # Load cameras from stereo calibration data
    camera1, camera2 = load_cameras_from_stereo_calibration(
        calibration_path, image_1_path, image_2_path
    )
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")


def test_stereo_calibration_1m_translation_z():
    """Test stereo calibration with 1m translation in Z direction"""
    calibration_path = str(TEST_ASSETS_DIR / "extrinsics_calibration_512x288.json")
    image_1_path = str(TEST_ASSETS_DIR / "images" / "frame_1.png")
    image_2_path = str(TEST_ASSETS_DIR / "images" / "frame_2.png")
    output_dir = str(OUTPUT_DIR / "stereo_calibration_1m_translation_z")

    # Create cameras with modified translation (1m in Z direction)
    camera1, camera2 = create_modified_cameras_from_stereo_calibration(
        calibration_path,
        image_1_path,
        image_2_path,
        translation_vector=[0, 0, 1],  # 1m in Z direction
        rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity rotation
    )
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")


def test_stereo_calibration_90_rotation_y():
    """Test stereo calibration with 90 degree rotation around Y axis"""
    calibration_path = str(TEST_ASSETS_DIR / "extrinsics_calibration_512x288.json")
    image_1_path = str(TEST_ASSETS_DIR / "images" / "frame_1.png")
    image_2_path = str(TEST_ASSETS_DIR / "images" / "frame_2.png")
    output_dir = str(OUTPUT_DIR / "stereo_calibration_90_rotation_y")

    # Create cameras with modified rotation (90 degrees around Y axis)
    camera1, camera2 = create_modified_cameras_from_stereo_calibration(
        calibration_path,
        image_1_path,
        image_2_path,
        translation_vector=[0, 0, 0],  # No translation
        rotation_matrix=[
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ],  # 90 degree rotation around Y
    )
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")


if __name__ == "__main__":
    # Test with stereo calibration format
    print("Testing with stereo calibration format...")
    test_stereo_calibration_normal()
    test_stereo_calibration_1m_translation_z()
    test_stereo_calibration_90_rotation_y()

    print("All tests completed!")
