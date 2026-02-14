"""Test checkerboard detector performance and accuracy."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.intrinsics_calibration import detect_checkerboard_fast


def test_checkerboard_detector_frame_0_should_fail():
    """Test that frame 0 normally fails detection (no checkerboard)."""
    project_root = Path(__file__).resolve().parents[2]
    frame_path = project_root / "assets" / "reconstruction" / "checkerboard" / "frame_0.png"
    board_size = (14, 10)  # (inner_corners_x, inner_corners_y)

    if not frame_path.exists():
        pytest.skip(f"Test frame not found: {frame_path}")

    # Load frame
    frame = cv2.imread(str(frame_path))
    assert frame is not None, f"Failed to load frame: {frame_path}"

    # Detect checkerboard using fast detector (with debug for frame 400)
    start_time = time.time()
    success, corners = detect_checkerboard_fast(frame, board_size, debug=True)
    detection_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"Test: frame_0.png (should fail)")
    print(f"Board size: {board_size}")
    print(f"Detection time: {detection_time*1000:.2f} ms")
    print(f"Detection success: {success}")
    print(f"{'='*60}\n")

    # Assert detection time is less than 1.5 seconds (allows for system variability)
    assert detection_time < 1.5, (
        f"Detection took {detection_time*1000:.2f} ms, exceeds 1.5 second limit"
    )

    # Frame 0 should normally fail (no checkerboard detected)
    assert not success, "Frame 0 should not detect checkerboard"
    assert corners is None, "Frame 0 should not return corners"


def test_checkerboard_detector_frame_400_should_succeed():
    """Test that frame 400 detects checkerboard successfully."""
    project_root = Path(__file__).resolve().parents[2]
    frame_path = project_root / "assets" / "reconstruction" / "checkerboard" / "frame_400.png"
    board_size = (13, 9)  # (inner_corners_x, inner_corners_y)

    if not frame_path.exists():
        pytest.skip(f"Test frame not found: {frame_path}")

    # Load frame
    frame = cv2.imread(str(frame_path))
    assert frame is not None, f"Failed to load frame: {frame_path}"

    # Detect checkerboard using fast detector (with debug for frame 400)
    start_time = time.time()
    success, corners = detect_checkerboard_fast(frame, board_size, debug=True)
    detection_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"Test: frame_400.png (should succeed)")
    print(f"Board size: {board_size}")
    print(f"Detection time: {detection_time*1000:.2f} ms")
    print(f"Detection success: {success}")

    if success and corners is not None:
        print(f"Corners found: {len(corners)} points")
        print(f"Corner shape: {corners.shape}")
        print(f"First 5 corners:")
        for i, corner in enumerate(corners[:5]):
            print(f"  Corner {i}: ({corner[0][0]:.2f}, {corner[0][1]:.2f})")

        # Draw corners on frame for visualization
        processed_frame = frame.copy()
        cv2.drawChessboardCorners(
            processed_frame, board_size, corners, success
        )

        # Save processed frame with corners drawn to output_tests/detector/
        output_dir = project_root / "output_tests" / "detector"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "chessboard_detection.png"
        cv2.imwrite(str(output_path), processed_frame)
        print(f"Processed frame saved to: {output_path}")
    else:
        print("No checkerboard detected")

    print(f"{'='*60}\n")

    # Assert detection time is less than 1.5 seconds (allows for system variability)
    assert detection_time < 1.5, (
        f"Detection took {detection_time*1000:.2f} ms, exceeds 1.5 second limit"
    )

    # Frame 400 should succeed (checkerboard detected)
    assert success, "Frame 400 should detect checkerboard"
    assert corners is not None, "Frame 400 should have detected corners"
    assert len(corners) == board_size[0] * board_size[1], (
        f"Expected {board_size[0] * board_size[1]} corners, got {len(corners)}"
    )
