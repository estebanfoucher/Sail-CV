#!/usr/bin/env python3
"""
Calibrate stereo from pre-captured intrinsics and extrinsics pairs.

Usage:
    python3 scripts/calibrate_from_pairs.py \
        --intrinsics-dir /path/to/intrinsics/pairs \
        --extrinsics-dir /path/to/extrinsics/pairs \
        --output /path/to/output/calibration.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

# Setup paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "reconstruction"))

from calibration.extrinsics_calibration import calibrate_stereo_many, CharucoDetector
from calibration.intrinsics_calibration import IntrinsicCalibration


def load_intrinsics_from_pairs(pairs_dir: Path) -> tuple[dict, dict]:
    """Load camera matrices from intrinsics calibration pairs."""
    logger.info(f"Loading intrinsics from {pairs_dir}")

    intrinsic_cal = IntrinsicCalibration(tag_type="checkerboard")

    # Process each pair
    for pair_dir in sorted(pairs_dir.glob("pair_*")):
        cam1_path = pair_dir / "cam1.jpg"
        cam2_path = pair_dir / "cam2.jpg"

        if not (cam1_path.exists() and cam2_path.exists()):
            continue

        cam1 = cv2.imread(str(cam1_path), cv2.IMREAD_GRAYSCALE)
        cam2 = cv2.imread(str(cam2_path), cv2.IMREAD_GRAYSCALE)

        if cam1 is None or cam2 is None:
            logger.warning(f"Failed to load images from {pair_dir}")
            continue

        try:
            # Detect checkerboard in cam1
            ret1, corners1 = cv2.findChessboardCorners(
                cam1, (9, 6), None,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret1:
                intrinsic_cal.add_camera_1_frame(cam1, corners1)

            # Detect checkerboard in cam2
            ret2, corners2 = cv2.findChessboardCorners(
                cam2, (9, 6), None,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret2:
                intrinsic_cal.add_camera_2_frame(cam2, corners2)

            if ret1 or ret2:
                logger.debug(f"Detected checkerboard in {pair_dir.name}: cam1={ret1}, cam2={ret2}")
        except Exception as e:
            logger.warning(f"Error processing {pair_dir}: {e}")

    # Calibrate both cameras
    logger.info("Computing intrinsics calibration...")
    intrinsics1 = intrinsic_cal.calibrate_camera_1()
    intrinsics2 = intrinsic_cal.calibrate_camera_2()

    return intrinsics1, intrinsics2


def load_extrinsics_pairs(pairs_dir: Path) -> tuple[list, list, list, list]:
    """Load extrinsics calibration pairs using ChArUco."""
    logger.info(f"Loading extrinsics pairs from {pairs_dir}")

    detector = CharucoDetector()

    object_points = []
    image_points1 = []
    image_points2 = []
    valid_pairs = 0

    for pair_dir in sorted(pairs_dir.glob("pair_*")):
        cam1_path = pair_dir / "cam1.jpg"
        cam2_path = pair_dir / "cam2.jpg"

        if not (cam1_path.exists() and cam2_path.exists()):
            continue

        cam1 = cv2.imread(str(cam1_path))
        cam2 = cv2.imread(str(cam2_path))

        if cam1 is None or cam2 is None:
            logger.warning(f"Failed to load images from {pair_dir}")
            continue

        try:
            result1 = detector.detect(cam1)
            result2 = detector.detect(cam2)

            if result1 and result2:
                obj_pts = result1["object_points"]
                img_pts1 = result1["image_points"]
                img_pts2 = result2["image_points"]

                # Match points if they have same count
                if len(obj_pts) == len(img_pts1) == len(img_pts2) > 4:
                    object_points.append(obj_pts)
                    image_points1.append(img_pts1)
                    image_points2.append(img_pts2)
                    valid_pairs += 1
                    logger.debug(f"Valid extrinsics pair in {pair_dir.name}: {len(obj_pts)} pts")
        except Exception as e:
            logger.warning(f"Error processing {pair_dir}: {e}")

    logger.info(f"Found {valid_pairs} valid extrinsics pairs")
    return object_points, image_points1, image_points2, [valid_pairs]


def main():
    parser = argparse.ArgumentParser(description="Calibrate stereo from captured pairs")
    parser.add_argument("--intrinsics-dir", type=Path, required=True,
                       help="Directory with intrinsics pairs (e.g., tmp/may22/0036/intrinsics)")
    parser.add_argument("--extrinsics-dir", type=Path, required=True,
                       help="Directory with extrinsics pairs")
    parser.add_argument("--output", type=Path, default=Path("output/calibration/calibration.json"),
                       help="Output calibration JSON")
    args = parser.parse_args()

    args.intrinsics_dir = args.intrinsics_dir.resolve()
    args.extrinsics_dir = args.extrinsics_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Calibrating from:")
    logger.info(f"  Intrinsics: {args.intrinsics_dir}")
    logger.info(f"  Extrinsics: {args.extrinsics_dir}")
    logger.info(f"  Output: {args.output}")

    # Step 1: Load intrinsics
    try:
        intrinsics1, intrinsics2 = load_intrinsics_from_pairs(args.intrinsics_dir)
    except Exception as e:
        logger.error(f"Failed to compute intrinsics: {e}")
        return 1

    logger.info(f"Intrinsics1 RMS: {intrinsics1.get('rms_error', 'N/A')}")
    logger.info(f"Intrinsics2 RMS: {intrinsics2.get('rms_error', 'N/A')}")

    # Step 2: Load extrinsics
    try:
        obj_pts, img_pts1, img_pts2, pair_count = load_extrinsics_pairs(args.extrinsics_dir)
    except Exception as e:
        logger.error(f"Failed to load extrinsics pairs: {e}")
        return 1

    if not obj_pts:
        logger.error("No valid extrinsics pairs found")
        return 1

    # Step 3: Run stereo calibration
    logger.info("Running stereo calibration...")
    try:
        image_size = (1920, 1080)  # Default, may need adjustment
        calib_result = calibrate_stereo_many(
            obj_pts, img_pts1, img_pts2,
            np.array(intrinsics1["camera_matrix"]),
            np.array(intrinsics1["dist_coeffs"]),
            np.array(intrinsics2["camera_matrix"]),
            np.array(intrinsics2["dist_coeffs"]),
            image_size
        )
    except Exception as e:
        logger.error(f"Stereo calibration failed: {e}")
        return 1

    # Add intrinsics info
    calib_result["intrinsics_rms_cam1"] = intrinsics1.get("rms_error", 0)
    calib_result["intrinsics_rms_cam2"] = intrinsics2.get("rms_error", 0)
    calib_result["intrinsics_frames_cam1"] = intrinsics1.get("num_frames", 0)
    calib_result["intrinsics_frames_cam2"] = intrinsics2.get("num_frames", 0)
    calib_result["baseline_m"] = float(np.linalg.norm(calib_result["translation_vector"]))

    # Save result
    with open(args.output, "w") as f:
        json.dump(calib_result, f, indent=2)

    logger.info(f"Calibration saved to {args.output}")
    logger.info(f"Reprojection error: {calib_result['reprojection_error']:.2f} px")
    logger.info(f"Baseline: {calib_result['baseline_m']:.4f} m")
    logger.info(f"Pairs used: {calib_result['num_pairs']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
