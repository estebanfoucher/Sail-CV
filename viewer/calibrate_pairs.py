"""Stereo calibration from image pairs.

Run inside Docker. Reads pairs from PAIRS_DIR:
  pair_001/cam1.jpg, pair_001/cam2.jpg
  pair_002/cam1.jpg, ...

Detects ChArUco corners, runs intrinsics + stereoCalibrate, writes calibration.json.

Stdout: JSON status lines consumed by server.py.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, "/app/src/reconstruction")

from loguru import logger

OUTPUT_DIR = Path("/app/output")
DEFAULT_CALIB_DIR = Path("/tmp/sailcv_calibrations")
DEFAULT_CALIB_DIR.mkdir(parents=True, exist_ok=True)


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def get_charuco_board(cols: int, rows: int, sq_m: float):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((cols, rows), sq_m, sq_m * 0.75, dictionary)
    return board, dictionary


def detect_charuco(img: np.ndarray, board, dictionary):
    """Detect ChArUco corners. Returns (charuco_corners, charuco_ids) or (None, None)."""
    charuco_detector = aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(img)
    if charuco_corners is None or len(charuco_corners) < 6:
        return None, None
    return charuco_corners, charuco_ids


def calibrate_intrinsics(images: list[np.ndarray], board, dictionary, img_size: tuple):
    obj_pts_list, img_pts_list = [], []
    board_corners = board.getChessboardCorners()
    for img in images:
        c, ids = detect_charuco(img, board, dictionary)
        if c is None:
            continue
        obj_pts = np.array([board_corners[int(i)] for i in ids.flatten()], dtype=np.float32)
        obj_pts_list.append(obj_pts)
        img_pts_list.append(c)

    if len(obj_pts_list) < 5:
        raise RuntimeError(f"Not enough detections for intrinsics: {len(obj_pts_list)}")

    logger.info(f"Intrinsics: using {len(obj_pts_list)} frames")
    err, K, dist, _, _ = cv2.calibrateCamera(obj_pts_list, img_pts_list, img_size, None, None)
    return K, dist, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--square-mm", type=float, default=30.78)
    # Default output: timestamped file in /tmp/sailcv_calibrations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = DEFAULT_CALIB_DIR / f"calibration_{timestamp}.json"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    sq_m = args.square_mm / 1000.0
    board, dictionary = get_charuco_board(args.cols, args.rows, sq_m)

    # Load all pairs
    pairs_dir = args.pairs_dir
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_"))
    if not pair_dirs:
        emit({"error": f"No pairs found in {pairs_dir}"})
        sys.exit(1)

    logger.info(f"Found {len(pair_dirs)} pairs")
    emit({"status": "detecting", "total_pairs": len(pair_dirs)})

    imgs1, imgs2 = [], []
    corners1_list, ids1_list = [], []
    corners2_list, ids2_list = [], []
    obj_points_list = []
    img_pts1_list, img_pts2_list = [], []
    img_size = None

    for pair_dir in pair_dirs:
        im1 = cv2.imread(str(pair_dir / "cam1.jpg"))
        im2 = cv2.imread(str(pair_dir / "cam2.jpg"))
        if im1 is None or im2 is None:
            continue
        if img_size is None:
            img_size = (im1.shape[1], im1.shape[0])

        imgs1.append(im1)
        imgs2.append(im2)

        c1, i1 = detect_charuco(im1, board, dictionary)
        c2, i2 = detect_charuco(im2, board, dictionary)
        if c1 is not None:
            corners1_list.append(c1); ids1_list.append(i1)
        if c2 is not None:
            corners2_list.append(c2); ids2_list.append(i2)

        # Stereo: need matching ids in both
        if c1 is not None and c2 is not None:
            ids1_set = {int(x): c1[j] for j, x in enumerate(i1.flatten())}
            ids2_set = {int(x): c2[j] for j, x in enumerate(i2.flatten())}
            common = sorted(set(ids1_set) & set(ids2_set))
            if len(common) >= 4:
                board_corners = board.getChessboardCorners()
                obj_pts = np.array([board_corners[cid] for cid in common], dtype=np.float32)
                p1 = np.array([ids1_set[cid] for cid in common], dtype=np.float32)
                p2 = np.array([ids2_set[cid] for cid in common], dtype=np.float32)
                obj_points_list.append(obj_pts)
                img_pts1_list.append(p1)
                img_pts2_list.append(p2)

    logger.info(f"Cam1 detections: {len(corners1_list)}, Cam2: {len(corners2_list)}, stereo pairs: {len(obj_points_list)}")
    emit({"status": "calibrating",
          "cam1_detections": len(corners1_list),
          "cam2_detections": len(corners2_list),
          "stereo_pairs": len(obj_points_list)})

    if len(obj_points_list) < 5:
        emit({"error": f"Not enough stereo pairs with board visible in both cameras: {len(obj_points_list)}"})
        sys.exit(1)

    # Intrinsics
    emit({"status": "calibrating_intrinsics"})
    K1, dist1, err1 = calibrate_intrinsics(imgs1, board, dictionary, img_size)
    logger.info(f"Cam1 intrinsics RMS: {err1:.4f}")
    K2, dist2, err2 = calibrate_intrinsics(imgs2, board, dictionary, img_size)
    logger.info(f"Cam2 intrinsics RMS: {err2:.4f}")

    # Stereo calibration
    emit({"status": "calibrating_stereo"})
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points_list, img_pts1_list, img_pts2_list,
        K1, dist1, K2, dist2, img_size,
        flags=flags,
    )
    logger.info(f"Stereo RMS reprojection error: {rms:.4f}px")

    result = {
        "success": True,
        "reprojection_error": float(rms),
        "num_pairs": len(obj_points_list),
        "camera_matrix1": K1.tolist(),
        "camera_matrix2": K2.tolist(),
        "dist_coeffs1": dist1.tolist(),
        "dist_coeffs2": dist2.tolist(),
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "image_size": list(img_size),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved calibration to {args.output}")

    # Also copy to legacy location for backwards compatibility
    legacy_path = OUTPUT_DIR / "calibration" / "calibration.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output, legacy_path)
    logger.info(f"Copied calibration to {legacy_path}")

    emit({"status": "done", "reprojection_error": float(rms), "output": str(args.output)})


if __name__ == "__main__":
    main()
