"""Calibration from pre-extracted still images (no video readers)."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .extrinsics_calibration import CharucoDetector, calibrate_stereo_many
from .image_utils import load_orientation_corrected_image
from .intrinsics_calibration import calibrate_camera_from_image_folder

_STEREO_PAIR_PATTERN = re.compile(r"shot_(.+)_cam([12])\.jpg$", re.IGNORECASE)


def discover_stereo_image_pairs(
    extrinsic_folder: Path | str,
) -> list[tuple[Path, Path]]:
    """Group extrinsic images by timestamp token into (cam1, cam2) pairs."""
    folder = Path(extrinsic_folder)
    cam1_by_id: dict[str, Path] = {}
    cam2_by_id: dict[str, Path] = {}

    for path in sorted(folder.glob("*.jpg")):
        match = _STEREO_PAIR_PATTERN.match(path.name)
        if not match:
            continue
        pair_id, cam = match.group(1), match.group(2)
        if cam == "1":
            cam1_by_id[pair_id] = path
        else:
            cam2_by_id[pair_id] = path

    common_ids = sorted(set(cam1_by_id) & set(cam2_by_id))
    pairs = [(cam1_by_id[i], cam2_by_id[i]) for i in common_ids]
    logger.info(f"Discovered {len(pairs)} stereo pair(s) in {folder}")
    return pairs


def compute_target_resolution(
    cam1_path: Path | str, cam2_path: Path | str
) -> tuple[int, int]:
    """Return (target_H, target_W) as min of both orientation-corrected frames."""
    img1 = load_orientation_corrected_image(cam1_path)
    img2 = load_orientation_corrected_image(cam2_path)
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    target_H = min(H1, H2)
    target_W = min(W1, W2)
    logger.info(
        f"Camera resolutions (EXIF-corrected): cam1={W1}x{H1}, cam2={W2}x{H2}, "
        f"target={target_W}x{target_H}"
    )
    return target_H, target_W


def _load_intrinsics(intrinsics_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    with open(intrinsics_path) as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])


def calibrate_stereo_from_image_pairs(
    extrinsic_folder: Path | str,
    intrinsics1_path: Path | str,
    intrinsics2_path: Path | str,
    pattern_specs_path: Path | str,
    target_H: int,
    target_W: int,
) -> dict[str, Any]:
    """Stereo calibrate from synced still pairs; same output schema as scene.py."""
    from stereo.image import crop_to_match_resolution

    pairs = discover_stereo_image_pairs(extrinsic_folder)
    if not pairs:
        raise RuntimeError(f"No stereo pairs found in {extrinsic_folder}")

    camera_matrix1, dist_coeffs1 = _load_intrinsics(intrinsics1_path)
    camera_matrix2, dist_coeffs2 = _load_intrinsics(intrinsics2_path)

    detector = CharucoDetector(config_path=str(pattern_specs_path))

    object_points_list: list[np.ndarray] = []
    image_points1_list: list[np.ndarray] = []
    image_points2_list: list[np.ndarray] = []

    try:
        for cam1_path, cam2_path in pairs:
            img1 = load_orientation_corrected_image(cam1_path)
            img2 = load_orientation_corrected_image(cam2_path)

            H1, W1 = img1.shape[:2]
            if target_H != H1 or target_W != W1:
                img1 = crop_to_match_resolution(img1, target_H, target_W)

            H2, W2 = img2.shape[:2]
            if target_H != H2 or target_W != W2:
                img2 = crop_to_match_resolution(img2, target_H, target_W)

            p3d, p2d1, p2d2 = detector.get_correspondences(img1, img2)
            if p3d is None or p2d1 is None or p2d2 is None:
                logger.warning(f"No Charuco correspondences for pair {cam1_path.name}")
                continue

            object_points_list.append(p3d)
            image_points1_list.append(p2d1)
            image_points2_list.append(p2d2)
            logger.info(f"Accepted pair {cam1_path.name} / {cam2_path.name}")
    finally:
        detector.cleanup()

    if not object_points_list:
        raise RuntimeError("No successful Charuco detections across stereo pairs")

    image_size = (target_W, target_H)
    return calibrate_stereo_many(
        object_points_list,
        image_points1_list,
        image_points2_list,
        camera_matrix1,
        dist_coeffs1,
        camera_matrix2,
        dist_coeffs2,
        image_size,
    )


def calibrate_extracted_scene(scene_dir: Path | str) -> dict[str, Any]:
    """
    Full calibration for a scene with intrinsic/ and extrinsic/ still folders.

    Writes intrinsic/camera_*/intrinsics.json and scene_dir/calibration.json.
    """
    scene_path = Path(scene_dir)
    intrinsic_dir = scene_path / "intrinsic"
    extrinsic_dir = scene_path / "extrinsic"
    checkerboard_specs = intrinsic_dir / "checkerboard_specs.yml"
    pattern_specs = extrinsic_dir / "extrinsics_calibration_pattern_specs.yml"

    pairs = discover_stereo_image_pairs(extrinsic_dir)
    if not pairs:
        raise RuntimeError(f"No extrinsic pairs in {extrinsic_dir}")

    target_H, target_W = compute_target_resolution(pairs[0][0], pairs[0][1])

    for camera in ("camera_1", "camera_2"):
        calibrate_camera_from_image_folder(
            image_folder=intrinsic_dir / camera,
            checkerboard_specs_path=str(checkerboard_specs),
            save_path=str(intrinsic_dir / camera / "intrinsics.json"),
            target_H=target_H,
            target_W=target_W,
        )

    results = calibrate_stereo_from_image_pairs(
        extrinsic_folder=extrinsic_dir,
        intrinsics1_path=intrinsic_dir / "camera_1" / "intrinsics.json",
        intrinsics2_path=intrinsic_dir / "camera_2" / "intrinsics.json",
        pattern_specs_path=pattern_specs,
        target_H=target_H,
        target_W=target_W,
    )

    calibration_path = scene_path / "calibration.json"
    with open(calibration_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved calibration to {calibration_path}")

    return results
