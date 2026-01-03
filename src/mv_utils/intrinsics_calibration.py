import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .utils import load_parameters

# High-performance checkerboard detector thresholds
DOWNSAMPLE_SCALE = 0.18  # Resize factor for early detection (0.15-0.2 range)
EDGE_DENSITY_THRESHOLD = (
    0.05  # Minimum edge density ratio (Canny) - lowered for better detection
)
FAST_THRESHOLD = 20  # FAST corner detector threshold
FAST_CORNER_RATIO = (
    0.4  # Minimum ratio of expected corners to detect - lowered for better detection
)
GRID_SPACING_VARIANCE_THRESHOLD = 0.35  # Maximum variance in grid spacing (moderate)
GRID_ORTHOGONALITY_THRESHOLD = (
    0.85  # Minimum cosine for near-orthogonality (~90°, moderate)
)


def detect_checkerboard_fast(  # noqa: PLR0911
    image: np.ndarray, board_size: tuple[int, int], debug: bool = False
) -> tuple[bool, np.ndarray | None]:
    """
    High-performance checkerboard detector with aggressive early rejection.

    Designed for real-time systems where most frames do not contain a checkerboard.
    Fails fast on 4K images when no checkerboard exists.

    Args:
        image: Input 4K RGB/BGR image as numpy array
        board_size: Tuple of (inner_corners_x, inner_corners_y) for checkerboard
        debug: If True, print debug information about rejection stages

    Returns:
        Tuple of (found: bool, corners: np.ndarray | None)
        - found: True if checkerboard detected, False otherwise
        - corners: Detected corners if found, None otherwise

    Performance targets:
        - Average runtime: <10 ms
        - Worst-case runtime: <25 ms
    """
    if image is None or image.size == 0:
        if debug:
            logger.debug("Stage 0: Empty image")
        return False, None

    h, w = image.shape[:2]
    expected_corners = board_size[0] * board_size[1]

    # Stage 1: Downscale + Grayscale (Early Exit)
    small = cv2.resize(
        image,
        None,
        fx=DOWNSAMPLE_SCALE,
        fy=DOWNSAMPLE_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    if small.size == 0:
        if debug:
            logger.debug("Stage 1: Resize failed")
        return False, None

    if len(small.shape) == 3:
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray_small = small

    # Stage 2: Edge Density Early Rejection
    # Checkerboards have extremely high edge density
    edges = cv2.Canny(gray_small, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size

    if debug:
        logger.debug(
            f"Stage 2: Edge ratio = {edge_ratio:.4f} (threshold: {EDGE_DENSITY_THRESHOLD})"
        )

    if edge_ratio < EDGE_DENSITY_THRESHOLD:
        if debug:
            logger.debug("Stage 2: Rejected - edge ratio too low")
        return False, None

    # Stage 3: FAST Corner Count Check
    # Use FAST detector to quickly estimate corner density
    fast = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD)
    fast_corners = fast.detect(gray_small, None)
    num_fast_corners = len(fast_corners)

    # Expected corners scaled to downsampled image
    min_expected_corners = expected_corners * FAST_CORNER_RATIO

    if debug:
        logger.debug(
            f"Stage 3: FAST corners = {num_fast_corners} (min expected: {min_expected_corners:.1f})"
        )

    if num_fast_corners < min_expected_corners:
        if debug:
            logger.debug("Stage 3: Rejected - not enough FAST corners")
        return False, None

    # Stage 4: Grid Structure Validation
    # Always validate grid structure - this is fast and catches false positives
    grid_valid = _validate_grid_structure(fast_corners, gray_small.shape, debug)

    if debug:
        logger.debug(f"Stage 4: Grid validation = {grid_valid}")

    if not grid_valid:
        if debug:
            logger.debug("Stage 4: Rejected - grid structure invalid")
        return False, None

    # Stage 5: Full-Resolution Verification
    # Only if all previous stages pass, check at full resolution
    if len(image.shape) == 3:
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = image

    # Try detection at 1080p first (faster), then scale to full-res if found
    # This is more reliable than going straight to 4K
    h, w = gray_full.shape
    target_height = 1080
    scale_factor_1080 = target_height / h
    target_width_1080 = int(w * scale_factor_1080)
    target_width_1080 = target_width_1080 + (target_width_1080 % 2)

    gray_1080 = cv2.resize(
        gray_full, (target_width_1080, target_height), interpolation=cv2.INTER_LINEAR
    )

    # Try detection at 1080p with the most promising flag combination first
    # Only try one combination to fail fast - if grid validation passed, this should work
    # If it fails, we exit immediately (no fallback to other flags for speed)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    found_1080, corners_1080 = cv2.findChessboardCorners(gray_1080, board_size, flags)

    if debug:
        logger.debug(f"Stage 5: 1080p detection with ADAPTIVE_THRESH = {found_1080}")

    # Fail fast if not found at 1080p - do NOT try full-res
    if not found_1080:
        if debug:
            logger.debug(
                "Stage 5: Rejected - 1080p detection failed (failing fast, no full-res fallback)"
            )
        return False, None

    # If found at 1080p, scale corners back to full resolution and refine
    corners_fullres = corners_1080 / scale_factor_1080
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.0015)
    corners_refined = cv2.cornerSubPix(
        gray_full, corners_fullres, (11, 11), (-1, -1), criteria
    )

    return True, corners_refined


def _validate_grid_structure(  # noqa: PLR0911
    corners: list, image_shape: tuple[int, int], debug: bool = False
) -> bool:
    """
    Validate that FAST corners form a grid-like structure.

    Checks:
    - Nearest-neighbor distances show two dominant spacing peaks
    - Near-orthogonality (~90°)
    - Low spacing variance

    Args:
        corners: List of FAST corner keypoints
        image_shape: (height, width) of the image

    Returns:
        True if grid structure is detected, False otherwise
    """
    if len(corners) < 9:  # Need minimum corners for grid validation
        return False

    # Extract corner coordinates
    points = np.array([[kp.pt[0], kp.pt[1]] for kp in corners])

    if len(points) < 9:
        return False

    # Compute nearest-neighbor distances for a sample of points
    # Use a subset to avoid O(n²) complexity - limit to 30 for speed
    sample_size = min(30, len(points))
    # Use deterministic sampling for reproducibility (take evenly spaced points)
    if len(points) <= sample_size:
        sample_indices = np.arange(len(points))
    else:
        # Take evenly spaced indices for deterministic sampling
        step = len(points) / sample_size
        sample_indices = (np.arange(sample_size) * step).astype(int)
    sample_points = points[sample_indices]

    # Find nearest neighbors and compute distances
    # Use pure numpy for efficiency
    distances_list = []
    directions_list = []

    for pt in sample_points:
        # Compute distances to all points (vectorized)
        diff = points - pt
        distances = np.linalg.norm(diff, axis=1)
        # Sort and get nearest neighbors (excluding self at distance 0)
        sorted_indices = np.argsort(distances)
        # Get next 2-3 nearest neighbors (enough for grid validation)
        for idx in sorted_indices[1:4]:
            if idx < len(points):
                neighbor = points[idx]
                dist = distances[idx]
                # Skip if too close (likely noise)
                if dist < 5:
                    continue
                direction = neighbor - pt
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                    distances_list.append(dist)
                    directions_list.append(direction)

    if len(distances_list) < 6:
        return False

    distances_array = np.array(distances_list)

    # Check spacing variance - should be relatively low for a grid
    mean_dist = np.mean(distances_array)
    if mean_dist < 1e-6:
        if debug:
            logger.debug("Grid validation: mean distance too small")
        return False
    spacing_variance = np.var(distances_array) / (mean_dist**2)
    if debug:
        logger.debug(
            f"Grid validation: spacing variance = {spacing_variance:.4f} (threshold: {GRID_SPACING_VARIANCE_THRESHOLD})"
        )
    if spacing_variance > GRID_SPACING_VARIANCE_THRESHOLD:
        if debug:
            logger.debug("Grid validation: spacing variance too high")
        return False

    # If spacing variance is very low, we can be more lenient with other checks
    very_low_variance = spacing_variance < 0.2

    # Check for dominant spacing peaks (grid pattern)
    # Use histogram to find peaks
    hist, _bin_edges = np.histogram(distances_array, bins=15)
    peak_count = 0
    for i in range(1, len(hist) - 1):
        if (
            hist[i] > hist[i - 1]
            and hist[i] > hist[i + 1]
            and hist[i] > len(distances_array) * 0.1
        ):
            peak_count += 1

    # Should have at least one dominant spacing peak (relaxed requirement)
    # Grids can have varying patterns, so we're more lenient here
    if peak_count == 0 and len(distances_array) < 15:
        if debug:
            logger.debug(
                f"Grid validation: no spacing peaks found (peak_count={peak_count}, distances={len(distances_array)})"
            )
        return False

    # Check near-orthogonality
    # Sample direction pairs and check if they're roughly orthogonal
    if len(directions_list) >= 4:
        directions_array = np.array(directions_list)
        orthogonal_count = 0
        total_pairs = 0

        # Sample pairs to check orthogonality (limit to 15 for speed)
        # Use deterministic sampling for reproducibility - try multiple pairs
        num_pairs = min(15, len(directions_array) * (len(directions_array) - 1) // 2)
        checked_pairs = set()
        pair_idx = 0
        for i in range(len(directions_array)):
            for j in range(i + 1, len(directions_array)):
                if pair_idx >= num_pairs:
                    break
                if (i, j) in checked_pairs:
                    continue
                checked_pairs.add((i, j))
                dot_product = abs(np.dot(directions_array[i], directions_array[j]))
                # Near orthogonal means dot product close to 0 (< 0.3 = ~72-108 degrees)
                if dot_product < 0.3:
                    orthogonal_count += 1
                total_pairs += 1
                pair_idx += 1
            if pair_idx >= num_pairs:
                break

        if total_pairs > 0:
            orthogonality_ratio = orthogonal_count / total_pairs
            if debug:
                logger.debug(
                    f"Grid validation: orthogonality ratio = {orthogonality_ratio:.4f} (threshold: 0.15, relaxed: 0.05 if low variance)"
                )
            # Should have reasonable orthogonality for a grid
            # If spacing variance is very low, use a more relaxed threshold
            orthogonality_threshold = 0.05 if very_low_variance else 0.15
            if orthogonality_ratio < orthogonality_threshold:
                if debug:
                    logger.debug(
                        f"Grid validation: orthogonality ratio too low (got {orthogonality_ratio:.4f}, need {orthogonality_threshold:.2f})"
                    )
                return False

    return True


def find_corners_in_images(
    image_numbers_list: list[int],
    video_reader,
    pattern_size: tuple[int, int],
    square_size: float,
    target_H: int | None = None,
    target_W: int | None = None,
) -> tuple[list, list, list]:
    """Find checkerboard corners in defined images in the video.

    Uses the high-performance detect_checkerboard_fast function for fast detection
    with aggressive early rejection. Detection happens at 1080p, and if found,
    corners are refined at full resolution.

    IMPORTANT: Frames from video_reader are already EXIF-corrected (landscape orientation).
    If target_H and target_W are provided, frames are cropped to match target resolution
    BEFORE detection. This ensures intrinsics calibration uses the same resolution as
    extrinsics calibration.

    Args:
        image_numbers_list: List of image numbers to process.
        video_reader: VideoReader object (returns EXIF-corrected frames).
        pattern_size: Tuple of (width, height) of the checkerboard pattern (inner corners).
        square_size: Size of the square in the checkerboard pattern.
        target_H: Target height for cropping (EXIF-corrected). If None, use original frame size.
        target_W: Target width for cropping (EXIF-corrected). If None, use original frame size.
    Returns:
        object_points: List of 3D points in real world space.
        image_points: List of 2D points in image plane.
        successful_images: List of image numbers that were successfully processed.
    """
    object_points = []  # 3D points in real world space
    image_points = []  # 2D points in image plane
    successful_images = []

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (pattern_size[0]-1, pattern_size[1]-1, 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    frames = video_reader.get_frames(image_numbers_list)

    # Import crop function if target resolution is provided
    if target_H is not None and target_W is not None:
        from stereo.image import crop_to_match_resolution

    logger.debug(
        f"Processing {len(image_numbers_list)} frames using detect_checkerboard_fast "
        f"(pattern_size={pattern_size}, square_size={square_size}mm)"
        + (
            f", target resolution: {target_W}x{target_H}"
            if target_H and target_W
            else ""
        )
    )

    for frame_number, frame in enumerate(frames):
        # IMPORTANT: Crop frame to target resolution if provided
        # This ensures intrinsics calibration uses the same resolution as extrinsics
        # Frame is already EXIF-corrected (landscape) from VideoReader
        if target_H is not None and target_W is not None:
            H, W = frame.shape[:2]
            if target_H != H or target_W != W:
                processed_frame = crop_to_match_resolution(frame, target_H, target_W)
            else:
                processed_frame = frame
        else:
            processed_frame = frame

        # Use detect_checkerboard_fast for high-performance detection
        # pattern_size is (inner_corners_x, inner_corners_y) which matches board_size
        found, corners = detect_checkerboard_fast(
            processed_frame, pattern_size, debug=False
        )

        if found and corners is not None:
            # Corners are already refined at full resolution by detect_checkerboard_fast
            object_points.append(objp)
            image_points.append(corners)
            successful_images.append(frame_number)
            logger.debug(
                f"Found corners in frame {frame_number} - "
                f"{len(successful_images)}/{frame_number + 1} images processed"
            )
        else:
            logger.debug(
                f"No corners found in frame {frame_number} - "
                f"{len(successful_images)}/{frame_number + 1} images processed"
            )

    logger.debug(
        f"Successfully processed {len(successful_images)}/{len(image_numbers_list)} images"
    )
    return object_points, image_points, successful_images


def calibrate_camera(
    object_points: list, image_points: list, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run camera calibration using OpenCV's calibrateCamera."""
    _ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    logger.debug(f"Calibration error: {mean_error / len(object_points):.4f} pixels")
    return camera_matrix, dist_coeffs, mean_error / len(object_points)


class IntrinsicCalibration:
    def __init__(
        self,
        video_path: str,
        checkerboard_specs_path: str,
        save_path: str,
        temporal_calib_step_sec: float = 1,
    ):
        logger.debug(
            f"Initializing IntrinsicCalibration with video_path: {video_path}, checkerboard_specs_path: {checkerboard_specs_path}, save_path: {save_path}, temporal_calib_step_sec: {temporal_calib_step_sec}"
        )
        self.video_path = video_path
        self.checkerboard_specs_path = checkerboard_specs_path
        self.save_path = save_path
        # Use VideoReader to load video specs and keep VideoReader for frame extraction
        from video import VideoReader

        self.video_reader = VideoReader.open_video_file(video_path)
        self.video = self.video_reader.video
        self.checkerboard_specs = load_parameters(checkerboard_specs_path)
        logger.debug(f"Checkerboard specs: {self.checkerboard_specs}")
        self.image_numbers_list = self.get_image_numbers_list(temporal_calib_step_sec)
        logger.debug(f"Image numbers list: {self.image_numbers_list}")

    def calibrate(
        self,
        save_images: bool = False,
        target_H: int | None = None,
        target_W: int | None = None,
    ):
        # IMPORTANT: If target_H and target_W are provided, frames will be cropped to match
        # This ensures intrinsics calibration uses the same resolution as extrinsics calibration
        # Frames from VideoReader are already EXIF-corrected (landscape orientation)
        object_points, image_points, successful_images = find_corners_in_images(
            self.image_numbers_list,
            self.video_reader,
            (
                self.checkerboard_specs["inner_corners_x"],
                self.checkerboard_specs["inner_corners_y"],
            ),
            self.checkerboard_specs["square_size_mm"],
            target_H=target_H,
            target_W=target_W,
        )
        # IMPORTANT: use the processed frame size (after EXIF transpose/orientation fix and cropping)
        # Frames returned by VideoReader are orientation-corrected; if target resolution is provided,
        # frames are cropped to match target resolution
        assert len(self.image_numbers_list) > 0, (
            "image_numbers_list is empty; cannot determine image size"
        )
        sample_frames = self.video_reader.get_frames([self.image_numbers_list[0]])
        assert sample_frames and sample_frames[0] is not None, (
            "Failed to read a sample frame for determining image size"
        )
        sample_frame = sample_frames[0]
        # If target resolution is provided, crop the sample frame to determine final image_size
        if target_H is not None and target_W is not None:
            from stereo.image import crop_to_match_resolution

            sample_frame = crop_to_match_resolution(sample_frame, target_H, target_W)
        image_size = (sample_frame.shape[1], sample_frame.shape[0])
        logger.debug(
            f"Using processed frame image_size for intrinsics: {image_size} (width, height)"
            + (
                f" (cropped to target {target_W}x{target_H})"
                if target_H and target_W
                else ""
            )
        )

        camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(
            object_points, image_points, image_size
        )
        self.save_calibration(
            camera_matrix,
            dist_coeffs,
            reprojection_error,
            successful_images,
            save_images,
        )
        logger.debug(f"Calibration results saved to {self.save_path}")
        return camera_matrix, dist_coeffs, reprojection_error

    def cleanup(self):
        """Release video reader resources"""
        if hasattr(self, "video_reader") and self.video_reader is not None:
            self.video_reader.release()
            self.video_reader = None

    def get_image_numbers_list(self, temporal_calib_step_sec: float) -> list[int]:
        """Get the list of image numbers from the video."""
        fps = self.video.fps
        # Convert FPS to integer step value for sampling frames
        step = int(fps * temporal_calib_step_sec) if fps > 0 else 1
        return list(range(self.video.frame_count))[::step]

    def save_calibration(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        reprojection_error: float,
        successful_images: list[int],
        save_images: bool = False,
    ):
        """Save the calibration results to a file."""
        with open(self.save_path, "w") as f:
            json.dump(
                {
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "reprojection_error": reprojection_error,
                },
                f,
                indent=2,
            )

        if save_images and successful_images:
            # Convert successful_images indices to actual frame numbers
            actual_frame_numbers = [
                self.image_numbers_list[i] for i in successful_images
            ]
            self._save_successful_images(actual_frame_numbers)

    def _save_successful_images(self, successful_images: list[int]):
        """Save successful calibration images at 360p resolution."""
        # Create successful_images directory
        save_dir = Path(self.save_path).parent / "successful_images"
        save_dir.mkdir(exist_ok=True)

        # 360p resolution (640x360)
        target_width, target_height = 640, 360

        logger.debug(f"Saving {len(successful_images)} successful images to {save_dir}")

        for _i, frame_number in enumerate(successful_images):
            # Get the frame from video
            frames = self.video_reader.get_frames([frame_number])
            if frames:
                frame = frames[0]

                # Resize to 360p
                resized_frame = cv2.resize(frame, (target_width, target_height))

                # Save the image
                image_path = save_dir / f"successful_frame_{frame_number:04d}.jpg"
                cv2.imwrite(str(image_path), resized_frame)

        print(f"Successfully saved {len(successful_images)} images to {save_dir}")
