from typing import Any

import cv2
import numpy as np
import pupil_apriltags as apriltag
import yaml
from loguru import logger


def calibrate_stereo_many(
    object_points_list: list[np.ndarray],
    image_points1_list: list[np.ndarray],
    image_points2_list: list[np.ndarray],
    camera_matrix1,
    dist_coeffs1,
    camera_matrix2,
    dist_coeffs2,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """
    Perform stereo calibration using multiple pairs of correspondences.

    Args:
        object_points_list: List of 3D points for each pair
        image_points1_list: List of 2D points from first camera for each pair
        image_points2_list: List of 2D points from second camera for each pair
        intrinsics1_path: Path to first camera intrinsics JSON
        intrinsics2_path: Path to second camera intrinsics JSON
        image_size: Image size as (width, height)

    Returns:
        Dictionary with stereo calibration results
    """
    total_points = int(sum(len(op) for op in object_points_list))

    logger.debug(
        f"Calibrating on collection of {total_points} 3D-2D correspondences from {len(object_points_list)} pair(s) of images"
    )

    flags = cv2.CALIB_FIX_INTRINSIC

    # cv2.stereoCalibrate returns R and T such that:
    # X_camera2 = R @ X_camera1 + T
    # where X_camera1 is a 3D point in camera1's coordinate system
    # and X_camera2 is the same point in camera2's coordinate system
    # This is a world-to-camera transform if camera1 is considered "world"
    (
        ret,
        camera_matrix1_cal,
        dist_coeffs1_cal,
        camera_matrix2_cal,
        dist_coeffs2_cal,
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        object_points_list,
        image_points1_list,
        image_points2_list,
        camera_matrix1,
        dist_coeffs1,
        camera_matrix2,
        dist_coeffs2,
        image_size,
        flags=flags,
    )

    results = {
        "success": True,
        "reprojection_error": float(ret),
        "num_correspondences": total_points,
        "num_pairs": len(object_points_list),
        "camera_matrix1": camera_matrix1_cal.tolist(),
        "camera_matrix2": camera_matrix2_cal.tolist(),
        "dist_coeffs1": dist_coeffs1_cal.tolist(),
        "dist_coeffs2": dist_coeffs2_cal.tolist(),
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "image_size": image_size,
    }
    return results


class StereoTagDetector:
    """Detect April tags in stereo images and extract correspondences."""

    def __init__(self, config_path: str | None = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.detector = self._create_detector()

        # Define 3D points of the April tag in tag coordinate system
        tag_size = self.config["apriltag"]["tag_size_meters"]
        half_size = tag_size / 2.0
        self.tag_3d_points = np.array(
            [
                [-half_size, -half_size, 0],  # Bottom-left
                [half_size, -half_size, 0],  # Bottom-right
                [half_size, half_size, 0],  # Top-right
                [-half_size, half_size, 0],  # Top-left
            ],
            dtype=np.float32,
        )

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Config file {config_path} not found.")
            raise

    def _create_detector(self):
        """Create AprilTag detector with configured parameters."""
        tag_config = self.config.get("apriltag", {})

        return apriltag.Detector(
            families=tag_config.get("tag_family", "tag36h11"),
            nthreads=4,
            quad_decimate=tag_config.get("decimation", 1.0),
            quad_sigma=tag_config.get("blur", 0.0),
            refine_edges=int(tag_config.get("refine_edges", True)),
            decode_sharpening=0.25,
            debug=0,
        )

    def detect_tags(self, frame: np.ndarray) -> list:
        """Detect AprilTags in the given frame."""
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect tags
        detections = self.detector.detect(gray)

        # Filter detections by decision margin
        min_margin = self.config.get("apriltag", {}).get("min_decision_margin", 10.0)
        filtered_detections = [
            detection
            for detection in detections
            if detection.decision_margin >= min_margin
        ]

        # Filter by specific tag ID if specified
        target_tag_id = self.config.get("apriltag", {}).get("target_tag_id")
        if target_tag_id is not None:
            filtered_detections = [
                detection
                for detection in filtered_detections
                if detection.tag_id == target_tag_id
            ]

        return filtered_detections

    def find_matching_tags(self, detections1: list, detections2: list) -> list[tuple]:
        """Find matching tags between two camera views."""
        matches = []

        for det1 in detections1:
            for det2 in detections2:
                if det1.tag_id == det2.tag_id:
                    matches.append((det1, det2))
                    break

        return matches

    def extract_correspondences(
        self, matches: list[tuple]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D-2D correspondences and 3D points from matching tags."""
        points_3d = []
        points_2d_1 = []
        points_2d_2 = []

        for det1, det2 in matches:
            # Add 3D points (same for both cameras since they're in world coordinates)
            points_3d.append(self.tag_3d_points)

            # Add 2D points from both cameras
            points_2d_1.append(det1.corners.astype(np.float32))
            points_2d_2.append(det2.corners.astype(np.float32))

        if not points_3d:
            return None, None, None

        # Stack all points
        points_3d = np.vstack(points_3d)
        points_2d_1 = np.vstack(points_2d_1)
        points_2d_2 = np.vstack(points_2d_2)

        return points_3d, points_2d_1, points_2d_2

    def get_correspondences(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take two stereo images and return the extracted correspondences.

        Args:
            img1: First camera image as numpy array
            img2: Second camera image as numpy array

        Returns:
            Tuple of (points_3d, points_2d_1, points_2d_2) correspondences
        """
        if img1 is None or img2 is None:
            raise ValueError("Images cannot be None")

        # Warn if images have different shapes, but allow it
        if img1.shape != img2.shape:
            logger.warning(
                f"Images have different shapes: img1={img1.shape}, img2={img2.shape}. "
                "This is allowed but may affect detection quality."
            )

        # Detect tags in both images
        detections1 = self.detect_tags(img1)
        detections2 = self.detect_tags(img2)

        # Find matching tags
        matches = self.find_matching_tags(detections1, detections2)

        if len(matches) < 1:
            logger.debug("No matching tags found")
            return None, None, None

        logger.debug(f"Found {len(matches)} matching tags")
        # Extract correspondences
        points_3d, points_2d_1, points_2d_2 = self.extract_correspondences(matches)

        if points_3d is None:
            logger.debug("No correspondences found")
            return None, None, None

        return points_3d, points_2d_1, points_2d_2

    def cleanup(self):
        """Explicitly cleanup the AprilTag detector to avoid segmentation faults."""
        if hasattr(self, "detector") and self.detector is not None:
            # Set detector to None to avoid garbage collection issues
            self.detector = None


class CharucoDetector:
    """
    Detect Charuco boards in stereo images and extract correspondences.

    Charuco corner IDs are fixed by the board structure and are consistent across viewpoints:
    - Each corner has a unique ID (0 to N-1) based on its position on the board
    - Corner IDs are assigned sequentially, row by row from top-left
    - The same physical corner always has the same ID, regardless of viewing angle
    - This ensures consistent stereo matching between camera views
    """

    def __init__(self, config_path: str | None = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.dictionary, self.charuco_board, self.charuco_detector = (
            self._create_charuco_board()
        )

        # Generate 3D points for all Charuco corners using board structure
        # This ensures IDs match the board's corner ID assignment (0 to N-1)
        self.charuco_3d_points = self._generate_charuco_3d_points()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Config file {config_path} not found.")
            raise

    def _get_aruco_dictionary(self, dict_name: str):
        """Get ArUco dictionary by name."""
        dict_mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        }
        if dict_name not in dict_mapping:
            raise ValueError(
                f"Unknown dictionary name: {dict_name}. "
                f"Supported: {list(dict_mapping.keys())}"
            )
        return cv2.aruco.getPredefinedDictionary(dict_mapping[dict_name])

    def _create_charuco_board(self):
        """Create Charuco board and detector with configured parameters."""
        charuco_config = self.config.get("charuco", {})

        # Get board dimensions
        array_size = charuco_config.get("array_size", [5, 7])
        if isinstance(array_size, list) and len(array_size) == 2:
            squares_x, squares_y = array_size[0], array_size[1]
        else:
            raise ValueError(
                "array_size must be a list/tuple of 2 integers [squares_x, squares_y]"
            )

        # Get sizes in meters
        checker_size = charuco_config.get("checker_size")
        target_size = charuco_config.get("target_size")

        if checker_size is None or target_size is None:
            raise ValueError(
                "charuco config must include 'checker_size' and 'target_size' in meters"
            )

        # Get dictionary
        dict_name = charuco_config.get("dictionary", "DICT_4X4_50")
        dictionary = self._get_aruco_dictionary(dict_name)

        # Create Charuco board
        charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            checker_size,
            target_size,
            dictionary,
        )

        # Set legacy pattern for compatibility with corner interpolation in OpenCV 4.6.0+
        charuco_board.setLegacyPattern(True)

        # Create Charuco detector
        charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

        # Configure ArUco detector parameters for reliable marker detection
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 23
        aruco_params.minMarkerPerimeterRate = 0.01
        aruco_params.maxMarkerPerimeterRate = 4.0
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.adaptiveThreshConstant = 7
        aruco_params.minOtsuStdDev = 2.0
        aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        aruco_params.maxErroneousBitsInBorderRate = 0.4
        aruco_params.errorCorrectionRate = 0.6
        aruco_params.polygonalApproxAccuracyRate = 0.03
        aruco_params.minDistanceToBorder = 3
        charuco_detector.setDetectorParameters(aruco_params)

        # Configure Charuco parameters for corner interpolation
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.minMarkers = 0
        charuco_params.cameraMatrix = None
        charuco_params.distCoeffs = None
        charuco_params.tryRefineMarkers = True
        charuco_detector.setCharucoParameters(charuco_params)

        # Configure refine parameters for corner detection
        refine_params = cv2.aruco.RefineParameters()
        refine_params.minRepDistance = 0.1
        refine_params.errorCorrectionRate = 20.0
        refine_params.checkAllOrders = True
        charuco_detector.setRefineParameters(refine_params)

        return dictionary, charuco_board, charuco_detector

    def _generate_charuco_3d_points(self):
        """
        Generate 3D points for all Charuco corners using the board's getChessboardCorners().

        This ensures consistency with OpenCV's Charuco board structure:
        - Uses board.getChessboardCorners() to get the exact 3D points
        - Corner IDs are the index in the returned array (0 to N-1)
        - This matches the IDs returned by detectBoard() for consistent stereo matching

        Returns:
            Dictionary mapping corner_id (int) to 3D point (np.array of shape (3,))
        """
        # Use the board's getChessboardCorners() to get the exact 3D points
        # This ensures consistency with OpenCV's ID assignment
        board_corners = self.charuco_board.getChessboardCorners()

        # Corner IDs are the index in the board_corners array
        # This matches the IDs returned by detectBoard()
        id_to_3d = {}
        for corner_id in range(len(board_corners)):
            id_to_3d[corner_id] = board_corners[corner_id].astype(np.float32)

        logger.debug(
            f"Generated {len(id_to_3d)} Charuco corner 3D points from board structure "
            f"(board size: {self.charuco_board.getChessboardSize()})"
        )

        return id_to_3d

    def _quick_reject(self, gray: np.ndarray) -> bool:
        """
        Fast rejection stage: check edge density at low resolution.

        Rejects frames with low edge density (~90% of frames) in 1-2ms.

        Args:
            gray: Grayscale image

        Returns:
            True if frame should be rejected (low edge density)
        """
        small = cv2.resize(gray, None, fx=0.18, fy=0.18)
        edges = cv2.Canny(small, 60, 120)
        edge_ratio = np.count_nonzero(edges) / edges.size
        return edge_ratio < 0.06

    def detect_tags(
        self, frame: np.ndarray, debug: bool = False, return_marker_count: bool = False
    ) -> tuple:
        """
        Detect Charuco corners and IDs using multi-stage detection with fast rejection.

        Detection stages:
        - Stage 0: Fast rejection based on edge density
        - Stage 1: Multi-resolution detection (720p → 1080p → full res fallback)
        - Stage 2: Refine corners at full resolution

        Args:
            frame: Input image as numpy array
            debug: If True, print debug information
            return_marker_count: If True, return marker count as third element

        Returns:
            If return_marker_count=False: Tuple of (charuco_corners, charuco_ids) or (None, None) if not found
            If return_marker_count=True: Tuple of (charuco_corners, charuco_ids, num_markers) or (None, None, 0) if not found

        Note:
            Corner IDs are sequential from 0 to (squares_x-1)*(squares_y-1)-1, numbered
            row by row from top-left. These IDs are consistent across camera views for
            stereo matching.
        """
        # Initialize result variables
        result_corners = None
        result_ids = None
        num_markers_detected = 0

        if frame is None or frame.size == 0:
            if debug:
                logger.debug("Charuco detection: Empty frame")
        elif len(frame.shape) < 2:
            if debug:
                logger.debug("Charuco detection: Invalid image shape")
        else:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_full = frame

            h, w = gray_full.shape
            if h < 100 or w < 100:
                if debug:
                    logger.debug("Charuco detection: Image too small")
            elif self._quick_reject(gray_full):
                if debug:
                    logger.debug("Charuco detection: Rejected by edge density")
            else:
                # Stage 1: Multi-resolution detection with systematic fallback
                # Try resolutions in order: 720p → 1080p → full resolution
                # Use the first resolution that gives us corners, or the one with most markers if no corners

                resolutions_to_try = [
                    (720, "720p"),
                    (1080, "1080p"),
                    (h, "full"),
                ]

                best_result = None
                best_markers = 0
                best_corners = 0
                best_scale = 1.0
                best_height = h

                # Early exit: if no markers at 720p, try 1080p once, then reject if still nothing
                early_exit_after_1080 = False
                should_early_exit = False

                for target_height, res_name in resolutions_to_try:
                    if target_height == h:
                        gray_resized = gray_full
                        scale_factor = 1.0
                    else:
                        scale_factor = target_height / h
                        target_width = int(w * scale_factor)
                        target_width = target_width + (target_width % 2)
                        gray_resized = cv2.resize(
                            gray_full,
                            (target_width, target_height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    if debug:
                        if target_height == h:
                            logger.debug(
                                f"Charuco detection: Trying {res_name} ({w}x{h})"
                            )
                        else:
                            logger.debug(
                                f"Charuco detection: Trying {res_name} ({target_width}x{target_height})"
                            )

                    # Detect board at this resolution
                    charuco_corners, charuco_ids, markers_board, _ = (
                        self.charuco_detector.detectBoard(gray_resized)
                    )

                    num_markers = len(markers_board) if markers_board is not None else 0
                    num_corners = (
                        len(charuco_corners) if charuco_corners is not None else 0
                    )

                    if debug:
                        logger.debug(
                            f"Charuco detection: {res_name} - {num_markers} markers, {num_corners} corners"
                        )

                    # If we found corners, use this result
                    if num_corners >= 4:
                        if debug:
                            logger.debug(
                                f"Charuco detection: Using {res_name} result ({num_corners} corners)"
                            )
                        best_result = (charuco_corners, charuco_ids, markers_board)
                        best_scale = scale_factor
                        best_height = target_height
                        num_markers_detected = num_markers
                        break

                    # Track best result (most markers, or most corners if markers equal)
                    if num_markers > best_markers or (
                        num_markers == best_markers and num_corners > best_corners
                    ):
                        best_result = (charuco_corners, charuco_ids, markers_board)
                        best_markers = num_markers
                        best_corners = num_corners
                        best_scale = scale_factor
                        best_height = target_height
                        num_markers_detected = num_markers

                    # Early exit optimization: if no markers at 720p, try 1080p, then exit if still nothing
                    if target_height == 720 and num_markers == 0:
                        early_exit_after_1080 = True
                    elif (
                        target_height == 1080
                        and early_exit_after_1080
                        and num_markers == 0
                    ):
                        if debug:
                            logger.debug(
                                "Charuco detection: No markers at 720p or 1080p, rejecting early"
                            )
                        should_early_exit = True
                        break

                # Use best result found
                if should_early_exit or best_result is None:
                    if debug and not should_early_exit:
                        logger.debug(
                            "Charuco detection: No markers or corners found at any resolution"
                        )
                else:
                    charuco_corners, charuco_ids, markers_board = best_result
                    scale_factor = best_scale
                    target_height = best_height

                    if debug:
                        if charuco_corners is not None and len(charuco_corners) > 0:
                            logger.debug(
                                f"Charuco detection: Found {len(charuco_corners)} corners at {target_height}p"
                            )
                        else:
                            logger.debug(
                                f"Charuco detection: No corners interpolated at {target_height}p"
                            )

                    if (
                        charuco_corners is not None
                        and len(charuco_corners) > 0
                        and charuco_ids is not None
                        and len(charuco_ids) > 0
                    ):
                        if debug:
                            logger.debug(
                                f"Charuco detection: Found {len(charuco_corners)} corners, "
                                f"refining at full resolution ({w}x{h})"
                            )

                        # Stage 2: Refine at full resolution
                        if scale_factor != 1.0:
                            corners_fullres = charuco_corners / scale_factor
                        else:
                            corners_fullres = charuco_corners

                        if (
                            len(corners_fullres.shape) == 2
                            or corners_fullres.shape[1] != 1
                        ):
                            corners_fullres = corners_fullres.reshape(-1, 1, 2)

                        # Clip corners to image bounds before cornerSubPix to avoid assertion errors
                        img_h, img_w = gray_full.shape[:2]
                        corners_fullres[:, :, 0] = np.clip(
                            corners_fullres[:, :, 0], 0, img_w - 1
                        )
                        corners_fullres[:, :, 1] = np.clip(
                            corners_fullres[:, :, 1], 0, img_h - 1
                        )

                        criteria = (
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            20,
                            0.0015,
                        )
                        corners_refined = cv2.cornerSubPix(
                            gray_full, corners_fullres, (11, 11), (-1, -1), criteria
                        )

                        # Validate and filter invalid corner IDs
                        valid_corner_ids = set(self.charuco_3d_points.keys())
                        ids_flat = charuco_ids.flatten()
                        invalid_ids = [
                            id_val
                            for id_val in ids_flat
                            if int(id_val) not in valid_corner_ids
                        ]

                        if invalid_ids:
                            if debug:
                                logger.warning(
                                    f"Charuco detection: Found {len(invalid_ids)} invalid corner IDs: {invalid_ids[:10]}"
                                    + (
                                        f" (showing first 10 of {len(invalid_ids)})"
                                        if len(invalid_ids) > 10
                                        else ""
                                    )
                                )
                            valid_mask = np.array(
                                [int(id_val) in valid_corner_ids for id_val in ids_flat]
                            )
                            if np.sum(valid_mask) == 0:
                                if debug:
                                    logger.debug(
                                        "Charuco detection: No valid corner IDs after filtering"
                                    )
                            else:
                                corners_refined = corners_refined[valid_mask]
                                charuco_ids = charuco_ids[valid_mask]
                                result_corners = corners_refined
                                result_ids = charuco_ids
                        else:
                            result_corners = corners_refined
                            result_ids = charuco_ids

                        if debug:
                            logger.debug(
                                f"Charuco detection: Returning {len(result_corners)} corners with IDs: "
                                f"{sorted(result_ids.flatten())[:10]}..."
                                + (
                                    f" (showing first 10 of {len(result_corners)})"
                                    if len(result_corners) > 10
                                    else ""
                                )
                            )
                            if return_marker_count:
                                logger.debug(
                                    f"Charuco detection: Detected {num_markers_detected} markers"
                                )
                    elif debug:
                        logger.debug(
                            f"Charuco detection: Not found at {target_height}p"
                        )

        # Single return statement at the end
        if return_marker_count:
            return result_corners, result_ids, num_markers_detected
        return result_corners, result_ids

    def find_matching_tags(self, detections1: tuple, detections2: tuple) -> list[tuple]:
        """
        Find matching Charuco corners between two camera views by corner ID.

        Each corner has a unique ID assigned by OpenCV's CharucoDetector, which
        corresponds to its position on the board. This method matches corners
        with the same ID between the two views.

        Args:
            detections1: Tuple of (corners, ids) from first camera
            detections2: Tuple of (corners, ids) from second camera

        Returns:
            List of tuples (corner_id, index_in_detections1, index_in_detections2)
            where corner_id is the unique corner ID, and indices are the positions
            in the respective corner arrays.
        """
        # Handle None returns from detect_tags
        if detections1 is None or detections1[0] is None or detections1[1] is None:
            return []
        if detections2 is None or detections2[0] is None or detections2[1] is None:
            return []

        _corners1, ids1 = detections1
        _corners2, ids2 = detections2

        if ids1 is None or ids2 is None or len(ids1) == 0 or len(ids2) == 0:
            return []

        # Find matching corner IDs
        matches = []
        ids1_flat = ids1.flatten()
        ids2_flat = ids2.flatten()

        # Create mapping from ID to corner index
        id_to_idx1 = {int(id): idx for idx, id in enumerate(ids1_flat)}
        id_to_idx2 = {int(id): idx for idx, id in enumerate(ids2_flat)}

        # Find common IDs
        common_ids = set(id_to_idx1.keys()) & set(id_to_idx2.keys())

        # Validate that all IDs are valid board corner IDs
        valid_corner_ids = set(self.charuco_3d_points.keys())
        invalid_ids_cam1 = set(id_to_idx1.keys()) - valid_corner_ids
        invalid_ids_cam2 = set(id_to_idx2.keys()) - valid_corner_ids

        if invalid_ids_cam1:
            logger.warning(
                f"Camera 1 detected {len(invalid_ids_cam1)} invalid corner IDs: "
                f"{sorted(invalid_ids_cam1)[:10]}"
                + (
                    f" (showing first 10 of {len(invalid_ids_cam1)})"
                    if len(invalid_ids_cam1) > 10
                    else ""
                )
            )
        if invalid_ids_cam2:
            logger.warning(
                f"Camera 2 detected {len(invalid_ids_cam2)} invalid corner IDs: "
                f"{sorted(invalid_ids_cam2)[:10]}"
                + (
                    f" (showing first 10 of {len(invalid_ids_cam2)})"
                    if len(invalid_ids_cam2) > 10
                    else ""
                )
            )

        # Filter to only valid IDs
        common_ids = common_ids & valid_corner_ids

        for corner_id in common_ids:
            idx1 = id_to_idx1[corner_id]
            idx2 = id_to_idx2[corner_id]
            # Validate indices are within bounds
            if idx1 < len(ids1_flat) and idx2 < len(ids2_flat):
                matches.append((corner_id, idx1, idx2))
            else:
                logger.warning(
                    f"Invalid index for corner ID {corner_id}: idx1={idx1}, idx2={idx2}"
                )

        if matches:
            logger.debug(
                f"Matched {len(matches)} corners with valid IDs: {sorted([m[0] for m in matches])[:10]}"
                + (
                    f" (showing first 10 of {len(matches)})"
                    if len(matches) > 10
                    else ""
                )
            )
        else:
            logger.warning(
                f"No valid matching corners found. Camera 1 IDs: {sorted(id_to_idx1.keys())[:10]}, "
                f"Camera 2 IDs: {sorted(id_to_idx2.keys())[:10]}"
            )

        return matches

    def extract_correspondences(
        self, matches: list[tuple], detections1: tuple, detections2: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D-2D correspondences and 3D points from matching Charuco corners."""
        # Handle None returns from detect_tags
        if detections1 is None or detections1[0] is None or detections1[1] is None:
            return None, None, None
        if detections2 is None or detections2[0] is None or detections2[1] is None:
            return None, None, None

        corners1, _ids1 = detections1
        corners2, _ids2 = detections2

        if not matches:
            return None, None, None

        points_3d = []
        points_2d_1 = []
        points_2d_2 = []

        for corner_id, idx1, idx2 in matches:
            # Get 2D corner positions
            # Handle different corner array shapes (N, 1, 2) or (N, 2)
            corner_2d_1 = corners1[idx1]
            corner_2d_2 = corners2[idx2]

            # Reshape if needed
            if len(corner_2d_1.shape) == 2 and corner_2d_1.shape[0] == 1:
                corner_2d_1 = corner_2d_1[0]
            elif len(corner_2d_1.shape) == 3:
                corner_2d_1 = corner_2d_1.reshape(-1, 2)[0]

            if len(corner_2d_2.shape) == 2 and corner_2d_2.shape[0] == 1:
                corner_2d_2 = corner_2d_2[0]
            elif len(corner_2d_2.shape) == 3:
                corner_2d_2 = corner_2d_2.reshape(-1, 2)[0]

            corner_2d_1 = corner_2d_1.astype(np.float32)
            corner_2d_2 = corner_2d_2.astype(np.float32)

            # Get 3D position from the pre-computed mapping
            corner_id_int = int(corner_id)
            if corner_id_int not in self.charuco_3d_points:
                logger.warning(
                    f"Corner ID {corner_id_int} not found in board 3D points, skipping"
                )
                continue

            corner_3d = self.charuco_3d_points[corner_id_int].astype(np.float32)

            points_3d.append(corner_3d)
            points_2d_1.append(corner_2d_1)
            points_2d_2.append(corner_2d_2)

        if not points_3d:
            return None, None, None

        # Validate all arrays have the same length
        if len(points_3d) != len(points_2d_1) or len(points_3d) != len(points_2d_2):
            logger.error(
                f"Mismatch in correspondence arrays: "
                f"points_3d={len(points_3d)}, points_2d_1={len(points_2d_1)}, "
                f"points_2d_2={len(points_2d_2)}"
            )
            return None, None, None

        # Stack all points
        points_3d = np.vstack(points_3d)
        points_2d_1 = np.vstack(points_2d_1)
        points_2d_2 = np.vstack(points_2d_2)

        # Validate stacked arrays have correct shape
        if points_3d.shape[1] != 3:
            logger.error(f"Invalid 3D points shape: {points_3d.shape}, expected (N, 3)")
            return None, None, None
        if points_2d_1.shape[1] != 2 or points_2d_2.shape[1] != 2:
            logger.error(
                f"Invalid 2D points shape: points_2d_1={points_2d_1.shape}, "
                f"points_2d_2={points_2d_2.shape}, expected (N, 2)"
            )
            return None, None, None

        return points_3d, points_2d_1, points_2d_2

    def get_correspondences(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take two stereo images and return the extracted correspondences.

        Args:
            img1: First camera image as numpy array
            img2: Second camera image as numpy array

        Returns:
            Tuple of (points_3d, points_2d_1, points_2d_2) correspondences
        """
        if img1 is None or img2 is None:
            raise ValueError("Images cannot be None")

        # Warn if images have different shapes, but allow it
        if img1.shape != img2.shape:
            logger.warning(
                f"Images have different shapes: img1={img1.shape}, img2={img2.shape}. "
                "This is allowed but may affect detection quality."
            )

        # Detect Charuco corners in both images
        detections1 = self.detect_tags(img1)
        detections2 = self.detect_tags(img2)

        # Find matching corners
        matches = self.find_matching_tags(detections1, detections2)

        # Minimum threshold for reliable calibration
        MIN_MATCHING_CORNERS = 10
        if len(matches) < MIN_MATCHING_CORNERS:
            logger.debug(
                f"Only {len(matches)} matching corners found, need at least {MIN_MATCHING_CORNERS} "
                "for reliable calibration"
            )
            return None, None, None

        logger.debug(f"Found {len(matches)} matching Charuco corners")
        # Extract correspondences
        points_3d, points_2d_1, points_2d_2 = self.extract_correspondences(
            matches, detections1, detections2
        )

        if points_3d is None:
            logger.debug("No correspondences found")
            return None, None, None

        return points_3d, points_2d_1, points_2d_2

    def cleanup(self):
        """Explicitly cleanup the Charuco detector."""
        if hasattr(self, "charuco_detector") and self.charuco_detector is not None:
            # Set detector to None to avoid garbage collection issues
            self.charuco_detector = None
            self.charuco_board = None
            self.dictionary = None


def get_summary(results: dict[str, Any]) -> str:
    """Get a summary of the calibration results."""
    summary = ""

    summary += "\n" + "=" * 50
    summary += "STEREO CALIBRATION RESULTS"
    summary += "=" * 50
    summary += f"\nSuccess: {results['success']}"
    summary += f"\nReprojection Error: {results['reprojection_error']:.6f}"
    summary += f"\nNumber of Correspondences: {results['num_correspondences']}"

    # Extract translation and rotation info
    T = np.array(results["translation_vector"])
    R = np.array(results["rotation_matrix"])

    # Convert rotation matrix to Euler angles
    euler_angles = _rotation_matrix_to_euler_angles(R)
    summary += f"\nRotation (roll, pitch, yaw): ({euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°)"

    summary += f"\nTranslation: ({T[0, 0]:.6f}, {T[1, 0]:.6f}, {T[2, 0]:.6f}) m"
    # Calculate baseline distance
    baseline = np.linalg.norm(T)
    summary += f"\nComputed baseline distance: {baseline:.6f} m"

    return summary


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    # Extract Euler angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.degrees([roll, pitch, yaw])
