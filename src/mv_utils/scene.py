import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from loguru import logger

from .extrinsics_calibration import (
    CharucoDetector,
    StereoTagDetector,
    calibrate_stereo_many,
    get_summary,
)
from .intrinsics_calibration import IntrinsicCalibration
from .stereo_data_folder_structure import load_scene_folder_structure
from .utils import load_parameters
from .video_utils import VideoReader, get_unique_video_name


class Scene:
    def __init__(self, scene_name: str, stereo_data_folder_path: str | None = None):
        self.scene_name = scene_name
        self.scene_folder_structure = load_scene_folder_structure(
            scene_name, stereo_data_folder_path
        )
        self.parameters = self.get_parameters()
        self.cameras = ["camera_1", "camera_2"]
        self.sync_frame_offset = self.get_sync_frame_offset()
        self.calibration = None

    def get_parameters(self):
        with open(
            os.path.join(
                self.scene_folder_structure.folder_path,
                self.scene_name,
                "parameters.yml",
            )
        ) as f:
            parameters = yaml.safe_load(f)
        return parameters

    def get_sync_frame_offset(self):
        parameters = self.parameters
        camera_1_sync_event_list_F = parameters["camera_1"]["sync_event_time_F"]
        camera_2_sync_event_list_F = parameters["camera_2"]["sync_event_time_F"]

        diff = np.array(camera_1_sync_event_list_F) - np.array(
            camera_2_sync_event_list_F
        )
        if not np.all(diff == diff[0]):
            logger.warning(f"frame offset diff is not constant: {diff}")
        diff_mean = np.mean(diff)
        return int(diff_mean)

    def get_video_paths(self):
        """Get the video paths for cameras of self.cameras"""
        video_paths = {}
        for camera_name in self.cameras:
            folder_path = os.path.join(
                self.scene_folder_structure.folder_path, self.scene_name, camera_name
            )
            video_name = get_unique_video_name(folder_path)
            if video_name is None:
                raise FileNotFoundError(f"No video file found in {folder_path}")
            video_paths[camera_name] = os.path.join(folder_path, video_name)
        return video_paths

    def create_calibration(self) -> "Calibration":
        self.calibration = Calibration(
            scene_name=self.scene_name,
            scene_folder_structure=self.scene_folder_structure,
            parameters=self.parameters,
            sync_frame_offset=self.sync_frame_offset,
        )
        return self.calibration

    def cleanup(self):
        """Explicitly cleanup resources to avoid segmentation faults."""
        self.calibration.cleanup()


class Calibration:
    def __init__(
        self,
        scene_name: str,
        scene_folder_structure,
        parameters: dict[str, Any],
        sync_frame_offset: int,
    ):
        """Calibrate extrinsics and intrinsics for a scene.

        Args:
            scene_name: Name of the scene (e.g. "scene_5")
            scene_folder_structure: Folder structure object returned by load_scene_folder_structure
            parameters: Parsed parameters.yml content for the scene
            sync_frame_offset: Frame offset between the two cameras
        """
        self.scene_name = scene_name
        self.scene_folder_structure = scene_folder_structure
        self.parameters = parameters
        self.sync_frame_offset = sync_frame_offset
        self.cameras = ["camera_1", "camera_2"]

        video_paths = self._get_video_paths()

        self.video_reader_1 = VideoReader.open_video_file(video_paths["camera_1"])
        self.video_reader_2 = VideoReader.open_video_file(video_paths["camera_2"])

        self.video_1 = self.video_reader_1.video
        self.video_2 = self.video_reader_2.video
        self.save_path = os.path.join(
            self.scene_folder_structure.folder_path, self.scene_name, "calibration.json"
        )
        # Initialize time offset attributes (will be set if FPS differ)
        self._time_offset_seconds = 0.0
        self._fps_ratio = 1.0
        self.extrinsics_calibration_pattern_specs = (
            self.load_extrinsics_calibration_pattern_specs()
        )
        config_path = os.path.join(
            self.scene_folder_structure.folder_path,
            self.scene_name,
            "extrinsics_calibration_pattern_specs.yml",
        )
        # Determine which detector to use based on pattern_type or config sections
        pattern_type = self.extrinsics_calibration_pattern_specs.get("pattern_type")
        has_charuco = "charuco" in self.extrinsics_calibration_pattern_specs
        has_apriltag = "apriltag" in self.extrinsics_calibration_pattern_specs

        if pattern_type == "charuco" or (
            pattern_type is None and has_charuco and not has_apriltag
        ):
            self.stereo_tag_detector = CharucoDetector(config_path=config_path)
        else:
            # Default to AprilTag for backward compatibility
            self.stereo_tag_detector = StereoTagDetector(config_path=config_path)
        self.check_video_fps()
        self.check_resolution()

        # Recalculate sync offset based on actual FPS if they differ
        self.sync_frame_offset = self._calculate_sync_offset()
        self.image_numbers_list = self.get_image_numbers_list()

    def _get_video_paths(self) -> dict[str, str]:
        """Get the video paths for both cameras within this scene."""
        video_paths: dict[str, str] = {}
        for camera_name in self.cameras:
            folder_path = os.path.join(
                self.scene_folder_structure.folder_path, self.scene_name, camera_name
            )
            video_name = get_unique_video_name(folder_path)
            if video_name is None:
                raise FileNotFoundError(f"No video file found in {folder_path}")
            video_paths[camera_name] = os.path.join(folder_path, video_name)
        return video_paths

    def _calculate_sync_offset(self) -> int:
        """
        Calculate sync frame offset, handling different FPS by converting to time.

        Returns:
            Frame offset for camera_2 relative to camera_1 (in camera_2's frame numbers)
        """
        camera_1_sync_event_list_F = self.parameters["camera_1"]["sync_event_time_F"]
        camera_2_sync_event_list_F = self.parameters["camera_2"]["sync_event_time_F"]

        fps_1 = self.video_1.fps
        fps_2 = self.video_2.fps

        # Check if FPS are effectively the same
        if abs(fps_1 - fps_2) < 0.01:
            # Same FPS: use frame-based calculation (original logic)
            diff = np.array(camera_1_sync_event_list_F) - np.array(
                camera_2_sync_event_list_F
            )
            if not np.allclose(diff, diff[0], atol=1):
                logger.warning(f"Frame offset diff is not constant: {diff}")
            diff_mean = np.mean(diff)
            # Set time offset to 0 for same FPS case
            self._time_offset_seconds = 0.0
            self._fps_ratio = 1.0
            return int(diff_mean)
        else:
            # Different FPS: convert to time, then back to frames
            # Convert sync events to time (seconds)
            times_1 = np.array(camera_1_sync_event_list_F) / fps_1
            times_2 = np.array(camera_2_sync_event_list_F) / fps_2

            # Calculate time offset
            time_offsets = times_1 - times_2
            if not np.allclose(time_offsets, time_offsets[0], atol=0.01):
                logger.warning(
                    f"Time offset diff is not constant: {time_offsets} seconds"
                )
            mean_time_offset = np.mean(time_offsets)

            # For frame selection, we need to know: if camera_1 is at frame F1,
            # what frame F2 in camera_2 corresponds to the same time?
            # time_1 = F1 / fps_1
            # time_2 = F2 / fps_2
            # time_1 = time_2 + mean_time_offset
            # F1 / fps_1 = F2 / fps_2 + mean_time_offset
            # F2 = (F1 / fps_1 - mean_time_offset) * fps_2
            # F2 = F1 * (fps_2 / fps_1) - mean_time_offset * fps_2
            # offset = F2 - F1 = F1 * (fps_2 / fps_1 - 1) - mean_time_offset * fps_2

            # Store time offset and FPS ratio for use in get_image_numbers_list
            self._time_offset_seconds = mean_time_offset
            self._fps_ratio = fps_2 / fps_1

            # Return a frame offset that will be used as a base
            # The actual offset depends on the frame number, so we'll calculate it per frame
            # For now, return the offset at frame 0 as a reference
            frame_offset_at_zero = -mean_time_offset * fps_2
            logger.debug(
                f"Time-based sync: offset={mean_time_offset:.4f}s, "
                f"FPS ratio={self._fps_ratio:.4f}, "
                f"frame_offset_at_zero={frame_offset_at_zero:.2f}"
            )
            return int(frame_offset_at_zero)

    def get_image_numbers_list(self, temporal_calib_step_sec: float = 1):
        """
        Get image numbers list for both cameras, handling different FPS.

        Args:
            temporal_calib_step_sec: Time step between calibration frames in seconds

        Returns:
            Tuple of (camera_1_image_numbers_list, camera_2_image_numbers_list)
        """
        fps_1 = self.video_1.fps
        fps_2 = self.video_2.fps

        # Check if FPS are effectively the same
        if abs(fps_1 - fps_2) < 0.01:
            # Same FPS: use frame-based calculation (original logic)
            step = int(fps_1 * temporal_calib_step_sec)
            camera_1_image_numbers_list = list(
                range(
                    self.parameters["camera_1"]["sync_calib_start_frame"],
                    self.parameters["camera_1"]["sync_calib_end_frame"],
                    step,
                )
            )
            # For camera_2, subtract the sync_frame_offset
            camera_2_image_numbers_list = [
                frame - self.sync_frame_offset for frame in camera_1_image_numbers_list
            ]
        else:
            # Different FPS: use time-based calculation
            # Start and end times in seconds for camera_1
            start_time_1 = self.parameters["camera_1"]["sync_calib_start_frame"] / fps_1
            end_time_1 = self.parameters["camera_1"]["sync_calib_end_frame"] / fps_1

            # Generate time points
            time_points = np.arange(start_time_1, end_time_1, temporal_calib_step_sec)

            # Convert to frame numbers for camera_1
            camera_1_image_numbers_list = [int(t * fps_1) for t in time_points]

            # Convert to frame numbers for camera_2 using time offset
            # time_2 = time_1 - time_offset
            camera_2_image_numbers_list = [
                int((t - self._time_offset_seconds) * fps_2) for t in time_points
            ]

            # Filter out negative frame numbers
            valid_pairs = [
                (f1, f2)
                for f1, f2 in zip(
                    camera_1_image_numbers_list,
                    camera_2_image_numbers_list,
                    strict=False,
                )
                if f2 >= 0
            ]
            if len(valid_pairs) < len(camera_1_image_numbers_list):
                logger.warning(
                    f"Filtered out {len(camera_1_image_numbers_list) - len(valid_pairs)} "
                    f"frame pairs with negative camera_2 frame numbers"
                )
            camera_1_image_numbers_list, camera_2_image_numbers_list = (
                zip(*valid_pairs, strict=False) if valid_pairs else ([], [])
            )
            camera_1_image_numbers_list = list(camera_1_image_numbers_list)
            camera_2_image_numbers_list = list(camera_2_image_numbers_list)

        return camera_1_image_numbers_list, camera_2_image_numbers_list

    def check_video_fps(self):
        """Check video FPS and log warning if different."""
        if (
            abs(self.video_1.fps - self.video_2.fps) > 0.01
        ):  # Allow small floating point differences
            logger.warning(
                f"Video FPS differ: camera_1={self.video_1.fps:.2f} fps, "
                f"camera_2={self.video_2.fps:.2f} fps. Using time-based synchronization."
            )
        else:
            logger.debug(f"Video FPS match: {self.video_1.fps:.2f} fps")

    def check_resolution(self):
        """Check video resolutions and warn if cameras have different resolutions."""
        logger.debug(f"Video 1 resolution: {self.video_1.resolution} (width, height)")
        logger.debug(f"Video 2 resolution: {self.video_2.resolution} (width, height)")

        # if not in landscape orientation, rotate 90 deg
        if self.video_1.resolution[0] < self.video_1.resolution[1]:
            logger.debug(
                f"Video 1 is not in landscape orientation, rotating 90 deg : {self.video_1.resolution} -> {self.video_1.resolution[1], self.video_1.resolution[0]}"
            )
        if self.video_2.resolution[0] < self.video_2.resolution[1]:
            logger.debug(
                f"Video 2 is not in landscape orientation, rotating 90 deg : {self.video_2.resolution} -> {self.video_2.resolution[1], self.video_2.resolution[0]}"
            )

        # Warn if cameras have different resolutions (but allow it)
        if (
            self.video_1.resolution != self.video_2.resolution
            and self.video_1.resolution
            != (
                self.video_2.resolution[1],
                self.video_2.resolution[0],
            )
        ):
            logger.warning(
                f"Cameras have different resolutions: camera_1={self.video_1.resolution}, "
                f"camera_2={self.video_2.resolution}. This is allowed but may affect calibration quality."
            )

        # Check that intrinsic calibration video resolution matches extrinsic for each camera
        self._check_intrinsic_extrinsic_resolution_match()

    def _check_intrinsic_extrinsic_resolution_match(self):
        """Check that intrinsic calibration video resolution matches extrinsic calibration video resolution for each camera."""
        calibration_folder_path = os.path.join(
            self.scene_folder_structure.folder_path,
            self.scene_folder_structure.get_calibration_intrinsics_folder_name(),
        )

        # Check camera_1
        calibration_camera_1_path = os.path.join(calibration_folder_path, "camera_1")
        video_name_1 = get_unique_video_name(calibration_camera_1_path)
        if video_name_1 is not None:
            intrinsic_video_path_1 = os.path.join(
                calibration_camera_1_path, video_name_1
            )
            intrinsic_video_reader_1 = VideoReader.open_video_file(
                intrinsic_video_path_1
            )
            intrinsic_resolution_1 = intrinsic_video_reader_1.video.resolution
            extrinsic_resolution_1 = self.video_1.resolution

            assert (
                intrinsic_resolution_1 == extrinsic_resolution_1
                or intrinsic_resolution_1
                == (extrinsic_resolution_1[1], extrinsic_resolution_1[0])
            ), (
                f"Camera 1: Intrinsic calibration video resolution {intrinsic_resolution_1} "
                f"does not match extrinsic calibration video resolution {extrinsic_resolution_1}"
            )
            intrinsic_video_reader_1.release()
            logger.debug(
                f"Camera 1: Intrinsic and extrinsic resolutions match: {intrinsic_resolution_1}"
            )
        else:
            logger.warning(
                "Camera 1: No intrinsic calibration video found, skipping resolution check"
            )

        # Check camera_2
        calibration_camera_2_path = os.path.join(calibration_folder_path, "camera_2")
        video_name_2 = get_unique_video_name(calibration_camera_2_path)
        if video_name_2 is not None:
            intrinsic_video_path_2 = os.path.join(
                calibration_camera_2_path, video_name_2
            )
            intrinsic_video_reader_2 = VideoReader.open_video_file(
                intrinsic_video_path_2
            )
            intrinsic_resolution_2 = intrinsic_video_reader_2.video.resolution
            extrinsic_resolution_2 = self.video_2.resolution

            assert (
                intrinsic_resolution_2 == extrinsic_resolution_2
                or intrinsic_resolution_2
                == (extrinsic_resolution_2[1], extrinsic_resolution_2[0])
            ), (
                f"Camera 2: Intrinsic calibration video resolution {intrinsic_resolution_2} "
                f"does not match extrinsic calibration video resolution {extrinsic_resolution_2}"
            )
            intrinsic_video_reader_2.release()
            logger.debug(
                f"Camera 2: Intrinsic and extrinsic resolutions match: {intrinsic_resolution_2}"
            )
        else:
            logger.warning(
                "Camera 2: No intrinsic calibration video found, skipping resolution check"
            )

    def load_extrinsics_calibration_pattern_specs(self):
        return load_parameters(
            os.path.join(
                self.scene_folder_structure.folder_path,
                self.scene_name,
                "extrinsics_calibration_pattern_specs.yml",
            )
        )

    def compute_extrinsics_calibration(self, recompute_intrinsics: bool = False):
        # STEP 1: Determine target resolution from EXIF-corrected extrinsics videos FIRST
        # This must be done before intrinsics calibration so we can crop intrinsics frames to match
        camera_1_image_numbers_list, camera_2_image_numbers_list = (
            self.get_image_numbers_list()
        )

        # Determine target resolution (smaller camera's resolution)
        # Get sample frames to determine resolutions
        if not camera_1_image_numbers_list or not camera_2_image_numbers_list:
            logger.error("No image numbers available for calibration")
            return None

        sample_1_frames = self.video_reader_1.get_frames(
            [camera_1_image_numbers_list[0]]
        )
        sample_2_frames = self.video_reader_2.get_frames(
            [camera_2_image_numbers_list[0]]
        )

        if not sample_1_frames or not sample_2_frames:
            logger.error("Could not read sample frames to determine resolution")
            return None

        camera_1_sample = sample_1_frames[0]
        camera_2_sample = sample_2_frames[0]

        # IMPORTANT: Get processed frame sizes (after EXIF transpose and rotation)
        # Frames from VideoReader.get_frames() are already orientation-corrected:
        # - EXIF transpose is applied first
        # - Then 90deg clockwise rotation if portrait (height > width)
        # We MUST use these processed frame sizes, not the original video resolution
        H1, W1 = camera_1_sample.shape[
            :2
        ]  # shape is (height, width) after EXIF correction
        H2, W2 = camera_2_sample.shape[
            :2
        ]  # shape is (height, width) after EXIF correction

        # Target resolution is the smaller camera's resolution (min of both dimensions)
        # IMPORTANT: H1, W1, H2, W2 are EXIF-corrected resolutions (from VideoReader processed frames)
        target_H = min(H1, H2)
        target_W = min(W1, W2)
        logger.info(
            f"Camera resolutions (EXIF-corrected): camera_1={W1}x{H1}, camera_2={W2}x{H2}, "
            f"target={target_W}x{target_H}"
        )

        # STEP 2: Compute intrinsics with target resolution (crops frames to match)
        if recompute_intrinsics:
            self._compute_intrinsics(
                camera_name="camera_1", target_H=target_H, target_W=target_W
            )
            self._compute_intrinsics(
                camera_name="camera_2", target_H=target_H, target_W=target_W
            )
            # Load the computed intrinsics into object attributes
            self._load_intrinsics()
        else:
            try:
                self._load_intrinsics()
            except:
                logger.info("Intrinsics not found, computing them...")
                self._compute_intrinsics(
                    camera_name="camera_1", target_H=target_H, target_W=target_W
                )
                self._compute_intrinsics(
                    camera_name="camera_2", target_H=target_H, target_W=target_W
                )
                logger.debug("Intrinsics computed, loading them...")
                self._load_intrinsics()

        # Import crop function for camera resolution matching
        from stereo.image import crop_to_match_resolution

        successful_pairs = []
        successful_detections = []  # Store detections for visualization
        object_points_list = []
        image_points1_list = []
        image_points2_list = []
        image_size = None

        for camera_1_image_number, camera_2_image_number in zip(
            camera_1_image_numbers_list, camera_2_image_numbers_list, strict=False
        ):
            camera_1_frames = self.video_reader_1.get_frames([camera_1_image_number])
            camera_2_frames = self.video_reader_2.get_frames([camera_2_image_number])

            if not camera_1_frames or not camera_2_frames:
                continue

            # Frames from VideoReader are already orientation-corrected:
            # - EXIF transpose is applied first
            # - Then 90deg clockwise rotation if portrait (height > width)
            # This ensures both intrinsics and extrinsics use the same frame orientation
            camera_1_image = camera_1_frames[0]
            camera_2_image = camera_2_frames[0]

            # IMPORTANT: Use processed frame sizes (after EXIF/rotation) for cropping
            # These frames are already orientation-corrected by VideoReader
            H1_current, W1_current = camera_1_image.shape[
                :2
            ]  # (height, width) after EXIF correction

            # Crop camera_1 to match target resolution if needed
            if H1_current != target_H or W1_current != target_W:
                logger.debug(
                    f"Cropping camera_1 from {W1_current}x{H1_current} to target {target_W}x{target_H}"
                )
                camera_1_image = crop_to_match_resolution(
                    camera_1_image, target_H, target_W
                )
                H1_current, W1_current = camera_1_image.shape[:2]

            # Crop camera_2 to match target resolution if needed
            H2_current, W2_current = camera_2_image.shape[
                :2
            ]  # (height, width) after EXIF correction
            if H2_current != target_H or W2_current != target_W:
                logger.debug(
                    f"Cropping camera_2 from {W2_current}x{H2_current} to target {target_W}x{target_H}"
                )
                camera_2_image = crop_to_match_resolution(
                    camera_2_image, target_H, target_W
                )
                H2_current, W2_current = camera_2_image.shape[:2]

            # Detect corners (for Charuco) to save for visualization
            if isinstance(self.stereo_tag_detector, CharucoDetector):
                detections1 = self.stereo_tag_detector.detect_tags(camera_1_image)
                detections2 = self.stereo_tag_detector.detect_tags(camera_2_image)
            else:
                detections1 = None
                detections2 = None

            p3d, p2d1, p2d2 = self.stereo_tag_detector.get_correspondences(
                camera_1_image, camera_2_image
            )
            if p3d is None or p2d1 is None or p2d2 is None:
                continue

            successful_pairs.append((camera_1_image_number, camera_2_image_number))
            successful_detections.append((detections1, detections2))

            object_points_list.append(p3d)
            image_points1_list.append(p2d1)
            image_points2_list.append(p2d2)
            if image_size is None:
                # Use the normalized frame size (after cropping if needed)
                # Both cameras should now have the same resolution (target_W x target_H)
                # This is the EXIF-corrected resolution that will be stored in calibration JSON
                image_size = (target_W, target_H)
                logger.info(
                    f"Using normalized image_size for calibration: {image_size} "
                    f"(width x height, EXIF-corrected)"
                )

        results = calibrate_stereo_many(
            object_points_list,
            image_points1_list,
            image_points2_list,
            self.camera_matrix1,
            self.dist_coeffs1,
            self.camera_matrix2,
            self.dist_coeffs2,
            image_size,
        )

        self.save_successful_pairs(successful_pairs)
        self.save_calibration(results)
        return results

    def load_calibration(self):
        with open(self.save_path) as f:
            return json.load(f)

    def save_calibration(self, results: dict[str, Any]):
        with open(self.save_path, "w") as f:
            json.dump(results, f, indent=2)

    def save_successful_pairs(self, successful_pairs: list[tuple[int, int]]):
        """
        Save successful calibration pairs with rendered detections to output/scene_x/successful_pairs/.

        Saves:
        - Rendered images with detected corners at full resolution
        - pairs.json file with list of successful pairs

        Note: Frames are already orientation-corrected by VideoReader (EXIF transpose + 90deg rotation
        if portrait), so both intrinsics and extrinsics calibration use the same rotated frames.
        """
        if not successful_pairs:
            logger.debug("No successful pairs to save")
            return

        # Save to output/scene_x/successful_pairs/
        output_base = Path("output")
        path = output_base / self.scene_name / "successful_pairs"
        path.mkdir(parents=True, exist_ok=True)

        # Check if detector supports Charuco (has detect_tags method that returns corners and ids)
        is_charuco = isinstance(self.stereo_tag_detector, CharucoDetector)

        for i, (camera_1_image_number, camera_2_image_number) in enumerate(
            successful_pairs
        ):
            camera_1_frames = self.video_reader_1.get_frames([camera_1_image_number])
            camera_2_frames = self.video_reader_2.get_frames([camera_2_image_number])

            if not camera_1_frames or not camera_2_frames:
                continue

            # Frames from VideoReader are already orientation-corrected (EXIF + rotation if needed)
            camera_1_image = camera_1_frames[0].copy()
            camera_2_image = camera_2_frames[0].copy()

            # Detect and render corners if using Charuco detector
            if is_charuco:
                corners1, ids1 = self.stereo_tag_detector.detect_tags(camera_1_image)
                corners2, ids2 = self.stereo_tag_detector.detect_tags(camera_2_image)

                # Render detections using OpenCV's drawDetectedCornersCharuco
                if corners1 is not None and ids1 is not None and len(corners1) > 0:
                    # Ensure corners are in correct format (Nx1x2) for drawing
                    if len(corners1.shape) == 2:
                        corners1_draw = corners1.reshape(-1, 1, 2)
                    else:
                        corners1_draw = corners1
                    cv2.aruco.drawDetectedCornersCharuco(
                        camera_1_image, corners1_draw, ids1, (0, 255, 0)
                    )

                if corners2 is not None and ids2 is not None and len(corners2) > 0:
                    # Ensure corners are in correct format (Nx1x2) for drawing
                    if len(corners2.shape) == 2:
                        corners2_draw = corners2.reshape(-1, 1, 2)
                    else:
                        corners2_draw = corners2
                    cv2.aruco.drawDetectedCornersCharuco(
                        camera_2_image, corners2_draw, ids2, (0, 255, 0)
                    )

            # Save rendered images at full resolution
            cv2.imwrite(str(path / f"pair_{i}_camera1.png"), camera_1_image)
            cv2.imwrite(str(path / f"pair_{i}_camera2.png"), camera_2_image)

        # Save pairs.json file
        pairs_json_path = path / "pairs.json"
        with open(pairs_json_path, "w") as f:
            json.dump(successful_pairs, f, indent=2)

        logger.info(
            f"Saved {len(successful_pairs)} successful pairs with rendered detections at full resolution to {path}"
        )

    def get_calibration_summary(self):
        calibration = self.load_calibration()
        return get_summary(calibration)

    def save_calibration_summary(self):
        measured_baseline = self.parameters["measured_baseline_m"]
        summary = self.get_calibration_summary()
        summary += f"\nMeasured Baseline: {measured_baseline:.3f} meters"
        with open(
            os.path.join(
                self.scene_folder_structure.folder_path,
                self.scene_name,
                "extrinsics_summary.txt",
            ),
            "w",
        ) as f:
            f.write(summary)

    def cleanup(self):
        """Explicitly cleanup resources to avoid segmentation faults."""
        if (
            hasattr(self, "stereo_tag_detector")
            and self.stereo_tag_detector is not None
        ):
            self.stereo_tag_detector.cleanup()
            self.stereo_tag_detector = None

        if hasattr(self, "video_reader_1") and self.video_reader_1 is not None:
            self.video_reader_1.release()
            self.video_reader_1 = None

        if hasattr(self, "video_reader_2") and self.video_reader_2 is not None:
            self.video_reader_2.release()
            self.video_reader_2 = None

    def _compute_intrinsics(
        self,
        camera_name: str,
        save_path: str | None = None,
        temporal_calib_step_sec: float = 1,
        target_H: int | None = None,
        target_W: int | None = None,
    ):
        """Run intrinsic calibration for a single camera and save results."""
        calibration_folder_path = os.path.join(
            self.scene_folder_structure.folder_path,
            self.scene_folder_structure.get_calibration_intrinsics_folder_name(),
        )
        calibration_camera_path = os.path.join(calibration_folder_path, camera_name)
        checkerboard_specs_path = os.path.join(
            calibration_camera_path, "checkerboard_specs.yml"
        )
        video_name = get_unique_video_name(calibration_camera_path)
        video_path = os.path.join(calibration_camera_path, video_name)

        if save_path is None:
            save_path = os.path.join(
                calibration_folder_path, camera_name, "intrinsics.json"
            )
        # If save_path is provided, ensure it's a file path, not a directory
        elif os.path.isdir(save_path) or not save_path.endswith(".json"):
            save_path = os.path.join(save_path, "intrinsics.json")

        intrinsic_calibration = IntrinsicCalibration(
            video_path, checkerboard_specs_path, save_path, temporal_calib_step_sec
        )
        camera_matrix, dist_coeffs, reprojection_error = (
            intrinsic_calibration.calibrate(
                save_images=False, target_H=target_H, target_W=target_W
            )
        )
        intrinsic_calibration.cleanup()  # Release video reader resources
        intrinsics_dict = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "reprojection_error": reprojection_error,
        }
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(intrinsics_dict, f, indent=2)

        logger.debug(f"Intrinsics of {camera_name} saved to {save_path}")

        return intrinsics_dict

    def _load_intrinsics(self) -> dict[str, Any]:
        """Load intrinsics for both cameras."""
        intrinsics: dict[str, Any] = {}
        calibration_folder_path = os.path.join(
            self.scene_folder_structure.folder_path,
            self.scene_folder_structure.get_calibration_intrinsics_folder_name(),
        )
        for camera_name in self.cameras:
            intrinsics_json_path = os.path.join(
                calibration_folder_path, camera_name, "intrinsics.json"
            )
            with open(intrinsics_json_path) as f:
                intrinsics[camera_name] = json.load(f)

        # Convert from lists back to numpy arrays
        self.camera_matrix1 = np.array(intrinsics["camera_1"]["camera_matrix"])
        self.dist_coeffs1 = np.array(intrinsics["camera_1"]["dist_coeffs"])
        self.camera_matrix2 = np.array(intrinsics["camera_2"]["camera_matrix"])
        self.dist_coeffs2 = np.array(intrinsics["camera_2"]["dist_coeffs"])

        logger.debug(
            f"Loaded intrinsics for camera_1: reprojection_error={intrinsics['camera_1']['reprojection_error']}"
        )
        logger.debug(
            f"Loaded intrinsics for camera_2: reprojection_error={intrinsics['camera_2']['reprojection_error']}"
        )
