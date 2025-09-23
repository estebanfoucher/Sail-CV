import json
import os
from typing import Any

import cv2
import numpy as np
import yaml
from loguru import logger

from .extrinsics_calibration import (
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
        self.extrinsics_calibration_pattern_specs = (
            self.load_extrinsics_calibration_pattern_specs()
        )
        self.stereo_tag_detector = StereoTagDetector(
            config_path=os.path.join(
                self.scene_folder_structure.folder_path,
                self.scene_name,
                "extrinsics_calibration_pattern_specs.yml",
            )
        )
        self.image_numbers_list = self.get_image_numbers_list()

        self.check_video_fps()
        self.check_resolution()

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

    def get_image_numbers_list(self, temporal_calib_step_sec: float = 1):
        # Convert fps to int for slice step
        step = int(self.video_1.fps * temporal_calib_step_sec)
        camera_1_image_numbers_list = list(
            range(
                self.parameters["camera_1"]["sync_calib_start_frame"],
                self.parameters["camera_1"]["sync_calib_end_frame"],
                step,
            )
        )
        # for camera_2, substract the sync_frame_offset
        camera_2_image_numbers_list = [
            frame - self.sync_frame_offset for frame in camera_1_image_numbers_list
        ]
        return camera_1_image_numbers_list, camera_2_image_numbers_list

    def check_video_fps(self):
        assert self.video_1.fps == self.video_2.fps, "Video fps are not the same"

    def check_resolution(self):
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

        assert (
            self.video_1.resolution == self.video_2.resolution
            or self.video_1.resolution
            == (self.video_2.resolution[1], self.video_2.resolution[0])
        ), "Video resolutions are not compatible "

    def load_extrinsics_calibration_pattern_specs(self):
        return load_parameters(
            os.path.join(
                self.scene_folder_structure.folder_path,
                self.scene_name,
                "extrinsics_calibration_pattern_specs.yml",
            )
        )

    def compute_extrinsics_calibration(self, recompute_intrinsics: bool = False):
        if recompute_intrinsics:
            self._compute_intrinsics(camera_name="camera_1")
            self._compute_intrinsics(camera_name="camera_2")
            # Load the computed intrinsics into object attributes
            self._load_intrinsics()
        else:
            try:
                self._load_intrinsics()
            except:
                logger.info("Intrinsics not found, computing them...")
                self._compute_intrinsics(camera_name="camera_1")
                self._compute_intrinsics(camera_name="camera_2")
                logger.debug("Intrinsics computed, loading them...")
                self._load_intrinsics()
        camera_1_image_numbers_list, camera_2_image_numbers_list = (
            self.get_image_numbers_list()
        )
        successful_pairs = []
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

            camera_1_image = camera_1_frames[0]
            camera_2_image = camera_2_frames[0]
            p3d, p2d1, p2d2 = self.stereo_tag_detector.get_correspondences(
                camera_1_image, camera_2_image
            )
            if p3d is None or p2d1 is None or p2d2 is None:
                continue

            successful_pairs.append((camera_1_image_number, camera_2_image_number))

            object_points_list.append(p3d)
            image_points1_list.append(p2d1)
            image_points2_list.append(p2d2)
            if image_size is None:
                image_size = (camera_1_image.shape[1], camera_1_image.shape[0])

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
        # save to a folder called successful_pairs to 360p
        # successful_pairs/pair_i
        # successful_pairs/pairs.json

        for i, (camera_1_image_number, camera_2_image_number) in enumerate(
            successful_pairs
        ):
            camera_1_frames = self.video_reader_1.get_frames([camera_1_image_number])
            camera_2_frames = self.video_reader_2.get_frames([camera_2_image_number])
            # mkdir successful_pairs if not exists
            path = os.path.join(
                self.scene_folder_structure.folder_path,
                self.scene_name,
                "successful_pairs",
            )
            os.makedirs(path, exist_ok=True)
            if camera_1_frames and camera_2_frames:
                camera_1_image = camera_1_frames[0]
                camera_2_image = camera_2_frames[0]

                # Resize images to 360p (640x360)
                camera_1_image_360p = cv2.resize(camera_1_image, (640, 360))
                camera_2_image_360p = cv2.resize(camera_2_image, (640, 360))

                cv2.imwrite(
                    os.path.join(path, f"pair_{i}_camera1.png"), camera_1_image_360p
                )
                cv2.imwrite(
                    os.path.join(path, f"pair_{i}_camera2.png"), camera_2_image_360p
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
            intrinsic_calibration.calibrate(save_images=False)
        )
        intrinsic_calibration.cleanup()  # Release video reader resources
        intrinsics_dict = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "reprojection_error": reprojection_error,
        }
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
