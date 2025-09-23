import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .utils import load_parameters


def find_corners_in_images(
    image_numbers_list: list[int],
    video_reader,
    pattern_size: tuple[int, int],
    square_size: float,
) -> tuple[list, list, list]:
    """Find checkerboard corners in defined images in the video.
    Args:
        image_numbers_list: List of image numbers to process.
        video_reader: VideoReader object.
        pattern_size: Tuple of (width, height) of the checkerboard pattern.
        square_size: Size of the square in the checkerboard pattern.
    Returns:
        object_points: List of 3D points in real world space.
        image_points: List of 2D points in image plane.
        successful_images: List of image numbers that were successfully processed.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = []  # 3D points in real world space
    image_points = []  # 2D points in image plane
    successful_images = []

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    frames = video_reader.get_frames(image_numbers_list)
    for frame_number, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            image_points.append(corners2)
            successful_images.append(frame_number)
            logger.debug(
                f"Found corners in {frame_number} - {frame_number + 1}/{len(image_numbers_list)} images processed"
            )
        else:
            logger.debug(
                f"No corners found in {frame_number} - {frame_number + 1}/{len(image_numbers_list)} images processed"
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

    def calibrate(self, save_images: bool = False):
        object_points, image_points, successful_images = find_corners_in_images(
            self.image_numbers_list,
            self.video_reader,
            (
                self.checkerboard_specs["inner_corners_x"],
                self.checkerboard_specs["inner_corners_y"],
            ),
            self.checkerboard_specs["square_size_mm"],
        )
        # IMPORTANT: use the processed frame size (after EXIF transpose/orientation fix)
        # Frames returned by VideoReader are orientation-corrected; use their size
        assert len(self.image_numbers_list) > 0, (
            "image_numbers_list is empty; cannot determine image size"
        )
        sample_frames = self.video_reader.get_frames([self.image_numbers_list[0]])
        assert sample_frames and sample_frames[0] is not None, (
            "Failed to read a sample frame for determining image size"
        )
        sample_frame = sample_frames[0]
        image_size = (sample_frame.shape[1], sample_frame.shape[0])
        logger.debug(
            f"Using processed frame image_size for intrinsics: {image_size} (width, height)"
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
