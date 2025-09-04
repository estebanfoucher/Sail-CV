from .video_utils import Video, get_unique_video_name
from .stereo_data_folder_structure import load_scene_folder_structure
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Tuple, List
from loguru import logger
import os
from .utils import load_parameters
import json
with open("config.json", "r") as f:
    config = json.load(f)
TEMPORAL_CALIB_STEP_SEC = config["temporal_calib_step_sec"]

def find_corners_in_images(image_numbers_list: List[int], video: Video, pattern_size: Tuple[int, int], 
                          square_size: float) -> Tuple[List, List, List]:
    """Find checkerboard corners in defined images in the video.
    Args:
        image_numbers_list: List of image numbers to process.
        video: Video object.
        pattern_size: Tuple of (width, height) of the checkerboard pattern.
        square_size: Size of the square in the checkerboard pattern.
    Returns:
        object_points: List of 3D points in real world space.
        image_points: List of 2D points in image plane.
        successful_images: List of image numbers that were successfully processed.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    successful_images = []
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    frames = video.get_frames(image_numbers_list)
    for frame_number, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            image_points.append(corners2)
            successful_images.append(frame_number)
            logger.debug(f"Found corners in {frame_number} - {frame_number + 1}/{len(image_numbers_list)} images processed")
        else:
            logger.debug(f"No corners found in {frame_number} - {frame_number + 1}/{len(image_numbers_list)} images processed")
    
    logger.debug(f"Successfully processed {len(successful_images)}/{len(image_numbers_list)} images")
    return object_points, image_points, successful_images

def calibrate_camera(object_points: List, image_points: List, 
                    image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run camera calibration using OpenCV's calibrateCamera."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], 
                                        camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    logger.debug(f"Calibration error: {mean_error/len(object_points):.4f} pixels")
    return camera_matrix, dist_coeffs, mean_error/len(object_points)

class IntrinsicCalibration:
    def __init__(self, video_path: str, checkerboard_specs_path: str, save_path: str):
        logger.debug(f"Initializing IntrinsicCalibration with video_path: {video_path}, checkerboard_specs_path: {checkerboard_specs_path}, save_path: {save_path}")
        self.video_path = video_path
        self.checkerboard_specs_path = checkerboard_specs_path
        self.save_path = save_path
        self.video = Video(video_path)
        self.checkerboard_specs = load_parameters(checkerboard_specs_path)
        logger.debug(f"Checkerboard specs: {self.checkerboard_specs}")
        self.image_numbers_list = self.get_image_numbers_list()
        logger.debug(f"Image numbers list: {self.image_numbers_list}")
    
    def calibrate(self, save_images: bool = False):
        object_points, image_points, successful_images = find_corners_in_images(self.image_numbers_list, self.video, (self.checkerboard_specs["inner_corners_x"], self.checkerboard_specs["inner_corners_y"]), self.checkerboard_specs["square_size_mm"])
        camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(object_points, image_points, self.video.get_resolution())
        self.save_calibration(camera_matrix, dist_coeffs, reprojection_error, successful_images, save_images)
        logger.debug(f"Calibration results saved to {self.save_path}")
        return camera_matrix, dist_coeffs, reprojection_error

    
    def get_image_numbers_list(self) -> List[int]:
        """Get the list of image numbers from the video."""
        fps = self.video.get_fps()
        # Convert FPS to integer step value for sampling frames
        step = int(fps*TEMPORAL_CALIB_STEP_SEC) if fps > 0 else 1
        return list(range(self.video.get_frame_count()))[::step]
    
    def save_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, reprojection_error: float, successful_images: List[int], save_images: bool = False):
        """Save the calibration results to a file."""
        with open(self.save_path, 'w') as f:
            json.dump({"camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist(), "reprojection_error": reprojection_error}, f, indent=2)
        
        if save_images and successful_images:
            # Convert successful_images indices to actual frame numbers
            actual_frame_numbers = [self.image_numbers_list[i] for i in successful_images]
            self._save_successful_images(actual_frame_numbers)
    
    def _save_successful_images(self, successful_images: List[int]):
        """Save successful calibration images at 360p resolution."""
        # Create successful_images directory
        save_dir = Path(self.save_path).parent / "successful_images"
        save_dir.mkdir(exist_ok=True)
        
        # 360p resolution (640x360)
        target_width, target_height = 640, 360
        
        logger.debug(f"Saving {len(successful_images)} successful images to {save_dir}")
        
        for i, frame_number in enumerate(successful_images):
            # Get the frame from video
            frames = self.video.get_frames([frame_number])
            if frames:
                frame = frames[0]
                
                # Resize to 360p
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                # Save the image
                image_path = save_dir / f"successful_frame_{frame_number:04d}.jpg"
                cv2.imwrite(str(image_path), resized_frame)
            
        print(f"Successfully saved {len(successful_images)} images to {save_dir}")
            
def test_calibrate_intrinsics_scene():
    with open("config.json", "r") as f:
        scene_name = json.load(f)["scene_name"]
    scene_folder_structure = load_scene_folder_structure(scene_name)
    calibration_folder_path = os.path.join(scene_folder_structure.folder_path, scene_folder_structure.get_calibration_intrinsics_folder_name())
    
    for camera_name in ["camera_1", "camera_2"]:
        calibration_camera_path = os.path.join(calibration_folder_path, camera_name)
        checkerboard_specs_path = os.path.join(calibration_camera_path, "checkerboard_specs.yml")
        video_name = get_unique_video_name(calibration_camera_path)
        video_path = os.path.join(calibration_camera_path, video_name)
        save_path = os.path.join(calibration_folder_path, camera_name, f"intrinsics.json")
        intrinsic_calibration = IntrinsicCalibration(video_path, checkerboard_specs_path, save_path)
        intrinsic_calibration.calibrate(save_images=True)
        print(f"Intrinsic calibration saved to {save_path}")
    
def main():
    test_calibrate_intrinsics_scene()
    
if __name__ == "__main__":
    main()
