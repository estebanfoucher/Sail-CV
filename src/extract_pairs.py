import json
import os

import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

from mv_utils import Scene
from video import StereoVideoReader

scene_name = "scene_8"
stereo_data_folder_path = "../data/"
frame_to_save_number_list = [4900, 4910, 4920, 4930, 4940, 4950, 4960]
output_folder = "../output/extracted_pairs"


def apply_exif_transpose_cv2(frame):
    """
    Apply EXIF transpose to OpenCV frame to handle orientation metadata.

    Args:
        frame: OpenCV frame (BGR numpy array)

    Returns:
        Corrected frame with proper orientation
    """
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Apply EXIF transpose
    corrected_pil = exif_transpose(pil_image)

    # Convert back to RGB numpy array
    corrected_rgb = np.asarray(corrected_pil)

    # Convert RGB back to BGR for OpenCV
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)

    return corrected_bgr


def extract_pairs(scene_name, frame_to_save_number_list, output_folder):
    scene = Scene(scene_name, stereo_data_folder_path)
    video_paths = scene.get_video_paths()
    sync_frame_offset = scene.sync_frame_offset

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, f"{scene_name}"), exist_ok=True)

    # Create StereoVideoReader once and reuse it for all frames
    stereo_video_reader = StereoVideoReader(
        video_paths["camera_1"], video_paths["camera_2"]
    )

    for frame_to_save_number in frame_to_save_number_list:
        # Use the new read_frames method to seek to specific frames
        ret_1, ret_2, frame_1, frame_2 = stereo_video_reader.read_frames(
            frame_to_save_number, frame_to_save_number - sync_frame_offset
        )

        assert ret_1 and ret_2, f"Failed to read frames for {frame_to_save_number}"

        # Apply EXIF transpose to handle orientation metadata
        frame_1_corrected = apply_exif_transpose_cv2(frame_1)
        frame_2_corrected = apply_exif_transpose_cv2(frame_2)

        # Check if both frames were successfully read
        output_path_1 = os.path.join(
            output_folder, f"{scene_name}", f"{frame_to_save_number}", "frame_1.png"
        )
        output_path_2 = os.path.join(
            output_folder, f"{scene_name}", f"{frame_to_save_number}", "frame_2.png"
        )
        os.makedirs(
            os.path.join(output_folder, f"{scene_name}", f"{frame_to_save_number}"),
            exist_ok=True,
        )

        cv2.imwrite(output_path_1, frame_1_corrected)
        cv2.imwrite(output_path_2, frame_2_corrected)

    # Release the video reader once at the end
    stereo_video_reader.release()

    scene.create_calibration()
    calibration = scene.calibration.load_calibration()

    with open(
        os.path.join(output_folder, f"{scene_name}", "calibration.json"), "w"
    ) as f:
        json.dump(calibration, f, indent=2)


if __name__ == "__main__":
    extract_pairs(scene_name, frame_to_save_number_list, output_folder)
