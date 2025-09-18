import cv2
import numpy as np
from loguru import logger
import os
import json
from PIL import Image
from PIL.ImageOps import exif_transpose

from video import StereoVideoReader
from mv_utils import Scene, load_stereo_data_folder_structure
    
scene_name = "scene_8"
frame_to_save_number_list = [4900,4910,4920,4930,4940,4950,4960]
output_folder = "/app/tmp/pairs"

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
    corrected_rgb = np.array(corrected_pil)
    
    # Convert RGB back to BGR for OpenCV
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    
    return corrected_bgr

def extract_pairs(scene_name, frame_to_save_number_list, output_folder):
    
    stereo_data_folder_structure = load_stereo_data_folder_structure()
    scene_names = stereo_data_folder_structure.get_scene_folders()
    
    assert scene_name in scene_names, f"Scene {scene_name} is not in the list of scene names"
    
    scene = Scene(scene_name)
    video_paths = scene.get_video_paths()
    sync_frame_offset = scene.sync_frame_offset
    
    os.makedirs(os.path.join(output_folder, f"{scene_name}"), exist_ok=True)

    
    # Create StereoVideoReader once
    stereo_video_reader = StereoVideoReader(video_paths["camera_1"], video_paths["camera_2"])
    
    for frame_to_save_number in frame_to_save_number_list:        
        # Seek to the specific frame for camera 1
        stereo_video_reader.video_1.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_save_number)
        # Seek to the specific frame for camera 2 (with sync offset)
        stereo_video_reader.video_2.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_save_number - sync_frame_offset)
        
        ret_1, ret_2, frame_1, frame_2 = stereo_video_reader.read()
        assert ret_1 and ret_2, f"Failed to read frames for {frame_to_save_number}"
        
        # Apply EXIF transpose to handle orientation metadata
        frame_1_corrected = apply_exif_transpose_cv2(frame_1)
        frame_2_corrected = apply_exif_transpose_cv2(frame_2)
        
        # Check if both frames were successfully read
        output_path_1 = os.path.join(output_folder, f"{scene_name}", f"{frame_to_save_number}", "frame_1.png")
        output_path_2 = os.path.join(output_folder, f"{scene_name}", f"{frame_to_save_number}", "frame_2.png")
        os.makedirs(os.path.join(output_folder, f"{scene_name}", f"{frame_to_save_number}"), exist_ok=True)
        
        cv2.imwrite(output_path_1, frame_1_corrected)
        cv2.imwrite(output_path_2, frame_2_corrected)


    # Release the video reader once at the end
    stereo_video_reader.release()

    extrinsic_calibration = scene.create_extrinsic_calibration()
    extrinsics_calibration = extrinsic_calibration.load_extrinsics()
    with open(os.path.join(output_folder, f"{scene_name}", "extrinsics_calibration.json"), "w") as f:
        json.dump(extrinsics_calibration, f, indent=2)

if __name__ == "__main__":
    extract_pairs(scene_name, frame_to_save_number_list, output_folder)
