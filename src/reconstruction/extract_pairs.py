import json
import os

import cv2

from video import StereoVideoReader

STEREO_DATA_FOLDER_PATH = "../assets/"
OUTPUT_FOLDER = "../output/extracted_pairs"


def extract_pairs(scene_name, frame_to_save_number_list, output_folder):
    video_paths = {
        "camera_1": os.path.join(
            STEREO_DATA_FOLDER_PATH, scene_name, "camera_1", "camera_1.mp4"
        ),
        "camera_2": os.path.join(
            STEREO_DATA_FOLDER_PATH, scene_name, "camera_2", "camera_2.mp4"
        ),
    }

    sync_frame_offset = 0

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

        cv2.imwrite(output_path_1, frame_1)
        cv2.imwrite(output_path_2, frame_2)

    # Release the video reader once at the end
    stereo_video_reader.release()

    with open(
        os.path.join(STEREO_DATA_FOLDER_PATH, scene_name, "calibration.json")
    ) as f:
        calibration = json.load(f)

    with open(
        os.path.join(output_folder, f"{scene_name}", "calibration.json"), "w"
    ) as f:
        json.dump(calibration, f, indent=2)


if __name__ == "__main__":
    extract_pairs("scene_3", range(0, 50, 10), OUTPUT_FOLDER)
    extract_pairs("scene_7", range(0, 100, 10), OUTPUT_FOLDER)
    extract_pairs("scene_8", range(0, 70, 10), OUTPUT_FOLDER)
