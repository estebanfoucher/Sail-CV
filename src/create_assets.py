import os

from loguru import logger

from mv_utils import Scene
from video import FFmpegVideoWriter, StereoVideoReader

save_asset_dict = {
    "scene_3": {
        "camera_1_start_frame_number": 1900,
        "camera_1_end_frame_number": 1950,
    },
    "scene_7": {
        "camera_1_start_frame_number": 3100,
        "camera_1_end_frame_number": 3200,
    },
    "scene_8": {
        "camera_1_start_frame_number": 4900,
        "camera_1_end_frame_number": 4970,
    },
}
stereo_data_folder_path = "../data/"
assets_folder_path = "../assets/"


def save_asset(save_asset_dict):
    for scene_name, scene_data in save_asset_dict.items():
        logger.info(f"Saving asset for scene {scene_name}")
        # save two video separately + save combined video
        scene = Scene(scene_name, stereo_data_folder_path)
        video_paths = scene.get_video_paths()
        sync_frame_offset = scene.sync_frame_offset

        stereo_video_reader = StereoVideoReader(
            video_paths["camera_1"],
            video_paths["camera_2"],
            start_video_1_frame=scene_data["camera_1_start_frame_number"],
            start_video_2_frame=scene_data["camera_1_start_frame_number"]
            - sync_frame_offset,
        )

        resolution_1 = stereo_video_reader.video_1.specs.resolution
        if resolution_1[0] < resolution_1[1]:
            resolution_1 = (resolution_1[1], resolution_1[0])

        resolution_2 = stereo_video_reader.video_2.specs.resolution
        if resolution_2[0] < resolution_2[1]:
            resolution_2 = (resolution_2[1], resolution_2[0])

        os.makedirs(
            os.path.join(assets_folder_path, scene_name, "camera_1"), exist_ok=True
        )
        os.makedirs(
            os.path.join(assets_folder_path, scene_name, "camera_2"), exist_ok=True
        )
        os.makedirs(
            os.path.join(assets_folder_path, scene_name, "combined"), exist_ok=True
        )

        video_1_writer = FFmpegVideoWriter(
            os.path.join(assets_folder_path, scene_name, "camera_1", "camera_1.mp4"),
            stereo_video_reader.video_1.specs.fps,
            resolution_1,
        )
        video_2_writer = FFmpegVideoWriter(
            os.path.join(assets_folder_path, scene_name, "camera_2", "camera_2.mp4"),
            stereo_video_reader.video_2.specs.fps,
            resolution_2,
        )

        # Read first frame to get the combined frame size
        ret, ret_2, frame_1, frame_2 = stereo_video_reader.read()
        if not ret or not ret_2:
            print("Failed to read first frame")
            return

        # Create a test combined frame to get the correct dimensions
        test_combined_frame = stereo_video_reader.render_side_by_side(frame_1, frame_2)
        combined_height, combined_width = test_combined_frame.shape[:2]
        combined_frame_size = (combined_width, combined_height)

        stereo_writer = FFmpegVideoWriter(
            os.path.join(assets_folder_path, scene_name, "combined", "combined.mp4"),
            stereo_video_reader.video_1.specs.fps,
            combined_frame_size,
        )
        stereo_writer.write(test_combined_frame)  # Write the first frame

        for frame_number in range(
            scene_data["camera_1_start_frame_number"],
            scene_data["camera_1_end_frame_number"],
        ):
            logger.info(f"Saving asset for scene {scene_name} frame {frame_number}")
            ret_1, ret_2, frame_1, frame_2 = stereo_video_reader.read()
            if not ret_1 or not ret_2:
                break

            video_1_writer.write(frame_1)
            video_2_writer.write(frame_2)
            stereo_writer.write(
                stereo_video_reader.render_side_by_side(frame_1, frame_2)
            )

        video_1_writer.release()
        video_2_writer.release()
        stereo_writer.release()
        stereo_video_reader.release()

        logger.info(f"Asset for scene {scene_name} saved")


if __name__ == "__main__":
    save_asset(save_asset_dict)
