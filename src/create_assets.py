import os
from pathlib import Path

import numpy as np
from loguru import logger

from mv_utils import Scene
from video import FFmpegVideoWriter, StereoVideoReader

save_asset_dict = {
    # "scene_3": {
    #     "camera_1_start_frame_number": 1900,
    #     "camera_1_end_frame_number": 1950,
    # },
    # "scene_7": {
    #     "camera_1_start_frame_number": 3100,
    #     "camera_1_end_frame_number": 3200,
    # },
    # "scene_8": {
    #     "camera_1_start_frame_number": 4900,
    #     "camera_1_end_frame_number": 4970,
    # },
    # "scene_10": {
    #     "camera_1_start_frame_number": 2640,
    #     "camera_1_end_frame_number": 2644,
    # },
    "scene_11": {
        "camera_1_start_frame_number": 2300,
        "camera_1_end_frame_number": 2304,
    },
    # "scene_12": {
    #     "camera_1_start_frame_number": 3560,
    #     "camera_1_end_frame_number": 3564,
    # },
    # "scene_13": {
    #     "camera_1_start_frame_number": 226,
    #     "camera_1_end_frame_number": 230,
    # },
    # "scene_14": {
    #     "camera_1_start_frame_number": 2150,
    #     "camera_1_end_frame_number": 2154,
    # },
    "scene_15": {
        "camera_1_start_frame_number": 2100,
        "camera_1_end_frame_number": 2104,
    },
}

# use path.parent to get the project root
project_root = Path(__file__).parent.parent
stereo_data_folder_path = project_root / "data"
assets_folder_path = project_root / "assets"


def _calculate_time_based_sync(parameters, fps_1, fps_2):
    """
    Calculate time-based sync offset, handling different FPS by converting to time.

    Returns:
        Tuple of (time_offset_seconds, fps_ratio)
    """
    camera_1_sync_event_list_F = parameters["camera_1"]["sync_event_time_F"]
    camera_2_sync_event_list_F = parameters["camera_2"]["sync_event_time_F"]

    # Check if FPS are effectively the same
    if abs(fps_1 - fps_2) < 0.01:
        # Same FPS: use frame-based calculation
        diff = np.array(camera_1_sync_event_list_F) - np.array(
            camera_2_sync_event_list_F
        )
        if not np.allclose(diff, diff[0], atol=1):
            logger.warning(f"Frame offset diff is not constant: {diff}")
        time_offset_seconds = 0.0
        fps_ratio = 1.0
    else:
        # Different FPS: convert to time, then calculate time offset
        times_1 = np.array(camera_1_sync_event_list_F) / fps_1
        times_2 = np.array(camera_2_sync_event_list_F) / fps_2

        time_offsets = times_1 - times_2
        if not np.allclose(time_offsets, time_offsets[0], atol=0.01):
            logger.warning(f"Time offset diff is not constant: {time_offsets} seconds")
        time_offset_seconds = np.mean(time_offsets)
        fps_ratio = fps_2 / fps_1

    return time_offset_seconds, fps_ratio


def save_asset(save_asset_dict):
    for scene_name, scene_data in save_asset_dict.items():
        logger.info(f"Saving asset for scene {scene_name}")
        # save two video separately + save combined video
        scene = Scene(scene_name, stereo_data_folder_path)
        video_paths = scene.get_video_paths()
        parameters = scene.parameters

        # Create video readers to get FPS
        from video import VideoReader

        temp_video_1 = VideoReader.open_video_file(video_paths["camera_1"])
        temp_video_2 = VideoReader.open_video_file(video_paths["camera_2"])
        fps_1 = temp_video_1.specs.fps
        fps_2 = temp_video_2.specs.fps
        temp_video_1.release()
        temp_video_2.release()

        # Calculate time-based sync
        time_offset_seconds, fps_ratio = _calculate_time_based_sync(
            parameters, fps_1, fps_2
        )

        # Log FPS information
        logger.info(f"Video FPS: camera_1={fps_1:.2f} fps, camera_2={fps_2:.2f} fps")
        if abs(fps_1 - fps_2) > 0.01:
            logger.info(
                f"FPS differ - using time-based sync: offset={time_offset_seconds:.4f}s, "
                f"FPS ratio={fps_ratio:.4f}"
            )
            output_fps = min(fps_1, fps_2)
            logger.info(
                f"Using lower FPS ({output_fps:.2f}) for output videos to maintain sync"
            )
        else:
            output_fps = fps_1
            logger.info("FPS match - using frame-based sync")

        # Calculate start frame for camera_2 using time-based sync
        start_frame_1 = scene_data["camera_1_start_frame_number"]
        time_1 = start_frame_1 / fps_1
        time_2 = time_1 - time_offset_seconds
        start_frame_2 = int(time_2 * fps_2)

        stereo_video_reader = StereoVideoReader(
            video_paths["camera_1"],
            video_paths["camera_2"],
            start_video_1_frame=start_frame_1,
            start_video_2_frame=start_frame_2,
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

        # Use lower FPS for both output videos to maintain sync
        video_1_writer = FFmpegVideoWriter(
            os.path.join(assets_folder_path, scene_name, "camera_1", "camera_1.mp4"),
            output_fps,
            resolution_1,
        )
        video_2_writer = FFmpegVideoWriter(
            os.path.join(assets_folder_path, scene_name, "camera_2", "camera_2.mp4"),
            output_fps,
            resolution_2,
        )

        # Process frames using time-based synchronization
        frame_numbers_1 = list(
            range(
                scene_data["camera_1_start_frame_number"],
                scene_data["camera_1_end_frame_number"],
            )
        )

        # Read first frame to get the combined frame size
        if not frame_numbers_1:
            logger.warning(f"No frames to process for scene {scene_name}")
            return

        first_frame_1 = frame_numbers_1[0]
        time_1_first = first_frame_1 / fps_1
        time_2_first = time_1_first - time_offset_seconds
        first_frame_2 = int(time_2_first * fps_2)

        ret_1, frame_1 = stereo_video_reader.video_1.read(first_frame_1)
        ret_2, frame_2 = stereo_video_reader.video_2.read(first_frame_2)
        if not ret_1 or not ret_2:
            logger.error(
                f"Failed to read first synchronized frames: camera_1={first_frame_1}, "
                f"camera_2={first_frame_2}"
            )
            return

        # Create a test combined frame to get the correct dimensions
        test_combined_frame = stereo_video_reader.render_side_by_side(frame_1, frame_2)
        combined_height, combined_width = test_combined_frame.shape[:2]
        combined_frame_size = (combined_width, combined_height)

        stereo_writer = FFmpegVideoWriter(
            os.path.join(assets_folder_path, scene_name, "combined", "combined.mp4"),
            output_fps,
            combined_frame_size,
        )
        stereo_writer.write(test_combined_frame)  # Write the first frame

        # Write first frame to individual videos
        video_1_writer.write(frame_1)
        video_2_writer.write(frame_2)

        # Process remaining frames using time-based synchronization
        for frame_number_1 in frame_numbers_1[1:]:
            logger.info(f"Saving asset for scene {scene_name} frame {frame_number_1}")

            # Convert camera_1 frame to time
            time_1 = frame_number_1 / fps_1

            # Calculate corresponding camera_2 time
            time_2 = time_1 - time_offset_seconds

            # Convert to camera_2 frame number
            frame_number_2 = int(time_2 * fps_2)

            # Read frames at synchronized positions
            ret_1, frame_1 = stereo_video_reader.video_1.read(frame_number_1)
            ret_2, frame_2 = stereo_video_reader.video_2.read(frame_number_2)

            if not ret_1 or not ret_2:
                logger.warning(
                    f"Failed to read synchronized frames: camera_1={frame_number_1}, "
                    f"camera_2={frame_number_2}"
                )
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
