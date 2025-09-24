import subprocess
from pathlib import Path

import cv2
import sys


def test_is_ffmpeg_available():
    """Check if ffmpeg is available in the system"""
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)


def test_video_reader_and_writer():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent

    video_path = str(
        project_root / "assets" / "scene_3" / "camera_1" / "camera_1.mp4"
    )
    output_folder = project_root / "output_tests" / "video_reader_and_writer"
    output_name = "output_test_video_reader_and_writer.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 10

    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()

    # check output video has same size and fps as input video fps upt to 0.1
    reader_output = VideoReader.open_video_file(str(output_path))
    assert abs(reader_output.specs.fps - reader.specs.fps) < 0.1, (
        f"Output video fps {reader_output.specs.fps} is not the same as input video fps {reader.specs.fps}"
    )
    assert reader_output.specs.resolution == reader.specs.resolution, (
        f"Output video frame size {reader_output.specs.resolution} is not the same as input video frame size {reader.specs.resolution}"
    )
    assert reader_output.specs.frame_count == frame_count, (
        f"Output video frame count {reader_output.specs.frame_count} is not the same as input video frame count {frame_count}"
    )
    reader_output.release()

    reader.release()

    print("Test video reader and writer passed")


def test_stereo_video_reader():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from video import FFmpegVideoWriter, StereoVideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent

    video_1_path = str(project_root / "assets" / "scene_3" / "camera_1" / "camera_1.mp4")
    video_2_path = str(project_root / "assets" / "scene_3" / "camera_2" / "camera_2.mp4")

    sync_frame_offset = 0
    start_frame = 0 # assets are already synced

    stereo_video_reader = StereoVideoReader(
        video_1_path,
        video_2_path,
        start_video_1_frame=start_frame,
        start_video_2_frame=start_frame - sync_frame_offset,
    )
    frame_count = 30
    output_folder = project_root / "output_tests" / "stereo_video_reader"
    output_name = "output_test_stereo_video_reader.mp4"
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Read first frame to get the combined frame size
    ret, ret_2, frame_1, frame_2 = stereo_video_reader.read()
    if not ret or not ret_2:
        print("Failed to read first frame")
        return

    # Create a test combined frame to get the correct dimensions
    test_combined_frame = stereo_video_reader.render_side_by_side(frame_1, frame_2)
    combined_height, combined_width = test_combined_frame.shape[:2]
    combined_frame_size = (combined_width, combined_height)

    writer = FFmpegVideoWriter(
        str(output_path), stereo_video_reader.video_1.specs.fps, combined_frame_size
    )
    writer.write(test_combined_frame)  # Write the first frame

    for i in range(start_frame + 1, start_frame + frame_count):
        ret, ret_2, frame_1, frame_2 = stereo_video_reader.read()
        if not ret or not ret_2:
            break
        frame = stereo_video_reader.render_side_by_side(frame_1, frame_2)
        writer.write(frame)
        print(f"Stereo rendered frame {i}/{start_frame + frame_count}")
    writer.release()
    stereo_video_reader.release()
    print("Test stereo video reader passed")


def test_stereo_image_saver():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from video import StereoVideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent

    video_1_path = str(project_root / "assets" / "scene_8" / "camera_1" / "camera_1.mp4")
    video_2_path = str(project_root / "assets" / "scene_8" / "camera_2" / "camera_2.mp4")
    sync_frame_offset = 0
    frame_to_save_number_list = [0, 10, 20, 30, 40, 50, 60]
    # save frame 1 and frame 2 to the output folder
    output_folder = project_root / "output_tests" / "stereo_image_saver"

    output_folder.mkdir(parents=True, exist_ok=True)
    for frame_to_save_number in frame_to_save_number_list:
        stereo_video_reader = StereoVideoReader(
            video_1_path,
            video_2_path,
            start_video_1_frame=frame_to_save_number,
            start_video_2_frame=frame_to_save_number - sync_frame_offset,
        )
        _ret_1, _ret_2, frame_1, frame_2 = stereo_video_reader.read()
        output_name_1 = f"frame_1_{frame_to_save_number}.png"
        output_name_2 = f"frame_2_{frame_to_save_number}.png"
        frame_folder = output_folder / f"{frame_to_save_number}"
        output_path_1 = frame_folder / output_name_1
        output_path_2 = frame_folder / output_name_2

        frame_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path_1), frame_1)
        cv2.imwrite(str(output_path_2), frame_2)
        stereo_video_reader.release()
