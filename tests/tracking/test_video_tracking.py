import subprocess
from pathlib import Path

import sys


def test_is_ffmpeg_available():
    """Check if ffmpeg is available in the system"""
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)


def test_video_reader_and_writer():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent

    video_path = str(project_root / "assets" / "IMG_9496_0.0_3.0.MOV")
    output_folder = project_root / "output_tests" / "video_reader_and_writer"
    output_name = "output_test_video_reader_and_writer.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 30

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
