import subprocess
from pathlib import Path

import pytest


def _is_ffmpeg_available():
    """Check if ffmpeg is available in the system"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not _is_ffmpeg_available(), reason="ffmpeg not available")
def test_tell_tale_detector(model_path=None, architecture="yolov8n"):
    from unitaries.tell_tale_detector import TellTaleDetector
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent.parent

    video_path = str(
        project_root / "data" / "calibration_intrinsics_1" / "camera_1" / "GH010815.MP4"
    )
    output_folder = project_root / "output" / "tests" / "tell_tale_detector"
    output_name = f"output_test_tell_tale_detector_{architecture}.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 10
    if model_path is None:
        # Skip test if no model path provided
        print("Skipping test_tell_tale_detector - no model path provided")
        return
    detector = TellTaleDetector(model_path=model_path, architecture=architecture)

    def process_frame(frame):
        result = detector.predict(frame, conf=0.1)
        frame = detector.render_result(frame, result)
        return frame

    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        frame = process_frame(frame)
        writer.write(frame)

    writer.release()
    reader.release()
    print("Test tell_tale_detector passed")


@pytest.mark.skipif(not _is_ffmpeg_available(), reason="ffmpeg not available")
def test_sam():
    from unitaries.sam import SAM
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent.parent

    video_path = str(
        project_root / "data" / "calibration_intrinsics_1" / "camera_1" / "GH010815.MP4"
    )
    output_folder = project_root / "output" / "tests" / "sam"
    output_name = "output_test_sam.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 10
    SAM_MODEL_PATH = project_root / "checkpoints" / "FastSAM-x.pt"
    sam = SAM(SAM_MODEL_PATH)
    sam = SAM()

    def process_frame(frame):
        point = (1400, 540)
        result = sam.predict(frame, point, 1)
        frame = sam.render_result(frame, result["mask"], [point])
        return frame

    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        frame = process_frame(frame)
        writer.write(frame)

    writer.release()
    reader.release()
    print("Test sam passed")
