from pathlib import Path
import sys


def tell_tale_detector(model_path=None, architecture=None):
    sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
    from unitaries.tell_tale_detector import TellTaleDetector
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent.parent

    video_path = str(
        project_root / "assets" / "scene_3" / "camera_1" / "camera_1.mp4"
    )
    output_folder = project_root / "output_tests" / "tell_tale_detector"
    output_name = f"output_test_tell_tale_detector_{architecture}.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 10

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


def test_tell_tale_detector_rt_detr():
    rt_detr_model_path = (
        Path(__file__).parent.parent.parent / "checkpoints" / "rt-detr.pt"
    )
    tell_tale_detector(model_path=rt_detr_model_path, architecture="rt-detr")


def test_tell_tale_detector_yolo():
    yolo_model_path = Path(__file__).parent.parent.parent / "checkpoints" / "yolos.pt"
    tell_tale_detector(model_path=yolo_model_path, architecture="yolo")


def sam(model_path=None):
    sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
    from unitaries.sam import SAM
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent.parent

    video_path = str(
        project_root / "assets" / "scene_8" / "camera_1" / "camera_1.mp4"
    )
    output_folder = project_root / "output_tests" / "sam"
    output_name = "output_test_sam.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 10
    sam = SAM(model_path)

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


def test_sam():
    sam_model_path = (
        Path(__file__).parent.parent.parent / "checkpoints" / "FastSAM-x.pt"
    )
    sam(model_path=sam_model_path)
