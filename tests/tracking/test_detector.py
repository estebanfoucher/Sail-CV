from pathlib import Path

from model_weights import resolve_model_path


# main class not to test but used for tests of architectures
def detector(model_path=None, architecture=None):
    from detector import Detector
    from video import FFmpegVideoWriter, VideoReader

    from models import Image, ModelSpecs

    project_root = Path(__file__).resolve().parents[2]

    video_path = str(project_root / "assets" / "tracking" / "2Ce-CKKCtV4.mp4")
    output_folder = project_root / "output_tests" / "detector"
    output_name = f"output_test_detector_{architecture}.mp4"
    reader = VideoReader.open_video_file(video_path)
    output_path = output_folder / output_name

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    frame_count = 10

    # Create ModelSpecs and Detector
    specs = ModelSpecs(model_path=model_path, architecture=architecture)
    detector = Detector(specs)

    def process_frame(frame):
        # Convert numpy array to Image
        image = Image(image=frame, rgb_bgr="BGR")
        detections = detector.detect(image)
        rendered_image = detector.render_result(image, detections)
        # Convert back to numpy array for writer
        return rendered_image.to_bgr()

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
    project_root = Path(__file__).resolve().parents[2]
    rt_detr_model_path = resolve_model_path(
        project_root / "checkpoints" / "sailcv-rtdetrl640.pt",
        project_root=project_root,
    )
    detector(model_path=rt_detr_model_path, architecture="rt-detr")


def test_tell_tale_detector_yolo():
    """Requires local yolo-s.pt (not on HF). Skipped if missing."""
    import pytest

    project_root = Path(__file__).resolve().parents[2]
    yolo_model_path = project_root / "checkpoints" / "yolo-s.pt"
    if not yolo_model_path.exists():
        pytest.skip("yolo-s.pt not found (not on HF); place locally to run")
    detector(model_path=yolo_model_path, architecture="yolo")
