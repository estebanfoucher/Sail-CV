from pathlib import Path
import sys


# main class not to test but used for tests of architectures
def detector(model_path=None, architecture=None):
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from models import ModelSpecs, Image
    from detector import Detector
    from video import FFmpegVideoWriter, VideoReader

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent

    video_path = str(project_root / "assets" / "2Ce-CKKCtV4.mp4")
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
    rt_detr_model_path = Path(__file__).parent.parent / "checkpoints" / "rt-detr.pt"
    detector(model_path=rt_detr_model_path, architecture="rt-detr")


def test_tell_tale_detector_yolo():
    yolo_model_path = Path(__file__).parent.parent / "checkpoints" / "yolo-s.pt"
    detector(model_path=yolo_model_path, architecture="yolo")
