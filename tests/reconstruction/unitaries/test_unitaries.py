from pathlib import Path


def sam(model_path=None):
    from video import FFmpegVideoWriter, VideoReader

    from unitaries.sam import SAM

    project_root = Path(__file__).resolve().parents[3]

    video_path = str(
        project_root
        / "assets"
        / "reconstruction"
        / "scene_8"
        / "camera_1"
        / "camera_1.mp4"
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
        # Use center point which is more likely to be in a detected object
        # Point format: (x, y) where x is width, y is height
        point = (frame.shape[1] // 2, frame.shape[0] // 2)
        result = sam.predict(frame, point, 1)
        assert result["mask"] is not None, (
            f"No mask found at point {point} in frame of shape {frame.shape}. "
            f"Try a different point or check if the frame contains detectable objects."
        )
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
        Path(__file__).resolve().parents[3] / "checkpoints" / "FastSAM-x.pt"
    )
    sam(model_path=sam_model_path)
