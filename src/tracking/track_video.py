import json
import sys
from pathlib import Path

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import Detector
from models import Image
from tracker import Tracker
from tracker_utils.render_tracks import draw_tracks
from video import FFmpegVideoWriter, VideoReader


def make_json_serializable(obj):
    """Convert numpy arrays and Pydantic models to JSON-serializable format."""
    import numpy as np

    # Handle numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)

    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # Handle collections
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]

    return obj


def track_video(
    detector: Detector,
    tracker: Tracker,
    video_path: Path,
    output_folder: Path,
    output_video_path: Path,
    output_json_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
):
    """Track video with detector and tracker, render tracks, and save serialized results"""

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Convert Path to string for OpenCV
    video_path_str = str(video_path) if isinstance(video_path, Path) else video_path

    # Open video reader
    reader = VideoReader.open_video_file(video_path_str, start_frame=start_frame)
    fps = reader.specs.fps
    total_frames = reader.specs.frame_count
    if end_frame is not None:
        total_frames = min(total_frames, end_frame)

    # Process full video
    logger.info(f"Processing full video: {total_frames} frames at {fps} fps")

    # Setup video writer
    writer = FFmpegVideoWriter(
        str(output_video_path), reader.specs.fps, reader.specs.resolution
    )

    # Class info for rendering (based on typical YOLO class structure)
    class_info = {
        0: {"name": "pennon_attached", "color": (0, 255, 0)},  # green
        1: {"name": "pennon_detached", "color": (0, 0, 255)},  # red (BGR)
        2: {"name": "pennon_leech", "color": (255, 0, 0)},  # blue
    }

    # Process frames and collect tracks
    tracks_timeline = []
    frame_number = start_frame

    while True:
        if end_frame is not None and frame_number >= end_frame:
            break
        ret, frame = reader.read()
        if not ret:
            break

        # Convert frame to Image model
        image = Image(image=frame, rgb_bgr="BGR")

        # Detect objects
        detections = detector.detect(image)

        # Track objects
        tracks = tracker.update(detections)

        # Render tracks on frame
        rendered_frame = draw_tracks(
            image.to_bgr(),
            tracks,
            class_info,
            show_confidence=True,
            show_class_name=False,
        )

        # Write rendered frame
        writer.write(rendered_frame)

        # Store tracks for serialization
        tracks_timeline.append({"frame_number": frame_number, "tracks": tracks})

        frame_number += 1
        if frame_number % 10 == 0 or frame_number == total_frames:
            logger.info(
                f"Processed frame {frame_number}/{total_frames} - {len(tracks)} tracks"
            )

    # Release resources
    writer.release()
    reader.release()

    # Serialize and save tracks
    serializable_tracks = make_json_serializable(tracks_timeline)
    with open(output_json_path, "w") as f:
        json.dump(serializable_tracks, f, indent=2)

    logger.info("✓ Tracking completed successfully!")
    logger.info(f"  - Output video: {output_video_path}")
    logger.info(f"  - Serialized tracks: {output_json_path}")
    logger.info(f"  - Total frames processed: {frame_number}")
    logger.info(
        f"  - Total track entries: {sum(len(frame['tracks']) for frame in tracks_timeline)}"
    )
