from pathlib import Path
import json
import numpy as np

from models import ModelSpecs, TrackerConfig
from detector import Detector
from tracker import Tracker
from track_video import track_video

def test_track_video_yolo():
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "assets" / "tracking" / "2Ce-CKKCtV4.mp4"
    output_folder = project_root / "output_tests" / "tracker"
    output_video_path = output_folder / "output_test_tracker_yolo.mp4"
    output_json_path = output_folder / "output_test_tracker_yolo_tracks.json"

    # Check if model exists
    yolo_model_path = project_root / "checkpoints" / "yolo-s.pt"
    if not yolo_model_path.exists():
        print(f"Skipping test: Model file not found: {yolo_model_path}")
        return

    # Initialize detector
    specs = ModelSpecs(model_path=yolo_model_path, architecture="yolo")
    detector = Detector(specs)

    # Initialize tracker
    tracker_config = TrackerConfig(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
    )
    tracker = Tracker(tracker_config)

    # Track video
    track_video(detector, tracker, video_path, output_folder, output_video_path, output_json_path, start_frame=0, end_frame=60)

def test_track_video_rt_detr():
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "assets" / "tracking" / "IMG_9496_0.0_3.0.MOV"
    output_folder = project_root / "output_tests" / "tracker"
    output_video_path = output_folder / "output_test_tracker_rt_detr.mp4"
    output_json_path = output_folder / "output_test_tracker_rt_detr_tracks.json"

    # Check if model exists
    rt_detr_model_path = project_root / "checkpoints" / "rt-detr.pt"
    if not rt_detr_model_path.exists():
        print(f"Skipping test: Model file not found: {rt_detr_model_path}")
        return

    # Initialize detector
    specs = ModelSpecs(model_path=rt_detr_model_path, architecture="rt-detr")
    detector = Detector(specs)

    # Initialize tracker
    tracker_config = TrackerConfig(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
    )
    tracker = Tracker(tracker_config)

    # Track video
    track_video(detector, tracker, video_path, output_folder, output_video_path, output_json_path, start_frame=30, end_frame=40)
