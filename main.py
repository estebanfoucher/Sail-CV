from pathlib import Path
from track_video import track_video
from models import ModelSpecs, TrackerConfig, Detector, Tracker
project_root = Path(__file__).parent

# process all videos in the assets folder (mp4 and MOV)
for video_path in Path("assets").glob("*.mp4") | Path("assets").glob("*.MOV"):
    track_video(video_path)
    output_folder = project_root / "assets" / "processed"
    output_video_path = output_folder / f"{video_path.stem}.mp4"
    output_json_path = output_folder / f"{video_path.stem}.json"
    # Check if model exists
    rt_detr_model_path = project_root / "checkpoints" / "rt-detr.pt"

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
    track_video(detector, tracker, video_path, output_folder, output_video_path, output_json_path, start_frame=0, end_frame=None)
