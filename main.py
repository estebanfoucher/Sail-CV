from pathlib import Path
import sys

project_root = Path(__file__).parent
# Ensure local 'src' modules are importable when running this script directly
sys.path.insert(0, str(project_root / "src"))

from models import ModelSpecs, TrackerConfig
from detector import Detector
from tracker import Tracker
from track_video import track_video


def main() -> None:
    assets_dir = project_root / "assets"
    output_folder = assets_dir / "processed"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Configure model (default to RT-DETR)
    model_path = project_root / "checkpoints" / "rt-detr.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize detector and tracker once
    specs = ModelSpecs(model_path=model_path, architecture="rt-detr")
    detector = Detector(specs)

    tracker_config = TrackerConfig(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
    )
    tracker = Tracker(tracker_config)

    # Process all videos in the assets folder (mp4 and MOV)
    videos = list(assets_dir.glob("*.mp4")) + list(assets_dir.glob("*.MOV"))
    for video_path in videos:
        output_video_path = output_folder / f"{video_path.stem}.mp4"
        output_json_path = output_folder / f"{video_path.stem}.json"

        track_video(
            detector,
            tracker,
            video_path,
            output_folder,
            output_video_path,
            output_json_path,
            start_frame=0,
            end_frame=None,
        )


if __name__ == "__main__":
    main()
