"""Render a detect_layout_classify JSON as a video with bboxes and class colors.

Reads the JSON produced by run_detect_layout_classify.py, seeks to each frame_number
in the source video, draws bounding boxes and labels with colors by class_id, and
writes an output video.

Example (from repo root):
  export PYTHONPATH="${PWD}/src/tracking"
  uv run python src/tracking/render_results_video.py \\
    --results-json output/tracking/2Ce-CKKCtV4_detect_layout_classify.json \\
    --video assets/tracking/2Ce-CKKCtV4.mp4 \\
    --output output/tracking/2Ce-CKKCtV4_rendered.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from loguru import logger
from video import FFmpegVideoWriter, VideoReader

from models import Track
from tracker_utils.render_tracks import draw_tracks

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default colors (BGR) for class_id 0, 1, 2, ...
CLASS_COLORS = [
    (0, 255, 0),  # green
    (0, 0, 255),  # red
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render detect_layout_classify JSON to video with bboxes and colors"
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to _detect_layout_classify.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Source video (default: assets/tracking/<stem>.mp4 from results filename)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: same dir as JSON, <stem>_rendered.mp4)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_path = args.results_json
    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    # Default video: e.g. 2Ce-CKKCtV4_detect_layout_classify -> assets/tracking/2Ce-CKKCtV4.mp4
    stem = results_path.stem
    if stem.endswith("_detect_layout_classify"):
        stem = stem[: -len("_detect_layout_classify")]
    video_path = args.video or (PROJECT_ROOT / "assets" / "tracking" / f"{stem}.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = args.output or (results_path.parent / f"{stem}_rendered.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open() as f:
        frames_data = json.load(f)
    if not frames_data:
        raise ValueError("Results JSON is empty")

    # Collect all class_ids used so we can build class_info
    all_class_ids: set[int] = set()
    for entry in frames_data:
        for cid in (entry.get("classifications") or {}).values():
            all_class_ids.add(int(cid))
        for t in entry.get("tracks") or []:
            all_class_ids.add(int(t.get("detection", {}).get("class_id", 0)))
    if not all_class_ids:
        all_class_ids = {0}
    class_info = {
        cid: {"name": f"class_{cid}", "color": CLASS_COLORS[cid % len(CLASS_COLORS)]}
        for cid in sorted(all_class_ids)
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    reader = VideoReader.open_video_file(str(video_path))
    writer = FFmpegVideoWriter(str(out_path), fps, (w, h))

    logger.info(f"Rendering {len(frames_data)} frames -> {out_path}")
    for i, entry in enumerate(frames_data):
        frame_number = entry["frame_number"]
        tracks_raw = entry.get("tracks") or []
        classifications = entry.get("classifications") or {}

        ret, frame = reader.read(frame_number)
        if not ret or frame is None:
            logger.warning(f"Frame {frame_number} not read, skipping")
            continue

        tracks = [Track.model_validate(t) for t in tracks_raw]
        # Only pass classifications if every track has one (draw_tracks requirement)
        cls_for_draw = (
            classifications
            if (tracks and len(classifications) == len(tracks))
            else None
        )
        frame = draw_tracks(
            frame,
            tracks,
            class_info,
            show_confidence=True,
            show_class_name=False,
            classifications=cls_for_draw,
        )
        writer.write(frame)

        if (i + 1) % 10 == 0 or (i + 1) == len(frames_data):
            logger.info(f"Rendered {i + 1}/{len(frames_data)}")

    reader.release()
    writer.release()
    logger.info(f"Done: {out_path}")


if __name__ == "__main__":
    main()
