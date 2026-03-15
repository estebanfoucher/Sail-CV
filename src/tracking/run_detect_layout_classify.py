"""Lightweight run: detector + layout tracker + classifier only (no masks, no PCA).

Processes a short segment (e.g. 30 frames from second 3 to 4). Use this when you
only need detection, static layout assignment, and classification.

Example (from repo root):
  export PYTHONPATH="${PWD}/src/tracking"
  uv run python src/tracking/run_detect_layout_classify.py \\
    --video assets/tracking/2Ce-CKKCtV4.mp4 \\
    --layout output/tracking_layouts/2Ce-CKKCtV4_layout.json \\
    --start-sec 3 \\
    --num-frames 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from detector import Detector
from layout_tracker import LayoutTracker
from loguru import logger
from pipeline import make_json_serializable
from video import FFmpegVideoWriter, VideoReader

if TYPE_CHECKING:
    import numpy as np

from classifyer import Classifier
from models import Image, Layout, ModelSpecs, PipelineConfig
from tracker_utils.render_tracks import draw_tracks

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detector + layout tracker + classifier only (no masks), on a short segment"
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video")
    parser.add_argument("--layout", type=Path, required=True, help="Layout JSON")
    parser.add_argument(
        "--parameters",
        type=Path,
        default=PROJECT_ROOT / "parameters" / "default.yaml",
        help="Parameters YAML (default: parameters/default.yaml)",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=3.0,
        help="Start time in seconds (default: 3)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of frames to process (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: output/tracking)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not write output video",
    )
    return parser.parse_args()


def _extract_padded_crop(
    frame_bgr: np.ndarray,
    bbox,
    padding_factor: float,
) -> np.ndarray | None:
    """Extract a padded crop from the frame (same logic as Pipeline)."""
    h, w = frame_bgr.shape[:2]
    x1 = float(bbox.xyxy.x1)
    y1 = float(bbox.xyxy.y1)
    x2 = float(bbox.xyxy.x2)
    y2 = float(bbox.xyxy.y2)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding_factor
    pad_y = bh * padding_factor
    x1_p = max(0, int(x1 - pad_x))
    y1_p = max(0, int(y1 - pad_y))
    x2_p = min(w, int(x2 + pad_x))
    y2_p = min(h, int(y2 + pad_y))
    if x2_p <= x1_p or y2_p <= y1_p:
        return None
    crop = frame_bgr[y1_p:y2_p, x1_p:x2_p]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return crop


def main() -> None:
    args = _parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.layout.exists():
        raise FileNotFoundError(f"Layout not found: {args.layout}")
    if not args.parameters.exists():
        raise FileNotFoundError(f"Parameters not found: {args.parameters}")

    out_dir = args.output or (PROJECT_ROOT / "output" / "tracking")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.video.stem}_detect_layout_classify.json"
    out_video = (
        None
        if args.no_video
        else (out_dir / f"{args.video.stem}_detect_layout_classify.mp4")
    )

    logger.info("Loading config and layout")
    config = PipelineConfig.from_yaml(args.parameters)
    with args.layout.open() as f:
        layout = Layout.from_json_dict(json.load(f))

    # Detector
    model_path = config.detector.model_path
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    detector = Detector(
        ModelSpecs(model_path=model_path, architecture=config.detector.architecture)
    )
    logger.info("Detector initialized")

    # Classifier (optional)
    classifier = None
    padding_factor = 0.25
    if config.classifier is not None:
        cp = config.classifier.model_path
        if not cp.is_absolute():
            cp = PROJECT_ROOT / cp
        cfg = config.classifier.model_copy()
        cfg.model_path = cp
        classifier = Classifier(cfg)
        padding_factor = config.classifier.padding_factor
        logger.info("Classifier initialized")

    # Open video and get dimensions
    reader = VideoReader.open_video_file(
        str(args.video),
        start_frame=0,
    )
    fps = reader.specs.fps
    w, h = reader.specs.resolution
    total = reader.specs.frame_count
    reader.release()

    start_frame = round(args.start_sec * fps)
    start_frame = max(0, min(start_frame, total - 1) if total else start_frame)
    num_frames = min(
        args.num_frames, (total - start_frame) if total else args.num_frames
    )

    # Layout tracker (needs width, height)
    lt_cfg = config.layout_tracker
    layout_tracker = LayoutTracker(
        layout,
        w,
        h,
        alpha=lt_cfg.alpha,
        beta=lt_cfg.beta,
        max_distance=lt_cfg.max_distance,
        confidence_thresh=lt_cfg.confidence_thresh,
    )
    logger.info("Layout tracker initialized")

    # Open again with start frame
    reader = VideoReader.open_video_file(str(args.video), start_frame=start_frame)
    video_writer = None
    if out_video:
        video_writer = FFmpegVideoWriter(str(out_video), fps, (w, h))

    results: list[dict] = []
    class_info = {
        0: {"name": "class_0", "color": (0, 255, 0)},
        1: {"name": "class_1", "color": (0, 0, 255)},
        2: {"name": "class_2", "color": (255, 0, 0)},
    }

    logger.info(
        f"Processing {num_frames} frames starting at frame {start_frame} (t={args.start_sec}s)"
    )
    for i in range(num_frames):
        ret, frame = reader.read()
        if not ret or frame is None:
            break
        frame_number = start_frame + i
        image = Image(image=frame, rgb_bgr="BGR")

        detections = detector.detect(image)
        tracks = layout_tracker.update(detections)

        classifications: dict[int | str, int] = {}
        if tracks and classifier is not None:
            for track in tracks:
                crop = _extract_padded_crop(frame, track.detection.bbox, padding_factor)
                if crop is None:
                    continue
                class_id, _conf = classifier.classify_crop(crop)
                if class_id is not None:
                    classifications[track.track_id] = class_id

        result = {
            "frame_number": frame_number,
            "tracks": tracks,
            "classifications": classifications,
        }
        results.append(result)

        if video_writer:
            # Only pass classifications if every track was classified (draw_tracks requires it)
            cls_for_draw = (
                classifications
                if (classifier and tracks and len(classifications) == len(tracks))
                else None
            )
            rendered = draw_tracks(
                frame,
                tracks,
                class_info,
                show_confidence=True,
                show_class_name=False,
                classifications=cls_for_draw,
            )
            video_writer.write(rendered)

        if (i + 1) % 10 == 0 or (i + 1) == num_frames:
            logger.info(
                f"Frame {frame_number + 1} | {len(tracks)} tracks, "
                f"{len(classifications)} classified"
            )

    reader.release()
    if video_writer:
        video_writer.release()

    # Write JSON (same serialization as dumper, without rendered_frame/movement_mask)
    serializable = [make_json_serializable(r) for r in results]
    with out_json.open("w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Done")
    logger.info(f"  JSON: {out_json}")
    if out_video:
        logger.info(f"  Video: {out_video}")
    logger.info(f"  Frames: {len(results)}")


if __name__ == "__main__":
    main()
