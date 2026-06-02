"""Replay raw detections from JSON as video overlays on DS_6 clips.

**Default:** :class:`FakeDetector` + :class:`LayoutTracker` + :func:`draw_tracks`
— each detection is assigned to a predefined **layout position ID** (e.g. ``TL``,
``MT-C``, ``BR``) via Hungarian matching on ``alpha * norm_dist + beta * (1-conf)``.
No invented incrementing IDs — only the layout's own IDs.

**Optional** ``--raw-detections``: draw all detector boxes (no association).

No GrabCut, no PCA, no classifier — only replay JSON → detect(+)layout-track → render.

Layout anchor dots: for clip ``C{i}``, uses ``assets/tracking/layouts/C{i}_layout.json``
when present, else ``--layout-fallback`` (default ``C1_layout.json``).

Outputs ``C{i}_tracked_replay.mp4`` (default) or ``C{i}_bbox_only.mp4`` (``--raw-detections``).

Example::

    uv run python scripts/replay_ds6_faker_bbox_video.py --only-video C6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_SRC = PROJECT_ROOT / "src" / "tracking"


def _ensure_tracking_path() -> None:
    p = str(TRACKING_SRC.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _draw_layout_dots(bgr: "cv2.Mat", layout, width: int, height: int) -> None:
    for pos in layout.positions:
        px, py = pos.to_pixel(width, height)
        cv2.circle(bgr, (px, py), 6, (0, 255, 0), 2)
        cv2.putText(
            bgr,
            pos.id,
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def _run_one(
    *,
    video: Path,
    raw_json: Path,
    out_mp4: Path,
    layout_path: Path | None,
    raw_detections: bool,
    layout_alpha: float,
    layout_beta: float,
    layout_max_distance: float,
    layout_conf_thresh: float,
) -> None:
    _ensure_tracking_path()
    from detector import FakeDetector
    from layout_tracker import LayoutTracker
    from loguru import logger
    from tracker_utils.render_tracks import draw_tracks
    from video import FFmpegVideoWriter, VideoReader

    from models import Image, Layout

    if not video.exists():
        raise FileNotFoundError(video)
    if not raw_json.exists():
        raise FileNotFoundError(raw_json)

    layout = None
    if layout_path is not None:
        if not layout_path.exists():
            raise FileNotFoundError(layout_path)
        with layout_path.open() as f:
            layout = Layout.from_json_dict(json.load(f))

    if not raw_detections and layout is None:
        raise ValueError(
            "LayoutTracker requires a layout file. Provide --layouts-dir/--layout-fallback "
            "or pass --raw-detections."
        )

    detector = FakeDetector(precomputed_results_json_path=raw_json)

    reader = VideoReader.open_video_file(str(video), start_frame=0)
    w, h = reader.specs.resolution
    fps = reader.specs.fps
    nframes = reader.specs.frame_count

    tracker = None
    class_info = None
    if not raw_detections:
        tracker = LayoutTracker(
            layout=layout,
            width=w,
            height=h,
            alpha=layout_alpha,
            beta=layout_beta,
            max_distance=layout_max_distance,
            confidence_thresh=layout_conf_thresh,
        )
        class_info = {
            0: {"name": "telltale", "color": (0, 255, 0)},
        }

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = FFmpegVideoWriter(str(out_mp4), fps, (w, h))

    mode = "raw boxes" if raw_detections else "LayoutTracker"
    logger.info("{} replay: {} -> {}", mode, video.name, out_mp4)

    for frame_idx in range(nframes):
        ret, frame = reader.read(frame_number=frame_idx)
        if not ret:
            break
        image = Image(image=frame, rgb_bgr="BGR")
        dets = detector.detect(image)
        if raw_detections:
            rendered = detector.render_result(image, dets, color=(0, 165, 255), thickness=2)
            out_bgr = rendered.to_bgr()
        else:
            assert tracker is not None and class_info is not None
            tracks = tracker.update(dets)
            out_bgr = image.to_bgr().copy()
            draw_tracks(
                out_bgr,
                tracks,
                class_info,
                show_confidence=True,
                show_class_name=False,
            )
        if layout is not None:
            _draw_layout_dots(out_bgr, layout, w, h)
        writer.write(out_bgr)
        if (frame_idx + 1) % 100 == 0 or frame_idx + 1 == nframes:
            logger.info("  frame {}/{}", frame_idx + 1, nframes)

    writer.release()
    reader.release()
    logger.info("Done: {}", out_mp4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=PROJECT_ROOT / "20260416_213654" / "ds6_compare_out",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "20260416_213654" / "bbox_replay",
        help="Output root: <output>/<model>/C<i>_tracked_replay.mp4 (or _bbox_only)",
    )
    parser.add_argument(
        "--raw-detections",
        action="store_true",
        help="Draw every detection (no tracker). Default: LayoutTracker + layout IDs.",
    )
    parser.add_argument(
        "--layout-alpha",
        type=float,
        default=0.7,
        help="LayoutTracker: weight for normalized distance (default 0.7)",
    )
    parser.add_argument(
        "--layout-beta",
        type=float,
        default=0.3,
        help="LayoutTracker: weight for (1 - confidence) (default 0.3)",
    )
    parser.add_argument(
        "--layout-max-distance",
        type=float,
        default=0.2,
        help="LayoutTracker: max normalized distance for a valid match (default 0.2)",
    )
    parser.add_argument(
        "--layout-conf-thresh",
        type=float,
        default=0.0,
        help="LayoutTracker: minimum confidence for a detection to be matched",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tracking" / "DS_6",
    )
    parser.add_argument(
        "--layouts-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tracking" / "layouts",
        help="Directory containing C*_layout.json",
    )
    parser.add_argument(
        "--layout-fallback",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tracking" / "layouts" / "C1_layout.json",
        help="Used when layouts-dir/<tag>_layout.json is missing",
    )
    parser.add_argument(
        "--no-layout",
        action="store_true",
        help="Do not draw layout anchor dots (boxes only)",
    )
    parser.add_argument("--only-model", type=str, default=None)
    parser.add_argument("--only-video", type=str, default=None)
    args = parser.parse_args()

    compare_dir = args.compare_dir.resolve()
    layouts_dir = args.layouts_dir.resolve()
    layout_fallback = args.layout_fallback.resolve()

    def resolve_layout_for_tag(tag: str) -> Path | None:
        if args.no_layout:
            return None
        specific = layouts_dir / f"{tag}_layout.json"
        if specific.is_file():
            return specific
        if layout_fallback.is_file():
            return layout_fallback
        return None

    for model_dir in sorted(compare_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if args.only_model and model_dir.name != args.only_model:
            continue
        if not list(model_dir.glob("C*_raw_detection.json")):
            continue

        out_dir = args.output.resolve() / model_dir.name
        for i in range(1, 9):
            tag = f"C{i}"
            if args.only_video and tag != args.only_video:
                continue
            vid = args.videos_dir.resolve() / f"{tag}.mp4"
            js = model_dir / f"{tag}_raw_detection.json"
            if not js.is_file():
                continue
            suffix = (
                f"{tag}_bbox_only.mp4"
                if args.raw_detections
                else f"{tag}_tracked_replay.mp4"
            )
            out_mp4 = out_dir / suffix
            clip_layout = resolve_layout_for_tag(tag)
            _run_one(
                video=vid,
                raw_json=js,
                out_mp4=out_mp4,
                layout_path=clip_layout,
                raw_detections=args.raw_detections,
                layout_alpha=args.layout_alpha,
                layout_beta=args.layout_beta,
                layout_max_distance=args.layout_max_distance,
                layout_conf_thresh=args.layout_conf_thresh,
            )

    print("Outputs under:", args.output.resolve())


if __name__ == "__main__":
    main()
