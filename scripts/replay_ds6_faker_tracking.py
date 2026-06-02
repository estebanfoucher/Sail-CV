"""Replay full crop-module tracking on DS_6 videos using FakeDetector + raw_detection JSONs.

For **bounding boxes only** (no PCA / crop / classifier / layout tracker), use
``scripts/replay_ds6_faker_bbox_video.py`` instead.

Reads per-model folders under a compare output (e.g. ``20260416_213654/ds6_compare_out/``),
runs the same pipeline as ``src/tracking/analyze_video.py`` with ``parameters/c1_faker.yaml``,
but points ``detector.model_path`` at each ``C*_raw_detection.json``.

Videos: ``assets/tracking/DS_6/C{1..8}.mp4``
Layout: ``assets/tracking/layouts/C1_layout.json`` (same normalized layout for all clips).

Usage (from repo root, with tracking env)::

    uv run python scripts/replay_ds6_faker_tracking.py

    uv run python scripts/replay_ds6_faker_tracking.py \\
        --compare-dir 20260416_213654/ds6_compare_out \\
        --output 20260416_213654/tracking_replay \\
        --only-model custom_ulysse \\
        --only-video C1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_SRC = PROJECT_ROOT / "src" / "tracking"


def _ensure_tracking_path() -> None:
    p = str(TRACKING_SRC.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_config_with_json(base_yaml: Path, json_path: Path):
    _ensure_tracking_path()
    from models.pipeline_config import PipelineConfig

    with base_yaml.open() as f:
        data = yaml.safe_load(f)
    data["detector"]["model_path"] = json_path.resolve()
    return PipelineConfig(**data)


def _run_one(
    *,
    video: Path,
    layout_path: Path,
    raw_json: Path,
    out_dir: Path,
    base_yaml: Path,
    frame_end: int = -1,
) -> None:
    _ensure_tracking_path()
    from dumper import Dumper
    from loguru import logger
    from pipeline import Pipeline
    from streamer import Streamer

    from models import Layout

    if not video.exists():
        raise FileNotFoundError(video)
    if not raw_json.exists():
        raise FileNotFoundError(raw_json)
    if not layout_path.exists():
        raise FileNotFoundError(layout_path)

    config = _load_config_with_json(base_yaml, raw_json)

    with layout_path.open() as f:
        layout_data = json.load(f)
    layout = Layout.from_json_dict(layout_data)

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video.stem
    output_json_path = out_dir / f"{stem}_crop_module_tracked.json"
    output_video_path = (
        out_dir / f"{stem}_crop_module_tracked.mp4"
        if config.output.output_tracking_video
        else None
    )
    output_fgmask_path = None
    if config.output.generate_fgmask_video:
        output_fgmask_path = out_dir / f"{stem}_fgmask.mp4"

    pipeline = Pipeline(config, layout, project_root=PROJECT_ROOT)
    dumper = Dumper(
        output_json_path=output_json_path,
        output_video_path=output_video_path,
        output_fgmask_path=output_fgmask_path,
    )

    logger.info("Video {} | detections {} | out {}", video, raw_json, out_dir)

    with Streamer(video, frame_start=0, frame_end=frame_end) as streamer:
        pipeline.initialize_for_video(streamer.width, streamer.height, streamer.fps)
        dumper.initialize_video_writers(streamer.fps, (streamer.width, streamer.height))
        for segment_index, (frame_number, frame) in enumerate(streamer, start=1):
            result = pipeline.process_frame(frame, frame_number)
            dumper.dump_frame(result)
            if segment_index % 50 == 0 or segment_index == streamer.segment_length:
                logger.info("  frame {}/{}", segment_index, streamer.segment_length)

    dumper.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=PROJECT_ROOT / "20260416_213654" / "ds6_compare_out",
        help="Folder containing model subdirs with C*_raw_detection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "20260416_213654" / "tracking_replay",
        help="Root folder for replay outputs (per-model subdirs created)",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tracking" / "layouts" / "C1_layout.json",
        help="Layout JSON (normalized positions)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tracking" / "DS_6",
        help="Directory with C1.mp4 … C8.mp4",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=PROJECT_ROOT / "parameters" / "c1_faker.yaml",
        help="Base YAML (FakeDetector-friendly); detector.model_path is overridden",
    )
    parser.add_argument(
        "--only-model",
        type=str,
        default=None,
        help="Process only this subfolder name (e.g. custom_ulysse)",
    )
    parser.add_argument(
        "--only-video",
        type=str,
        default=None,
        help="Process only this stem (e.g. C1)",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=-1,
        metavar="N",
        help="Last frame index inclusive, or -1 for full video (for quick tests)",
    )
    args = parser.parse_args()

    compare_dir = args.compare_dir.resolve()
    if not compare_dir.is_dir():
        raise FileNotFoundError(compare_dir)

    for model_dir in sorted(compare_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if args.only_model and model_dir.name != args.only_model:
            continue
        # skip non-model folders if any
        jsons = list(model_dir.glob("C*_raw_detection.json"))
        if not jsons:
            continue

        out_root = args.output.resolve() / model_dir.name
        for i in range(1, 9):
            tag = f"C{i}"
            if args.only_video and tag != args.only_video:
                continue
            video = args.videos_dir.resolve() / f"{tag}.mp4"
            raw_json = model_dir / f"{tag}_raw_detection.json"
            if not raw_json.is_file():
                continue
            clip_out = out_root / tag
            _run_one(
                video=video,
                layout_path=args.layout.resolve(),
                raw_json=raw_json,
                out_dir=clip_out,
                base_yaml=args.parameters.resolve(),
                frame_end=args.frame_end,
            )

    print("Done. Outputs under:", args.output.resolve())


if __name__ == "__main__":
    main()
