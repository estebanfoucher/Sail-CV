"""CLI script to run crop module tracking pipeline on a video with layout.

Run via Makefile:
    make track VIDEO=fixtures/C1_fixture.mp4 LAYOUT=fixtures/C1_layout.json
"""

import argparse
import json
from pathlib import Path

from dumper import Dumper
from loguru import logger
from pipeline import Pipeline
from streamer import Streamer

from models import Layout, PipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    """Main function to run pipeline with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Track telltales in video using layout tracker and crop module PCA analysis"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        required=True,
        help="Path to layout JSON file",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=PROJECT_ROOT / "parameters" / "default_classifier.yml",
        help="Path to parameters YAML file (default: parameters/default_classifier.yml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output directory (default: output/tracking)",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        metavar="N",
        help="First frame index to process (0-based). Default: 0",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=-1,
        metavar="N",
        help="Last frame index to process (0-based, inclusive). -1 = up to last frame. Default: -1",
    )

    args = parser.parse_args()

    # Validate input paths
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.layout.exists():
        raise FileNotFoundError(f"Layout not found: {args.layout}")
    if not args.parameters.exists():
        raise FileNotFoundError(f"Parameters file not found: {args.parameters}")

    # Set default output folder if not provided
    if args.output is None:
        args.output = PROJECT_ROOT / "output" / "tracking"
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sail-CV Tell-Tales Tracking Pipeline")
    logger.info("=" * 60)
    logger.info(f"Video:      {args.video}")
    logger.info(f"Layout:     {args.layout}")
    logger.info(f"Parameters: {args.parameters}")
    logger.info(f"Output:     {args.output}")
    logger.info("=" * 60)

    # Load configuration
    logger.info(f"Loading configuration from {args.parameters}")
    config = PipelineConfig.from_yaml(args.parameters)
    logger.info("Configuration loaded")

    # Load layout
    logger.info(f"Loading layout from {args.layout}")
    with args.layout.open() as f:
        layout_data = json.load(f)
    layout = Layout.from_json_dict(layout_data)
    logger.info("Layout loaded")

    # Initialize pipeline
    logger.info("Initializing pipeline")
    pipeline = Pipeline(config, layout, project_root=PROJECT_ROOT)
    logger.info("Pipeline initialized")

    # Prepare output paths
    output_json_path = args.output / f"{args.video.stem}_crop_module_tracked.json"
    output_video_path = args.output / f"{args.video.stem}_crop_module_tracked.mp4"
    output_fgmask_path = None
    if config.output.generate_fgmask_video:
        output_fgmask_path = args.output / f"{args.video.stem}_fgmask.mp4"

    # Initialize dumper
    dumper = Dumper(
        output_json_path=output_json_path,
        output_video_path=output_video_path
        if config.output.output_tracking_video
        else None,
        output_fgmask_path=output_fgmask_path,
    )

    logger.info("=" * 60)
    logger.info("Processing frames")
    logger.info("=" * 60)

    # Stream video, process frames, and dump results
    with Streamer(
        args.video,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    ) as streamer:
        # Initialize pipeline for video dimensions
        pipeline.initialize_for_video(streamer.width, streamer.height, streamer.fps)

        # Initialize dumper video writers
        dumper.initialize_video_writers(streamer.fps, (streamer.width, streamer.height))

        # Main processing loop (only segment frames are iterated; output videos/JSON contain only this segment)
        for segment_index, (frame_number, frame) in enumerate(streamer, start=1):
            # Process frame through pipeline
            result = pipeline.process_frame(frame, frame_number)

            # Dump result (one frame per segment frame → output videos are the segment only)
            dumper.dump_frame(result)

            if segment_index % 10 == 0 or segment_index == streamer.segment_length:
                logger.info(
                    f"Processed segment {segment_index}/{streamer.segment_length} "
                    f"(global frame {frame_number + 1}) - "
                    f"{len(result.get('tracks', []))} tracks, "
                    f"{len(result.get('pca_vectors', {}))} PCA vectors"
                )

    # Close dumper
    dumper.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline execution completed successfully")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info(f"  - JSON: {output_json_path}")
    if output_video_path:
        logger.info(f"  - Video: {output_video_path}")
    if output_fgmask_path:
        logger.info(f"  - Foreground mask: {output_fgmask_path}")


if __name__ == "__main__":
    main()
