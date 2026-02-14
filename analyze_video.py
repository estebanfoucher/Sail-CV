"""CLI script to run crop module tracking pipeline on a video with layout."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dumper import Dumper
from models import Layout, PipelineConfig
from pipeline import Pipeline
from streamer import Streamer


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
        default=project_root / "parameters" / "default.yaml",
        help="Path to parameters YAML file (default: parameters/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output directory (default: assets/processed)",
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
        args.output = project_root / "assets" / "processed"
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Crop Module Tracking Pipeline")
    logger.info("=" * 60)
    logger.info(f"Video: {args.video}")
    logger.info(f"Layout: {args.layout}")
    logger.info(f"Parameters: {args.parameters}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    # Load configuration
    logger.info(f"Loading configuration from {args.parameters}")
    config = PipelineConfig.from_yaml(args.parameters)
    logger.info("✓ Configuration loaded")

    # Load layout
    logger.info(f"Loading layout from {args.layout}")
    with args.layout.open() as f:
        layout_data = json.load(f)
    layout = Layout.from_json_dict(layout_data)
    logger.info("✓ Layout loaded")

    # Initialize pipeline
    logger.info("Initializing pipeline")
    pipeline = Pipeline(config, layout, project_root=project_root)
    logger.info("✓ Pipeline initialized")

    # Prepare output paths
    output_json_path = args.output / f"{args.video.stem}_crop_module_tracked.json"
    output_video_path = args.output / f"{args.video.stem}_crop_module_tracked.mp4"
    output_fgmask_path = None
    if config.output.generate_fgmask_video:
        output_fgmask_path = args.output / f"{args.video.stem}_fgmask.mp4"

    # Initialize dumper
    dumper = Dumper(
        output_json_path=output_json_path,
        output_video_path=output_video_path if config.output.render_masks or config.output.render_arrows else None,
        output_fgmask_path=output_fgmask_path,
    )

    logger.info("=" * 60)
    logger.info("Processing frames")
    logger.info("=" * 60)

    # Stream video, process frames, and dump results
    with Streamer(args.video) as streamer:
        # Initialize pipeline for video dimensions
        pipeline.initialize_for_video(streamer.width, streamer.height, streamer.fps)

        # Initialize dumper video writers
        dumper.initialize_video_writers(streamer.fps, (streamer.width, streamer.height))

        # Main processing loop
        for frame_number, frame in streamer:
            # Process frame through pipeline
            result = pipeline.process_frame(frame, frame_number)

            # Dump result
            dumper.dump_frame(result)

            if (frame_number + 1) % 10 == 0 or (frame_number + 1) == streamer.total_frames:
                logger.info(
                    f"Processed frame {frame_number + 1}/{streamer.total_frames} - "
                    f"{len(result.get('tracks', []))} tracks, "
                    f"{len(result.get('pca_vectors', {}))} PCA vectors"
                )

    # Close dumper
    dumper.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Pipeline execution completed successfully")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info(f"  - JSON: {output_json_path}")
    if output_video_path:
        logger.info(f"  - Video: {output_video_path}")
    if output_fgmask_path:
        logger.info(f"  - Foreground mask: {output_fgmask_path}")


if __name__ == "__main__":
    main()
