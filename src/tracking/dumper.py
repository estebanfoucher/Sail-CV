"""Dumper class for writing pipeline results to files."""

import json
from pathlib import Path

import cv2
from loguru import logger

from pipeline import make_json_serializable
from video import FFmpegVideoWriter


class Dumper:
    """
    Dumps pipeline results to JSON and optional video files.

    Handles incremental writing of results as frames are processed.
    """

    def __init__(
        self,
        output_json_path: Path | str,
        output_video_path: Path | str | None = None,
        output_fgmask_path: Path | str | None = None,
    ):
        """
        Initialize dumper.

        Args:
            output_json_path: Path to write JSON results
            output_video_path: Optional path to write rendered video
            output_fgmask_path: Optional path to write foreground mask video
        """
        self.output_json_path = Path(output_json_path)
        self.output_video_path = Path(output_video_path) if output_video_path else None
        self.output_fgmask_path = (
            Path(output_fgmask_path) if output_fgmask_path else None
        )

        # Create output directory
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize JSON file (write empty array to start)
        with self.output_json_path.open("w") as f:
            json.dump([], f)

        # Video writers (initialized when video specs are known)
        self.video_writer = None
        self.fgmask_writer = None
        self.frame_count = 0

    def initialize_video_writers(self, fps: float, resolution: tuple[int, int]):
        """
        Initialize video writers with video specifications.

        Args:
            fps: Frames per second
            resolution: Tuple of (width, height)
        """
        if self.output_video_path:
            self.video_writer = FFmpegVideoWriter(
                str(self.output_video_path), fps, resolution
            )
            logger.info(f"Output video: {self.output_video_path}")

        if self.output_fgmask_path:
            self.fgmask_writer = FFmpegVideoWriter(
                str(self.output_fgmask_path), fps, resolution
            )
            logger.info(f"Output fgmask video: {self.output_fgmask_path}")

    def dump_frame(self, result: dict):
        """
        Dump a single frame result.

        Args:
            result: Dictionary with frame_number, tracks, pca_vectors, and optionally
                   rendered_frame and movement_mask
        """
        # Write result to JSON incrementally
        self._append_result_to_json(result)

        # Write rendered frame if available
        if self.video_writer and "rendered_frame" in result:
            self.video_writer.write(result["rendered_frame"])

        # Write foreground mask if available
        if self.fgmask_writer and "movement_mask" in result:
            movement_mask = result["movement_mask"]
            fgmask_bgr = cv2.cvtColor(movement_mask * 255, cv2.COLOR_GRAY2BGR)
            self.fgmask_writer.write(fgmask_bgr)

        self.frame_count += 1

    def close(self):
        """Close all writers and finalize output."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.fgmask_writer:
            self.fgmask_writer.release()
            self.fgmask_writer = None

        logger.info("=" * 60)
        logger.info("✓ Dumping completed successfully!")
        logger.info("=" * 60)
        logger.info(f"  - Output JSON: {self.output_json_path}")
        if self.output_video_path:
            logger.info(f"  - Output video: {self.output_video_path}")
        if self.output_fgmask_path:
            logger.info(f"  - Output fgmask video: {self.output_fgmask_path}")
        logger.info(f"  - Total frames dumped: {self.frame_count}")

    def _append_result_to_json(self, result: dict):
        """
        Append a single frame result to JSON file incrementally.

        Reads current JSON, appends new result, writes back.
        This is not the most efficient but ensures incremental writing.
        """
        # Read current results
        with self.output_json_path.open("r") as f:
            results = json.load(f)

        # Serialize and append new result
        serializable_result = make_json_serializable(result)
        # Remove rendered_frame and movement_mask from JSON (they're binary data)
        serializable_result.pop("rendered_frame", None)
        serializable_result.pop("movement_mask", None)
        results.append(serializable_result)

        # Write back
        with self.output_json_path.open("w") as f:
            json.dump(results, f, indent=2)
