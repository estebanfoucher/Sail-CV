"""Streamer class for feeding video frames one by one."""

from pathlib import Path

from loguru import logger
from video import VideoReader


class Streamer:
    """
    Streams video frames one by one.

    Simple iterator that reads video and yields frames with their frame numbers.
    """

    def __init__(self, video_path: Path | str):
        """
        Initialize streamer.

        Args:
            video_path: Path to input video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.reader = None
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None

    def __enter__(self):
        """Context manager entry - open video."""
        logger.info(f"Opening video: {self.video_path}")
        self.reader = VideoReader.open_video_file(str(self.video_path))
        self.width, self.height = self.reader.specs.resolution
        self.fps = self.reader.specs.fps
        self.total_frames = self.reader.specs.frame_count

        logger.info(
            f"Video specs: {self.width}x{self.height} @ {self.fps} fps, {self.total_frames} frames"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close video."""
        if self.reader:
            self.reader.release()
            self.reader = None

    def __iter__(self):
        """Make streamer iterable."""
        if self.reader is None:
            raise RuntimeError(
                "Streamer must be used as context manager: 'with Streamer(...) as streamer:'"
            )

        frame_number = 0
        while frame_number < self.total_frames:
            ret, frame = self.reader.read()
            if not ret:
                break
            yield frame_number, frame
            frame_number += 1
