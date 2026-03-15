"""Streamer class for feeding video frames one by one."""

from pathlib import Path

from loguru import logger
from video import VideoReader


class Streamer:
    """
    Streams video frames one by one.

    Simple iterator that reads video and yields frames with their frame numbers.
    """

    def __init__(
        self,
        video_path: Path | str,
        frame_start: int = 0,
        frame_end: int = -1,
    ):
        """
        Initialize streamer.

        Args:
            video_path: Path to input video file
            frame_start: First frame index to process (0-based). Default 0.
            frame_end: Last frame index to process (0-based, inclusive). -1 means
                process up to the last frame. Default -1.
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.frame_start = frame_start
        self.frame_end = frame_end

        self.reader = None
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None
        self.segment_start = 0
        self.segment_end = -1
        self.segment_length = 0

    def __enter__(self):
        """Context manager entry - open video."""
        logger.info(f"Opening video: {self.video_path}")
        self.reader = VideoReader.open_video_file(str(self.video_path))
        self.width, self.height = self.reader.specs.resolution
        self.fps = self.reader.specs.fps
        self.total_frames = self.reader.specs.frame_count

        end = self.total_frames - 1 if self.frame_end < 0 else self.frame_end
        end = min(end, self.total_frames - 1)
        start = max(0, min(self.frame_start, end))
        self.segment_start = start
        self.segment_end = end
        self.segment_length = end - start + 1

        logger.info(
            f"Video specs: {self.width}x{self.height} @ {self.fps} fps, {self.total_frames} frames"
        )
        if self.segment_length < self.total_frames:
            logger.info(
                f"Processing segment: frames {self.segment_start}-{self.segment_end} ({self.segment_length} frames)"
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

        for frame_number in range(self.segment_start, self.segment_end + 1):
            ret, frame = self.reader.read(frame_number)
            if not ret:
                break
            yield frame_number, frame
