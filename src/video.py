import os
import subprocess

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator


class VideoSpecs(BaseModel):
    """Video specifications and metadata using Pydantic validation"""

    path: str = Field(..., description="Path to the video file")
    width: int = Field(..., ge=1, description="Video width in pixels")
    height: int = Field(..., ge=1, description="Video height in pixels")
    fps: float = Field(..., gt=0, le=120, description="Frames per second")
    frame_count: int = Field(..., ge=0, description="Total number of frames")
    duration_seconds: float | None = Field(
        None, ge=0, description="Video duration in seconds"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Video file does not exist: {v}")
        return v

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution as (width, height) tuple"""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Get video aspect ratio (width/height)"""
        return self.width / self.height

    @property
    def duration_seconds_calculated(self) -> float:
        """Calculate duration from frame count and FPS"""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class Video:
    """Pure data container for video specifications - no loading capability"""

    def __init__(self, specs: VideoSpecs):
        self.specs = specs

    @property
    def fps(self) -> float:
        """Get video FPS"""
        return self.specs.fps

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution (width, height)"""
        return self.specs.resolution

    @property
    def frame_count(self) -> int:
        """Get total number of frames"""
        return self.specs.frame_count

    @property
    def width(self) -> int:
        """Get video width"""
        return self.specs.width

    @property
    def height(self) -> int:
        """Get video height"""
        return self.specs.height

    @property
    def duration_seconds(self) -> float | None:
        """Get video duration in seconds"""
        return self.specs.duration_seconds


class VideoReader:
    """Video reader for sequential frame reading with OpenCV"""

    def __init__(self, video_path: str, start_frame: int = 0):
        self.video_path = video_path
        self.start_frame = start_frame
        self.cap: cv2.VideoCapture | None = None
        self._video: Video | None = None

    @classmethod
    def open_video_file(cls, video_path: str, start_frame: int = 0) -> "VideoReader":
        """Factory method to create VideoReader and associated Video instance"""
        reader = cls(video_path, start_frame)
        # Load video specs and create Video instance
        specs = reader._load_video_specs()
        reader._video = Video(specs)
        reader._open_capture()
        return reader

    def _load_video_specs(self) -> VideoSpecs:
        """Load video specifications from file"""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        # Get basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        # Calculate duration
        duration_seconds = frame_count / fps if fps > 0 else None

        return VideoSpecs(
            path=self.video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )

    def _open_capture(self):
        """Open the video capture"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        # Set starting frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    @property
    def video(self) -> Video:
        """Get the associated Video instance"""
        if self._video is None:
            # Load video specs and create Video instance
            specs = self._load_video_specs()
            self._video = Video(specs)
        return self._video

    @property
    def specs(self) -> VideoSpecs:
        """Get video specifications"""
        return self.video.specs

    def read(self, frame_number: int | None = None) -> tuple[bool, np.ndarray | None]:
        """Read next frame from video or frame at the given frame number"""
        if self.cap is None:
            self._open_capture()

        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        return ret, frame

    def get_frames(self, list_of_frames: list[int]) -> list[np.ndarray]:
        """Get specific frames from the video"""
        frames = []

        for frame_number in list_of_frames:
            if frame_number < 0 or frame_number >= self.video.frame_count:
                continue

            ret, frame = self.read(frame_number)
            if not ret:
                break

            frames.append(frame)

        return frames

    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FFmpegVideoWriter:
    """Video writer using FFmpeg subprocess for high-quality output"""

    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]):
        self.output_path = output_path
        self.width, self.height = frame_size
        self.fps = fps

        # Launch ffmpeg process
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",  # overwrite
                "-loglevel",
                "error",  # only show error messages
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "-",  # read from stdin
                "-an",  # no audio
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                self.output_path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray):
        """Write a single frame"""
        self.process.stdin.write(frame.tobytes())

    def release(self):
        """Close the writer and wait for completion"""
        self.process.stdin.close()
        self.process.wait()


class FrameExtractor:
    """Utility class for extracting specific frames from video to files"""

    def __init__(self, video_path: str, output_dir: str, list_of_frames: list[int]):
        self.video_path = video_path
        self.output_dir = output_dir
        self.list_of_frames = list_of_frames

    def extract_frames(self):
        """Extract frames from the video at the given list of frames"""
        video = Video(self.video_path)
        specs = video.specs

        # Filter valid frame numbers
        frame_numbers = np.array(self.list_of_frames)
        frame_numbers = frame_numbers[
            frame_numbers < specs.frame_count
        ]  # remove frames that are out of range
        frame_numbers = frame_numbers[frame_numbers >= 0]  # remove negative frames

        cap = cv2.VideoCapture(self.video_path)

        for frame_number in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{self.output_dir}/frame_{frame_number}.jpg", frame)

        cap.release()


def get_unique_video_name(folder_path: str) -> str:
    """
    Get the video name in the folder (mp4, MP4)

    Args:
        folder_path: Path to folder containing video files

    Returns:
        Video filename or None if no video found
    """
    video_name = None
    for file in os.listdir(folder_path):
        if file.endswith((".mp4", ".MP4")):
            video_name = file
            break
    return video_name
