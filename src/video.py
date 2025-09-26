import os
import subprocess

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

# Import moved to avoid circular dependency - will be imported when needed


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
        if ret and frame is not None:
            # Apply EXIF transpose and ensure landscape orientation
            corrected_frame = self._ensure_landscape_orientation(frame)
            return ret, corrected_frame
        return ret, None

    def _ensure_landscape_orientation(self, frame: np.ndarray) -> np.ndarray:
        """Ensure frame is in landscape orientation (width >= height)"""
        # Apply EXIF transpose first
        # Import here to avoid circular dependency
        from mv_utils.image_utils import apply_exif_transpose_to_frames

        corrected_frame = apply_exif_transpose_to_frames([frame])[0]

        # Get current dimensions
        height, width = corrected_frame.shape[:2]

        # If already landscape (width >= height), return as is
        if width >= height:
            return corrected_frame

        # If portrait (height > width), rotate 90 degrees clockwise
        rotated_frame = cv2.rotate(corrected_frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated_frame

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


class StereoVideoReader:
    """Stereo video reader for handling two synchronized video streams"""

    def __init__(
        self,
        video_1_path: str,
        video_2_path: str,
        start_video_1_frame: int = 0,
        start_video_2_frame: int = 0,
    ):
        self.video_1_path = video_1_path
        self.video_2_path = video_2_path
        self.video_1 = VideoReader.open_video_file(
            video_1_path, start_frame=start_video_1_frame
        )
        self.video_2 = VideoReader.open_video_file(
            video_2_path, start_frame=start_video_2_frame
        )

    def read(self) -> tuple[bool, bool, np.ndarray | None, np.ndarray | None]:
        """Read frames from both video streams"""
        ret_1, frame_1 = self.video_1.read()
        ret_2, frame_2 = self.video_2.read()
        return ret_1, ret_2, frame_1, frame_2

    def read_frames(
        self, frame_1_number: int, frame_2_number: int
    ) -> tuple[bool, bool, np.ndarray | None, np.ndarray | None]:
        """Read specific frames from both video streams"""
        ret_1, frame_1 = self.video_1.read(frame_1_number)
        ret_2, frame_2 = self.video_2.read(frame_2_number)
        return ret_1, ret_2, frame_1, frame_2

    def release(self):
        """Release both video capture resources"""
        self.video_1.release()
        self.video_2.release()

    def render_side_by_side(
        self,
        frame_1: np.ndarray,
        frame_2: np.ndarray,
        add_labels: bool = True,
        separator_width: int = 5,
        label_font_scale: float = 1.0,
        label_thickness: int = 2,
    ) -> np.ndarray:
        """
        Render two frames side by side with optional labels and separator.

        Args:
            frame_1: First frame (left side)
            frame_2: Second frame (right side)
            add_labels: Whether to add camera labels on top of frames
            separator_width: Width of the separator line between frames
            label_font_scale: Scale of the label text
            label_thickness: Thickness of the label text

        Returns:
            Combined frame with both videos side by side
        """
        if frame_1 is None or frame_2 is None:
            raise ValueError("Both frames must be valid numpy arrays")

        # Get original dimensions
        h1, w1 = frame_1.shape[:2]
        h2, w2 = frame_2.shape[:2]

        # Resize frames to have the same height (use the smaller height)
        target_height = min(h1, h2)

        if h1 != target_height:
            new_w1 = int(w1 * target_height / h1)
            frame_1 = cv2.resize(frame_1, (new_w1, target_height))
            w1 = new_w1
        if h2 != target_height:
            new_w2 = int(w2 * target_height / h2)
            frame_2 = cv2.resize(frame_2, (new_w2, target_height))
            w2 = new_w2

        # Create combined frame using np.hstack for reliable concatenation
        if separator_width > 0:
            # Create separator
            separator = np.full(
                (target_height, separator_width, 3), 128, dtype=np.uint8
            )
            combined_frame = np.hstack([frame_1, separator, frame_2])
        else:
            combined_frame = np.hstack([frame_1, frame_2])

        # Ensure dimensions are even for H.264 encoding
        h, w = combined_frame.shape[:2]
        if w % 2 != 0:
            # Add one pixel column to make width even
            padding = np.zeros((h, 1, 3), dtype=np.uint8)
            combined_frame = np.hstack([combined_frame, padding])
        if h % 2 != 0:
            # Add one pixel row to make height even
            padding = np.zeros((1, combined_frame.shape[1], 3), dtype=np.uint8)
            combined_frame = np.vstack([combined_frame, padding])

        # Add labels if requested
        if add_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_color = (255, 255, 255)  # White color
            label_bg_color = (0, 0, 0)  # Black background

            # Calculate label positions
            label_height = 30
            label_padding = 5

            # Camera 1 label (left side)
            label_1 = "Camera 1"
            (text_w1, text_h1), _ = cv2.getTextSize(
                label_1, font, label_font_scale, label_thickness
            )
            label_x1 = label_padding
            label_y1 = label_height

            # Draw background rectangle for label 1
            cv2.rectangle(
                combined_frame,
                (label_x1 - label_padding, label_y1 - text_h1 - label_padding),
                (label_x1 + text_w1 + label_padding, label_y1 + label_padding),
                label_bg_color,
                -1,
            )

            # Draw text for label 1
            cv2.putText(
                combined_frame,
                label_1,
                (label_x1, label_y1),
                font,
                label_font_scale,
                label_color,
                label_thickness,
            )

            # Camera 2 label (right side)
            label_2 = "Camera 2"
            (text_w2, text_h2), _ = cv2.getTextSize(
                label_2, font, label_font_scale, label_thickness
            )
            label_x2 = w1 + separator_width + label_padding
            label_y2 = label_height

            # Draw background rectangle for label 2
            cv2.rectangle(
                combined_frame,
                (label_x2 - label_padding, label_y2 - text_h2 - label_padding),
                (label_x2 + text_w2 + label_padding, label_y2 + label_padding),
                label_bg_color,
                -1,
            )

            # Draw text for label 2
            cv2.putText(
                combined_frame,
                label_2,
                (label_x2, label_y2),
                font,
                label_font_scale,
                label_color,
                label_thickness,
            )

        return combined_frame


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
