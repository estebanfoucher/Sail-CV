"""
Video processing wrapper for MVS web application
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Import existing classes from src
from video import VideoReader
from utils.validation import get_video_info

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Wrapper class for video processing operations"""

    def __init__(self):
        self.video_1_reader: Optional[VideoReader] = None
        self.video_2_reader: Optional[VideoReader] = None
        self.video_1_path: Optional[str] = None
        self.video_2_path: Optional[str] = None
        self.sync_offset: int = 0

    def load_videos(self, video_1_path: str, video_2_path: str) -> Tuple[bool, str]:
        """
        Load both videos for processing

        Args:
            video_1_path: Path to video 1
            video_2_path: Path to video 2

        Returns:
            tuple: (success, message)
        """
        try:
            logger.info(f"Loading videos: {video_1_path}, {video_2_path}")

            # Load video 1
            self.video_1_reader = VideoReader.open_video_file(video_1_path)
            if self.video_1_reader is None:
                return False, f"Could not load video 1: {video_1_path}"

            # Load video 2
            self.video_2_reader = VideoReader.open_video_file(video_2_path)
            if self.video_2_reader is None:
                self.video_1_reader.release()
                return False, f"Could not load video 2: {video_2_path}"

            self.video_1_path = video_1_path
            self.video_2_path = video_2_path

            logger.info("Both videos loaded successfully")
            return True, "Videos loaded successfully"

        except Exception as e:
            logger.error(f"Error loading videos: {str(e)}", exc_info=True)
            return False, f"Error loading videos: {str(e)}"

    def load_single_video(self, video_path: str, video_name: str) -> Tuple[bool, str]:
        """
        Load a single video for processing

        Args:
            video_path: Path to the video file
            video_name: "video_1" or "video_2"

        Returns:
            tuple: (success, message)
        """
        try:
            logger.info(f"Loading single video: {video_path} as {video_name}")

            reader = VideoReader.open_video_file(video_path)
            if reader is None:
                return False, f"Could not load {video_name}: {video_path}"

            if video_name == "video_1":
                # Release existing video 1 if any
                if self.video_1_reader:
                    self.video_1_reader.release()
                self.video_1_reader = reader
                self.video_1_path = video_path
            elif video_name == "video_2":
                # Release existing video 2 if any
                if self.video_2_reader:
                    self.video_2_reader.release()
                self.video_2_reader = reader
                self.video_2_path = video_path
            else:
                reader.release()
                return False, f"Invalid video name: {video_name}"

            logger.info(f"{video_name} loaded successfully")
            return True, f"{video_name} loaded successfully"

        except Exception as e:
            logger.error(f"Error loading {video_name}: {str(e)}", exc_info=True)
            return False, f"Error loading {video_name}: {str(e)}"

    def get_video_info(self) -> Dict[str, Any]:
        """Get information about loaded videos"""
        info = {
            'video_1_loaded': self.video_1_reader is not None,
            'video_2_loaded': self.video_2_reader is not None,
            'sync_offset': self.sync_offset
        }

        if self.video_1_reader:
            specs_1 = self.video_1_reader.specs
            info['video_1'] = {
                'path': self.video_1_path,
                'width': specs_1.width,
                'height': specs_1.height,
                'fps': specs_1.fps,
                'frame_count': specs_1.frame_count,
                'duration': specs_1.duration_seconds,
                'resolution': f"{specs_1.width}x{specs_1.height}"
            }

        if self.video_2_reader:
            specs_2 = self.video_2_reader.specs
            info['video_2'] = {
                'path': self.video_2_path,
                'width': specs_2.width,
                'height': specs_2.height,
                'fps': specs_2.fps,
                'frame_count': specs_2.frame_count,
                'duration': specs_2.duration_seconds,
                'resolution': f"{specs_2.width}x{specs_2.height}"
            }

        return info

    def set_sync_offset(self, offset: int) -> None:
        """Set synchronization offset between videos"""
        self.sync_offset = offset
        logger.info(f"Sync offset set to: {offset}")

    def read_frame(self, video_num: int, frame_number: int) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Read a specific frame from a video

        Args:
            video_num: 1 or 2 for video 1 or 2
            frame_number: Frame number to read

        Returns:
            tuple: (success, frame, message)
        """
        try:
            if video_num == 1:
                if self.video_1_reader is None:
                    return False, None, "Video 1 not loaded"

                # Check frame bounds
                if frame_number < 0 or frame_number >= self.video_1_reader.specs.frame_count:
                    return False, None, f"Frame {frame_number} out of bounds (0-{self.video_1_reader.specs.frame_count-1})"

                ret, frame = self.video_1_reader.read(frame_number)
                if ret and frame is not None:
                    return True, frame, f"Frame {frame_number} loaded successfully"
                else:
                    return False, None, f"Could not read frame {frame_number} from video 1"

            elif video_num == 2:
                if self.video_2_reader is None:
                    return False, None, "Video 2 not loaded"

                # Apply sync offset
                adjusted_frame = frame_number - self.sync_offset

                # Check frame bounds
                if adjusted_frame < 0 or adjusted_frame >= self.video_2_reader.specs.frame_count:
                    return False, None, f"Adjusted frame {adjusted_frame} out of bounds (0-{self.video_2_reader.specs.frame_count-1})"

                ret, frame = self.video_2_reader.read(adjusted_frame)
                if ret and frame is not None:
                    return True, frame, f"Frame {frame_number} (adjusted: {adjusted_frame}) loaded successfully"
                else:
                    return False, None, f"Could not read frame {adjusted_frame} from video 2"

            else:
                return False, None, f"Invalid video number: {video_num}"

        except Exception as e:
            logger.error(f"Error reading frame {frame_number} from video {video_num}: {str(e)}")
            return False, None, f"Error reading frame: {str(e)}"

    def read_synchronized_frames(self, frame_number: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Read synchronized frames from both videos

        Args:
            frame_number: Frame number for video 1

        Returns:
            tuple: (success, frame_1, frame_2, message)
        """
        try:
            # Read frame from video 1
            success_1, frame_1, msg_1 = self.read_frame(1, frame_number)
            if not success_1:
                return False, None, None, f"Video 1: {msg_1}"

            # Read frame from video 2 (with sync offset)
            success_2, frame_2, msg_2 = self.read_frame(2, frame_number)
            if not success_2:
                return False, frame_1, None, f"Video 2: {msg_2}"

            return True, frame_1, frame_2, f"Both frames loaded: {msg_1}, {msg_2}"

        except Exception as e:
            logger.error(f"Error reading synchronized frames: {str(e)}")
            return False, None, None, f"Error reading synchronized frames: {str(e)}"

    def release(self) -> None:
        """Release video readers"""
        if self.video_1_reader:
            self.video_1_reader.release()
            self.video_1_reader = None

        if self.video_2_reader:
            self.video_2_reader.release()
            self.video_2_reader = None

        logger.info("Video readers released")

    def __del__(self):
        """Cleanup on destruction"""
        self.release()
