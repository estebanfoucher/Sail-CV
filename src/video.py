import cv2
import subprocess
import numpy as np
import os
from mv_utils.image_utils import apply_exif_transpose_to_frames


class FFmpegVideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]):
        self.output_path = output_path
        self.width, self.height = frame_size
        self.fps = fps

        # Launch ffmpeg process
        self.process = subprocess.Popen([
            'ffmpeg',
            '-y',  # overwrite
            '-loglevel', 'error',  # only show error messages
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # read from stdin
            '-an',  # no audio
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            self.output_path
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        """Write a single frame"""
        self.process.stdin.write(frame.tobytes())

    def release(self):
        self.process.stdin.close()
        self.process.wait()

class VideoReader:
    """Video reader using cv2"""
    def __init__(self, video_path: str, start_frame: int = 0):
        self.video_path = video_path
        self._check_video_exists()
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.width, self.height)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_frame = start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    def _check_video_exists(self):
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file {self.video_path} does not exist")
    
    def read(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Apply EXIF transpose and ensure landscape orientation
            corrected_frame = self._ensure_landscape_orientation(frame)
            return ret, corrected_frame
        return ret, frame
    
    def _ensure_landscape_orientation(self, frame: np.ndarray) -> np.ndarray:
        """
        Ensure frame is in landscape orientation (width >= height).
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            
        Returns:
            Frame in landscape orientation
        """
        # Apply EXIF transpose first
        corrected_frame = apply_exif_transpose_to_frames([frame])[0]
        
        # Get current dimensions
        height, width = corrected_frame.shape[:2]
        
        # If already landscape (width >= height), return as is
        if width >= height:
            return corrected_frame
        
        # If portrait (height > width), rotate 90 degrees clockwise
        # This ensures landscape orientation
        rotated_frame = cv2.rotate(corrected_frame, cv2.ROTATE_90_CLOCKWISE)
        
        return rotated_frame
    
    def release(self):
        self.cap.release()

class StereoVideoReader:
    def __init__(self, video_1_path: str, video_2_path: str, start_video_1_frame: int = 0, start_video_2_frame: int = 0):
        self.video_1_path = video_1_path
        self.video_2_path = video_2_path
        self.video_1 = VideoReader(self.video_1_path, start_frame=start_video_1_frame)
        self.video_2 = VideoReader(self.video_2_path, start_frame=start_video_2_frame)
        
    def read(self):
        ret_1, frame_1 = self.video_1.read()
        ret_2, frame_2 = self.video_2.read()
        return ret_1, ret_2, frame_1, frame_2
    
    def release(self):
        self.video_1.release()
        self.video_2.release()
        
    def render_side_by_side(self, frame_1: np.ndarray, frame_2: np.ndarray, 
                           add_labels: bool = True, separator_width: int = 5,
                           label_font_scale: float = 1.0, label_thickness: int = 2) -> np.ndarray:
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
            separator = np.full((target_height, separator_width, 3), 128, dtype=np.uint8)
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
            (text_w1, text_h1), _ = cv2.getTextSize(label_1, font, label_font_scale, label_thickness)
            label_x1 = label_padding
            label_y1 = label_height
            
            # Draw background rectangle for label 1
            cv2.rectangle(combined_frame, 
                         (label_x1 - label_padding, label_y1 - text_h1 - label_padding),
                         (label_x1 + text_w1 + label_padding, label_y1 + label_padding),
                         label_bg_color, -1)
            
            # Draw text for label 1
            cv2.putText(combined_frame, label_1, (label_x1, label_y1), 
                       font, label_font_scale, label_color, label_thickness)
            
            # Camera 2 label (right side)
            label_2 = "Camera 2"
            (text_w2, text_h2), _ = cv2.getTextSize(label_2, font, label_font_scale, label_thickness)
            label_x2 = w1 + separator_width + label_padding
            label_y2 = label_height
            
            # Draw background rectangle for label 2
            cv2.rectangle(combined_frame,
                         (label_x2 - label_padding, label_y2 - text_h2 - label_padding),
                         (label_x2 + text_w2 + label_padding, label_y2 + label_padding),
                         label_bg_color, -1)
            
            # Draw text for label 2
            cv2.putText(combined_frame, label_2, (label_x2, label_y2),
                       font, label_font_scale, label_color, label_thickness)
        
        return combined_frame
        