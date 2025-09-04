import cv2
import numpy as np
import os
from typing import Tuple

class FrameExtractor:
    def __init__(self, video_path: str, output_dir: str, list_of_frames: list[int]):
        self.video_path = video_path
        self.output_dir = output_dir
        self.list_of_frames = list_of_frames

    def extract_frames(self):
        """
        Extract frames from the video at the given list of frames.
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_numbers = np.array(self.list_of_frames)
        frame_numbers = frame_numbers[frame_numbers < frame_count] # remove frames that are out of range
        frame_numbers = frame_numbers[frame_numbers >= 0] # remove negative frames
        
        for frame_number in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{self.output_dir}/frame_{frame_number}.jpg", frame)
        cap.release()

class Video:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frame_count(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    
    def get_fps(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # If FPS is 0 or invalid, try to calculate it manually
        if fps <= 0:
            fps = self._calculate_fps_manually()
        
        return fps
    
    def _calculate_fps_manually(self):
        """Calculate FPS manually by reading frames and timing"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Read first few frames to get timing
        frame_count = 0
        start_time = None
        end_time = None
        
        # Read up to 30 frames or until we have enough timing data
        max_frames = min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Get timestamp for first and last frame
            if i == 0:
                start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if i == max_frames - 1:
                end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        cap.release()
        
        # Calculate FPS based on timing
        if start_time is not None and end_time is not None and end_time > start_time:
            duration = end_time - start_time
            if duration > 0:
                calculated_fps = frame_count / duration
                # Sanity check: FPS should be reasonable (between 1 and 120)
                if 1 <= calculated_fps <= 120:
                    return calculated_fps
        
        # Fallback: assume 30 FPS for GoPro videos if we can't determine it
        print(f"Warning: Could not determine FPS for {self.video_path}, using default 30 FPS")
        return 30.0
    
    def get_frames(self, list_of_frames: list[int]):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        for frame_number in list_of_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    def get_resolution(self) -> Tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

def get_unique_video_name(folder_path: str) -> str:
    """
    Get the video name in the folder (mp4, MP4)
    """
    video_name = None
    for file in os.listdir(folder_path):
        if file.endswith(('.mp4', '.MP4')):
            video_name = file
            break
    return video_name
    
    