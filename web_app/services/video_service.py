"""
Video service for handling video operations
"""

import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import cv2

from processing.video_processor import VideoProcessor
from utils.validation import validate_video_file, validate_calibration_file, validate_video_compatibility
from exceptions import ValidationError, VideoProcessingError, VideoCompatibilityError
from constants import STATUS_MESSAGES, ERROR_MESSAGES, SUCCESS_MESSAGES

logger = logging.getLogger(__name__)


class VideoService:
    """Service class for video operations"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self._current_video_1_path: Optional[str] = None
        self._current_video_2_path: Optional[str] = None
        self._current_calibration_path: Optional[str] = None
    
    def validate_and_load_videos(
        self, 
        video_1_path: Optional[str], 
        video_2_path: Optional[str], 
        calibration_path: Optional[str]
    ) -> Tuple[str, int, Optional[str], Optional[str], str, int]:
        """
        Validate and load videos with smart loading logic
        
        Returns:
            tuple: (status_message, max_frames, video_1_path, video_2_path, upload_status, frame_slider_max)
        """
        try:
            logger.info(f"Validating and loading videos: {video_1_path}, {video_2_path}, {calibration_path}")
            
            upload_messages = []
            max_frames = 0
            
            # Validate individual files
            video_1_loaded = self._validate_single_video(video_1_path, "Video 1", upload_messages)
            video_2_loaded = self._validate_single_video(video_2_path, "Video 2", upload_messages)
            calibration_loaded = self._validate_calibration(calibration_path, upload_messages)
            
            # Load videos with smart loading
            if video_1_loaded and video_2_loaded:
                max_frames = self._load_both_videos(video_1_path, video_2_path, upload_messages)
            elif video_1_loaded:
                max_frames = self._load_single_video(video_1_path, "video_1", upload_messages)
            elif video_2_loaded:
                max_frames = self._load_single_video(video_2_path, "video_2", upload_messages)
            else:
                return STATUS_MESSAGES['upload_at_least_one'], 0, None, None, "\n".join(upload_messages), 0
            
            
            status_msg = f"{STATUS_MESSAGES['videos_ready']} | Max frames: {max_frames}"
            return status_msg, max_frames, video_1_path, video_2_path, "\n".join(upload_messages), max_frames
            
        except Exception as e:
            logger.error(f"Error in validate_and_load_videos: {str(e)}", exc_info=True)
            return f"{ERROR_MESSAGES['unexpected_error']}: {str(e)}", 0, video_1_path, video_2_path, f"{ERROR_MESSAGES['unexpected_error']}: {str(e)}", 0
    
    def handle_file_deletion(
        self, 
        video_1_path: Optional[str], 
        video_2_path: Optional[str], 
        calibration_path: Optional[str]
    ) -> Tuple[str, int, Optional[str], Optional[str], str, int]:
        """
        Handle file deletion and release specific video resources
        
        Returns:
            tuple: (status_message, max_frames, video_1_path, video_2_path, upload_status, frame_slider_max)
        """
        try:
            logger.info(f"Handling file deletion: video_1={video_1_path}, video_2={video_2_path}, calibration={calibration_path}")
            
            # Determine which video was deleted
            video_1_deleted = self._current_video_1_path is not None and video_1_path is None
            video_2_deleted = self._current_video_2_path is not None and video_2_path is None
            
            # Release specific video resources
            if video_1_deleted:
                self._release_video("video_1")
                logger.info("Video 1 deleted - released video 1 resources")
            elif video_2_deleted:
                self._release_video("video_2")
                logger.info("Video 2 deleted - released video 2 resources")
            
            # Update current paths
            self._current_video_1_path = video_1_path
            self._current_video_2_path = video_2_path
            self._current_calibration_path = calibration_path
            
            # Calculate max frames and create status
            max_frames = self._calculate_max_frames(video_1_path, video_2_path)
            available_files = self._get_available_files(video_1_path, video_2_path, calibration_path)
            
            if available_files:
                status_msg = f"📁 {STATUS_MESSAGES['files_available']}: {', '.join(available_files)}"
                upload_msg = f"🗑️ {STATUS_MESSAGES['some_files_removed']}. Available: {', '.join(available_files)}"
            else:
                status_msg = f"📁 {STATUS_MESSAGES['no_videos']}"
                upload_msg = f"🗑️ {STATUS_MESSAGES['all_files_removed']}"
            
            return status_msg, max_frames, video_1_path, video_2_path, upload_msg, max_frames
            
        except Exception as e:
            logger.error(f"Error handling file deletion: {str(e)}", exc_info=True)
            return f"{ERROR_MESSAGES['unexpected_error']}: {str(e)}", 0, video_1_path, video_2_path, f"{ERROR_MESSAGES['unexpected_error']}: {str(e)}", 0
    
    def update_video_display(self, frame_number: int, sync_offset: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Update video display with current frame and sync offset
        
        Returns:
            tuple: (video_1_frame, video_2_frame, status_message)
        """
        try:
            logger.debug(f"Updating video display: frame={frame_number}, sync_offset={sync_offset}")
            
            # Check if any videos are loaded
            video_1_loaded = self.video_processor.video_1_reader is not None
            video_2_loaded = self.video_processor.video_2_reader is not None
            
            if not video_1_loaded and not video_2_loaded:
                return None, None, STATUS_MESSAGES['no_videos']
            
            # Update sync offset
            self.video_processor.set_sync_offset(sync_offset)
            
            # Extract frames from videos
            video_1_frame = None
            video_2_frame = None
            
            if video_1_loaded:
                success, frame, msg = self.video_processor.read_frame(1, frame_number)
                if success and frame is not None:
                    video_1_frame = frame
                    logger.debug(f"Video 1 frame {frame_number} extracted successfully")
                else:
                    logger.warning(f"Failed to extract Video 1 frame {frame_number}: {msg}")
            
            if video_2_loaded:
                success, frame, msg = self.video_processor.read_frame(2, frame_number)
                if success and frame is not None:
                    video_2_frame = frame
                    logger.debug(f"Video 2 frame {frame_number} extracted successfully")
                else:
                    logger.warning(f"Failed to extract Video 2 frame {frame_number}: {msg}")
            
            # Create status message
            loaded_videos = []
            if video_1_loaded:
                loaded_videos.append("Video 1")
            if video_2_loaded:
                loaded_videos.append("Video 2")
            
            status_msg = f"Frame {frame_number} | Videos synced | Loaded: {', '.join(loaded_videos)}"
            return video_1_frame, video_2_frame, status_msg
                
        except Exception as e:
            logger.error(f"Error updating video display: {str(e)}", exc_info=True)
            return None, None, f"{ERROR_MESSAGES['unexpected_error']}: {str(e)}"
    
    def extract_frames_for_selection(self, frame_number: int, sync_offset: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Extract frames for image selection (returns numpy arrays for display)
        
        Returns:
            tuple: (frame_1_image, frame_2_image, status_message)
        """
        try:
            logger.debug(f"Extracting frames for selection: frame={frame_number}, sync_offset={sync_offset}")
            
            # Check if any videos are loaded
            video_1_loaded = self.video_processor.video_1_reader is not None
            video_2_loaded = self.video_processor.video_2_reader is not None
            
            if not video_1_loaded and not video_2_loaded:
                return None, None, STATUS_MESSAGES['no_videos']
            
            # Update sync offset
            self.video_processor.set_sync_offset(sync_offset)
            
            # Extract frames from videos
            frame_1_image = None
            frame_2_image = None
            
            if video_1_loaded:
                success, frame, msg = self.video_processor.read_frame(1, frame_number)
                if success and frame is not None:
                    # Convert BGR to RGB for display
                    frame_1_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Video 1 frame {frame_number} extracted successfully")
                else:
                    logger.warning(f"Failed to extract Video 1 frame {frame_number}: {msg}")
            
            if video_2_loaded:
                success, frame, msg = self.video_processor.read_frame(2, frame_number)
                if success and frame is not None:
                    # Convert BGR to RGB for display
                    frame_2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Video 2 frame {frame_number} extracted successfully")
                else:
                    logger.warning(f"Failed to extract Video 2 frame {frame_number}: {msg}")
            
            # Create status message
            loaded_videos = []
            if video_1_loaded:
                loaded_videos.append("Video 1")
            if video_2_loaded:
                loaded_videos.append("Video 2")
            
            status_msg = f"Frames extracted: Video 1 frame {frame_number}, Video 2 frame {frame_number} | Loaded: {', '.join(loaded_videos)}"
            return frame_1_image, frame_2_image, status_msg
                
        except Exception as e:
            logger.error(f"Error extracting frames for selection: {str(e)}", exc_info=True)
            return None, None, f"{ERROR_MESSAGES['unexpected_error']}: {str(e)}"
    
    def _validate_single_video(self, video_path: Optional[str], video_name: str, upload_messages: list) -> bool:
        """Validate a single video file"""
        if not video_path:
            upload_messages.append(f"⚠️ {video_name}: No file uploaded")
            return False
        
        try:
            is_valid, error_msg = validate_video_file(video_path)
            if is_valid:
                upload_messages.append(f"✅ {video_name}: {SUCCESS_MESSAGES['video_valid']}")
                return True
            else:
                upload_messages.append(f"❌ {video_name}: {error_msg}")
                raise ValidationError(f"{video_name} validation failed: {error_msg}")
        except Exception as e:
            logger.error(f"Error validating {video_name}: {str(e)}")
            raise ValidationError(f"Error validating {video_name}: {str(e)}")
    
    def _validate_calibration(self, calibration_path: Optional[str], upload_messages: list) -> bool:
        """Validate calibration file"""
        if not calibration_path:
            upload_messages.append("⚠️ Calibration: No file uploaded")
            return True  # Calibration is optional
        
        try:
            is_valid, error_msg = validate_calibration_file(calibration_path)
            if is_valid:
                upload_messages.append(f"✅ Calibration: {SUCCESS_MESSAGES['calibration_valid']}")
                return True
            else:
                upload_messages.append(f"❌ Calibration: {error_msg}")
                raise ValidationError(f"Calibration validation failed: {error_msg}")
        except Exception as e:
            logger.error(f"Error validating calibration: {str(e)}")
            raise ValidationError(f"Error validating calibration: {str(e)}")
    
    def _load_both_videos(self, video_1_path: str, video_2_path: str, upload_messages: list) -> int:
        """Load both videos with smart loading logic"""
        current_video_1 = self._current_video_1_path
        current_video_2 = self._current_video_2_path
        
        # Smart loading - only load videos that aren't already loaded
        if current_video_1 == video_1_path and current_video_2 == video_2_path:
            upload_messages.append(f"📹 {STATUS_MESSAGES['both_videos_already_loaded']}")
        elif current_video_1 == video_1_path:
            # Video 1 already loaded, only load video 2
            success, message = self.video_processor.load_single_video(video_2_path, "video_2")
            if success:
                upload_messages.append("📹 Video 2 loaded, Video 1 already loaded!")
            else:
                raise VideoProcessingError(message)
        elif current_video_2 == video_2_path:
            # Video 2 already loaded, only load video 1
            success, message = self.video_processor.load_single_video(video_1_path, "video_1")
            if success:
                upload_messages.append("📹 Video 1 loaded, Video 2 already loaded!")
            else:
                raise VideoProcessingError(message)
        else:
            # Load both videos (neither is loaded or different videos)
            # Check compatibility first
            is_compatible, compat_msg = validate_video_compatibility(video_1_path, video_2_path)
            if not is_compatible:
                raise VideoCompatibilityError(f"Video compatibility failed: {compat_msg}")
            
            upload_messages.append(f"✅ Compatibility: {compat_msg}")
            
            success, message = self.video_processor.load_videos(video_1_path, video_2_path)
            if success:
                upload_messages.append(f"🎉 {STATUS_MESSAGES['both_videos_loaded']}")
            else:
                raise VideoProcessingError(message)
        
        # Update current paths
        self._current_video_1_path = video_1_path
        self._current_video_2_path = video_2_path
        
        # Get video information for max frames calculation
        video_info = self.video_processor.get_video_info()
        frame_count_1 = video_info['video_1']['frame_count']
        frame_count_2 = video_info['video_2']['frame_count']
        max_frames = max(0, min(frame_count_1, frame_count_2) - 1)
        
        
        return max_frames
    
    def _load_single_video(self, video_path: str, video_name: str, upload_messages: list) -> int:
        """Load a single video with smart loading logic"""
        current_video_path = self._current_video_1_path if video_name == "video_1" else self._current_video_2_path
        
        if current_video_path == video_path:
            upload_messages.append(f"📹 {STATUS_MESSAGES['video_already_loaded']}")
        else:
            success, message = self.video_processor.load_single_video(video_path, video_name)
            if success:
                upload_messages.append(f"📹 {STATUS_MESSAGES['video_loaded']}")
            else:
                raise VideoProcessingError(message)
        
        # Update current path
        if video_name == "video_1":
            self._current_video_1_path = video_path
        else:
            self._current_video_2_path = video_path
        
        # Get video information for max frames calculation
        video_info = self.video_processor.get_video_info()
        frame_count = video_info.get(video_name, {}).get('frame_count', 0)
        max_frames = max(0, frame_count - 1)
        
        
        return max_frames
    
    def _release_video(self, video_name: str):
        """Release specific video resources"""
        if video_name == "video_1":
            if hasattr(self.video_processor, 'video_1_reader') and self.video_processor.video_1_reader:
                self.video_processor.video_1_reader.release()
                self.video_processor.video_1_reader = None
                self.video_processor.video_1_path = None
        elif video_name == "video_2":
            if hasattr(self.video_processor, 'video_2_reader') and self.video_processor.video_2_reader:
                self.video_processor.video_2_reader.release()
                self.video_processor.video_2_reader = None
                self.video_processor.video_2_path = None
    
    def _calculate_max_frames(self, video_1_path: Optional[str], video_2_path: Optional[str]) -> int:
        """Calculate max frames based on available videos"""
        if video_1_path and video_2_path:
            try:
                video_info = self.video_processor.get_video_info()
                frame_count_1 = video_info.get('video_1', {}).get('frame_count', 0)
                frame_count_2 = video_info.get('video_2', {}).get('frame_count', 0)
                return max(0, min(frame_count_1, frame_count_2) - 1)
            except:
                return 0
        elif video_1_path:
            try:
                video_info = self.video_processor.get_video_info()
                frame_count = video_info.get('video_1', {}).get('frame_count', 0)
                return max(0, frame_count - 1)
            except:
                return 0
        elif video_2_path:
            try:
                video_info = self.video_processor.get_video_info()
                frame_count = video_info.get('video_2', {}).get('frame_count', 0)
                return max(0, frame_count - 1)
            except:
                return 0
        else:
            return 0
    
    def _get_available_files(self, video_1_path: Optional[str], video_2_path: Optional[str], calibration_path: Optional[str]) -> list:
        """Get list of available files"""
        available_files = []
        if video_1_path:
            available_files.append("Video 1")
        if video_2_path:
            available_files.append("Video 2")
        if calibration_path:
            available_files.append("Calibration")
        return available_files
    
    def get_frame_slider_range(self) -> Tuple[int, int]:
        """
        Get the frame slider range based on loaded videos
        
        Returns:
            tuple: (minimum, maximum) for frame slider
        """
        try:
            max_frames = self._calculate_max_frames(self._current_video_1_path, self._current_video_2_path)
            return 0, max_frames
        except Exception as e:
            logger.error(f"Error getting frame slider range: {str(e)}")
            return 0, 0
