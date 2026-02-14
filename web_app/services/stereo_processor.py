"""
Stereo processor service for MVS web application
"""

import logging
import threading
import time
from typing import Tuple, Optional, Dict, Any
from services.video_service import VideoService

logger = logging.getLogger(__name__)


class StereoProcessor:
    """Service class for stereo video processing and playback control"""
    
    def __init__(self, video_service: VideoService):
        self.video_service = video_service
        self.is_playing = False
        self.play_thread = None
        self.current_frame = 0
        self.sync_offset = 0
        self.max_frames = 0
        self.play_speed = 1.0  # frames per second
    
    def start_playback(self) -> Tuple[bool, str]:
        """
        Start stereo video playback (manual control mode)
        
        Returns:
            tuple: (success, message)
        """
        try:
            if self.is_playing:
                return False, "Playback is already running"
            
            # Check if videos are loaded
            if not self._check_videos_loaded():
                return False, "No videos loaded for playback"
            
            # Check if max_frames is valid
            if self.max_frames is None or self.max_frames <= 0:
                return False, f"Invalid max_frames: {self.max_frames}. Please reload videos."
            
            self.is_playing = True
            
            logger.info("Stereo playback state set to playing (manual control)")
            return True, f"Playback started - Use frame controls to navigate (Max: {self.max_frames} frames)"
            
        except Exception as e:
            logger.error(f"Error starting playback: {str(e)}", exc_info=True)
            return False, f"Error starting playback: {str(e)}"
    
    def stop_playback(self) -> Tuple[bool, str]:
        """
        Stop stereo video playback
        
        Returns:
            tuple: (success, message)
        """
        try:
            if not self.is_playing:
                return False, "Playback is not running"
            
            self.is_playing = False
            if self.play_thread and self.play_thread.is_alive():
                self.play_thread.join(timeout=1.0)
            
            logger.info("Stereo playback stopped")
            return True, "Playback state: Stopped"
            
        except Exception as e:
            logger.error(f"Error stopping playback: {str(e)}", exc_info=True)
            return False, f"Error stopping playback: {str(e)}"
    
    def set_frame(self, frame_number: int) -> Tuple[bool, str]:
        """
        Set current frame position
        
        Args:
            frame_number: Frame number to set
            
        Returns:
            tuple: (success, message)
        """
        try:
            if frame_number < 0 or frame_number > self.max_frames:
                return False, f"Frame number {frame_number} is out of range (0-{self.max_frames})"
            
            self.current_frame = frame_number
            logger.debug(f"Frame set to: {frame_number}")
            return True, f"Frame set to {frame_number}"
            
        except Exception as e:
            logger.error(f"Error setting frame: {str(e)}", exc_info=True)
            return False, f"Error setting frame: {str(e)}"
    
    def set_sync_offset(self, sync_offset: int) -> Tuple[bool, str]:
        """
        Set sync offset between videos
        
        Args:
            sync_offset: Sync offset in frames
            
        Returns:
            tuple: (success, message)
        """
        try:
            self.sync_offset = sync_offset
            self.video_service.video_processor.set_sync_offset(sync_offset)
            logger.debug(f"Sync offset set to: {sync_offset}")
            return True, f"Sync offset set to {sync_offset}"
            
        except Exception as e:
            logger.error(f"Error setting sync offset: {str(e)}", exc_info=True)
            return False, f"Error setting sync offset: {str(e)}"
    
    def update_frame_display(self) -> str:
        """
        Update frame display information
        
        Returns:
            str: Frame display text
        """
        try:
            return f"Frame: {self.current_frame} | Sync: {self.sync_offset} | Max: {self.max_frames}"
        except Exception as e:
            logger.error(f"Error updating frame display: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    def get_current_frame_info(self) -> Dict[str, Any]:
        """
        Get current frame information for processing
        
        Returns:
            dict: Frame information including frame number, sync offset, and video paths
        """
        try:
            video_1_path = self.video_service._current_video_1_path
            video_2_path = self.video_service._current_video_2_path
            
            return {
                'frame_number': self.current_frame,
                'sync_offset': self.sync_offset,
                'video_1_path': video_1_path,
                'video_2_path': video_2_path,
                'max_frames': self.max_frames,
                'is_playing': self.is_playing
            }
        except Exception as e:
            logger.error(f"Error getting frame info: {str(e)}", exc_info=True)
            return {
                'frame_number': 0,
                'sync_offset': 0,
                'video_1_path': None,
                'video_2_path': None,
                'max_frames': 0,
                'is_playing': False
            }
    
    def get_current_frame_for_ui(self) -> int:
        """
        Get current frame number for UI updates
        
        Returns:
            int: Current frame number
        """
        return self.current_frame
    
    def advance_frame(self) -> Tuple[bool, str]:
        """
        Advance to the next frame if playing
        
        Returns:
            tuple: (success, message)
        """
        try:
            if not self.is_playing:
                return False, "Playback not active"
            
            if self.current_frame >= self.max_frames:
                self.is_playing = False
                return False, "Reached end of video"
            
            self.current_frame += 1
            logger.debug(f"Advanced to frame {self.current_frame}")
            return True, f"Frame {self.current_frame}/{self.max_frames}"
            
        except Exception as e:
            logger.error(f"Error advancing frame: {str(e)}", exc_info=True)
            return False, f"Error advancing frame: {str(e)}"
    
    def _check_videos_loaded(self) -> bool:
        """Check if videos are loaded"""
        try:
            video_1_loaded = self.video_service.video_processor.video_1_reader is not None
            video_2_loaded = self.video_service.video_processor.video_2_reader is not None
            return video_1_loaded or video_2_loaded
        except:
            return False
    
    def _playback_loop(self):
        """Main playback loop - manages frame advancement only"""
        try:
            # Ensure max_frames is a valid number
            if self.max_frames is None or self.max_frames <= 0:
                logger.warning(f"Invalid max_frames: {self.max_frames}, stopping playback")
                self.is_playing = False
                return
                
            while self.is_playing and self.current_frame < self.max_frames:
                # Wait for next frame
                time.sleep(1.0 / self.play_speed)
                
                # Move to next frame
                self.current_frame += 1
                logger.debug(f"Playback loop: frame {self.current_frame}")
                
        except Exception as e:
            logger.error(f"Error in playback loop: {str(e)}", exc_info=True)
        finally:
            self.is_playing = False
            logger.info("Playback loop finished")
