"""
Video reader components for MVS web application
"""

import gradio as gr
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from processing.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


def create_video_display_interface() -> Tuple[gr.File, gr.File, gr.Video, gr.Video, gr.Slider, gr.Slider, gr.Number, gr.Textbox]:
    """
    Create integrated video upload and display interface components
    Each video has its upload widget and display fused into a single visual unit
    
    Returns:
        tuple: (video_1_upload, video_2_upload, video_1_display, video_2_display, frame_slider, sync_offset_slider, frame_number_input, status_display)
    """
    
    with gr.Row():
        with gr.Column(scale=1):
            # Video 1 - Integrated Upload + Display
            with gr.Group():
                gr.Markdown("### 📹 Video 1 (Primary Camera)")
                video_1_upload = gr.File(
                    label="Upload Video 1",
                    file_types=[".mp4"],
                    type="filepath",
                    height=80,
                    container=True
                )
                video_1_display = gr.Video(
                    label="Video 1 Player",
                    height=250,
                    interactive=False,
                    container=True
                )
        
        with gr.Column(scale=1):
            # Video 2 - Integrated Upload + Display
            with gr.Group():
                gr.Markdown("### 📹 Video 2 (Secondary Camera)")
                video_2_upload = gr.File(
                    label="Upload Video 2",
                    file_types=[".mp4"],
                    type="filepath",
                    height=80,
                    container=True
                )
                video_2_display = gr.Video(
                    label="Video 2 Player",
                    height=250,
                    interactive=False,
                    container=True
                )
    
    # Frame Control and Sync Settings - Integrated Control Panel
    with gr.Group():
        gr.Markdown("### 🎬 Video Control Panel")
        
        with gr.Row():
            with gr.Column(scale=3):
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Frame Selector",
                    interactive=True,
                    container=True
                )
            
            with gr.Column(scale=1):
                frame_number_input = gr.Number(
                    value=0,
                    label="Frame #",
                    precision=0,
                    interactive=True,
                    container=True
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                sync_offset_slider = gr.Slider(
                    minimum=-50,
                    maximum=50,
                    value=0,
                    step=1,
                    label="Sync Offset (frames)",
                    interactive=True,
                    container=True
                )
            
            with gr.Column(scale=1):
                status_display = gr.Textbox(
                    label="Status",
                    value="No videos loaded",
                    interactive=False,
                    lines=1,
                    container=True
                )
    
    return video_1_upload, video_2_upload, video_1_display, video_2_display, frame_slider, sync_offset_slider, frame_number_input, status_display


def create_video_controls() -> Tuple[gr.Button]:
    """
    Create video control buttons
    
    Returns:
        tuple: (refresh_frame_btn,)
    """
    
    with gr.Row():
        refresh_frame_btn = gr.Button(
            "🔄 Refresh Display",
            variant="secondary", 
            size="sm"
        )
    
    return refresh_frame_btn,


def update_video_display(
    video_processor: VideoProcessor,
    frame_number: int,
    sync_offset: int
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Update video display with current frame - supports single video display
    
    Args:
        video_processor: VideoProcessor instance
        frame_number: Frame number to display
        sync_offset: Sync offset between videos
        
    Returns:
        tuple: (video_1_path, video_2_path, status_message)
    """
    try:
        logger.debug(f"Updating video display: frame={frame_number}, sync_offset={sync_offset}")
        
        # Check if any videos are loaded
        video_1_loaded = video_processor.video_1_reader is not None
        video_2_loaded = video_processor.video_2_reader is not None
        
        if not video_1_loaded and not video_2_loaded:
            return None, None, "No videos loaded"
        
        # Update sync offset
        video_processor.set_sync_offset(sync_offset)
        
        # Return video paths for display (None for missing videos)
        video_1_path = video_processor.video_1_path if video_1_loaded else None
        video_2_path = video_processor.video_2_path if video_2_loaded else None
        
        # Create status message
        loaded_videos = []
        if video_1_loaded:
            loaded_videos.append("Video 1")
        if video_2_loaded:
            loaded_videos.append("Video 2")
        
        status_msg = f"Frame {frame_number} | Sync offset: {sync_offset} | Loaded: {', '.join(loaded_videos)}"
        return video_1_path, video_2_path, status_msg
            
    except Exception as e:
        logger.error(f"Error updating video display: {str(e)}", exc_info=True)
        return None, None, f"Error updating display: {str(e)}"


def handle_video_upload_and_display(
    video_processor: VideoProcessor,
    video_1_path: Optional[str],
    video_2_path: Optional[str],
    calibration_path: Optional[str]
) -> Tuple[str, int, Optional[str], Optional[str], str]:
    """
    Handle video upload, validation, and display - supports single video uploads
    
    Args:
        video_processor: VideoProcessor instance
        video_1_path: Path to video 1
        video_2_path: Path to video 2
        calibration_path: Path to calibration file
        
    Returns:
        tuple: (status_message, max_frames, video_1_path, video_2_path, upload_status)
    """
    try:
        logger.info(f"Handling integrated video upload: {video_1_path}, {video_2_path}, {calibration_path}")
        
        # Import validation functions
        from utils.validation import validate_video_file, validate_calibration_file, validate_video_compatibility
        
        upload_messages = []
        max_frames = 0
        
        # Validate and load individual videos
        video_1_loaded = False
        video_2_loaded = False
        
        # Handle video 1
        if video_1_path:
            is_valid, error_msg = validate_video_file(video_1_path)
            if is_valid:
                upload_messages.append("✅ Video 1: Valid MP4 file")
                video_1_loaded = True
            else:
                upload_messages.append(f"❌ Video 1: {error_msg}")
                return f"❌ Video 1 validation failed: {error_msg}", 0, None, video_2_path, "\n".join(upload_messages)
        else:
            upload_messages.append("⚠️ Video 1: No file uploaded")
        
        # Handle video 2
        if video_2_path:
            is_valid, error_msg = validate_video_file(video_2_path)
            if is_valid:
                upload_messages.append("✅ Video 2: Valid MP4 file")
                video_2_loaded = True
            else:
                upload_messages.append(f"❌ Video 2: {error_msg}")
                return f"❌ Video 2 validation failed: {error_msg}", 0, video_1_path, None, "\n".join(upload_messages)
        else:
            upload_messages.append("⚠️ Video 2: No file uploaded")
        
        # Handle calibration
        if calibration_path:
            is_valid, error_msg = validate_calibration_file(calibration_path)
            if is_valid:
                upload_messages.append("✅ Calibration: Valid JSON file")
            else:
                upload_messages.append(f"❌ Calibration: {error_msg}")
                return f"❌ Calibration validation failed: {error_msg}", 0, video_1_path, video_2_path, "\n".join(upload_messages)
        else:
            upload_messages.append("⚠️ Calibration: No file uploaded")
        
        # Load videos into processor (supports single video loading)
        if video_1_loaded and video_2_loaded:
            # Both videos available - check compatibility first
            is_compatible, compat_msg = validate_video_compatibility(video_1_path, video_2_path)
            if is_compatible:
                upload_messages.append(f"✅ Compatibility: {compat_msg}")
            else:
                upload_messages.append(f"❌ Compatibility: {compat_msg}")
                return f"❌ Video compatibility failed: {compat_msg}", 0, video_1_path, video_2_path, "\n".join(upload_messages)
            
            # Smart loading - only load videos that aren't already loaded
            current_video_1 = video_processor.video_1_path if hasattr(video_processor, 'video_1_path') else None
            current_video_2 = video_processor.video_2_path if hasattr(video_processor, 'video_2_path') else None
            
            if current_video_1 == video_1_path and current_video_2 == video_2_path:
                # Both videos already loaded with same paths
                upload_messages.append("📹 Both videos already loaded!")
            elif current_video_1 == video_1_path:
                # Video 1 already loaded, check if video 2 needs loading
                if current_video_2 == video_2_path:
                    upload_messages.append("📹 Both videos already loaded!")
                else:
                    success, message = video_processor.load_single_video(video_2_path, "video_2")
                    if success:
                        upload_messages.append("📹 Video 2 loaded, Video 1 already loaded!")
                    else:
                        return f"❌ {message}", 0, video_1_path, video_2_path, "\n".join(upload_messages)
            elif current_video_2 == video_2_path:
                # Video 2 already loaded, check if video 1 needs loading
                if current_video_1 == video_1_path:
                    upload_messages.append("📹 Both videos already loaded!")
                else:
                    success, message = video_processor.load_single_video(video_1_path, "video_1")
                    if success:
                        upload_messages.append("📹 Video 1 loaded, Video 2 already loaded!")
                    else:
                        return f"❌ {message}", 0, video_1_path, video_2_path, "\n".join(upload_messages)
            else:
                # Load both videos (neither is loaded or different videos)
                success, message = video_processor.load_videos(video_1_path, video_2_path)
                if success:
                    upload_messages.append("🎉 Both videos loaded successfully!")
                else:
                    return f"❌ {message}", 0, video_1_path, video_2_path, "\n".join(upload_messages)
            
            # Get video information for max frames calculation
            video_info = video_processor.get_video_info()
            frame_count_1 = video_info['video_1']['frame_count']
            frame_count_2 = video_info['video_2']['frame_count']
            max_frames = max(0, min(frame_count_1, frame_count_2) - 1)
                
        elif video_1_loaded:
            # Only video 1 available - smart loading
            current_video_1 = video_processor.video_1_path if hasattr(video_processor, 'video_1_path') else None
            if current_video_1 == video_1_path:
                upload_messages.append("📹 Video 1 already loaded!")
            else:
                success, message = video_processor.load_single_video(video_1_path, "video_1")
                if success:
                    upload_messages.append("📹 Video 1 loaded successfully!")
                else:
                    return f"❌ {message}", 0, video_1_path, None, "\n".join(upload_messages)
            
            video_info = video_processor.get_video_info()
            frame_count = video_info.get('video_1', {}).get('frame_count', 0)
            max_frames = max(0, frame_count - 1)
                
        elif video_2_loaded:
            # Only video 2 available - smart loading
            current_video_2 = video_processor.video_2_path if hasattr(video_processor, 'video_2_path') else None
            if current_video_2 == video_2_path:
                upload_messages.append("📹 Video 2 already loaded!")
            else:
                success, message = video_processor.load_single_video(video_2_path, "video_2")
                if success:
                    upload_messages.append("📹 Video 2 loaded successfully!")
                else:
                    return f"❌ {message}", 0, None, video_2_path, "\n".join(upload_messages)
            
            video_info = video_processor.get_video_info()
            frame_count = video_info.get('video_2', {}).get('frame_count', 0)
            max_frames = max(0, frame_count - 1)
        else:
            # No videos available
            return "Please upload at least one video", 0, None, None, "\n".join(upload_messages)
        
        return f"✅ Videos ready | Max frames: {max_frames}", max_frames, video_1_path, video_2_path, "\n".join(upload_messages)
            
    except Exception as e:
        logger.error(f"Error in integrated video upload: {str(e)}", exc_info=True)
        return f"❌ Error during upload: {str(e)}", 0, video_1_path, video_2_path, f"❌ Error during upload: {str(e)}"


def load_videos_from_upload(
    video_processor: VideoProcessor,
    video_1_path: Optional[str],
    video_2_path: Optional[str]
) -> Tuple[str, int, Optional[str], Optional[str]]:
    """
    Load videos from uploaded files
    
    Args:
        video_processor: VideoProcessor instance
        video_1_path: Path to video 1
        video_2_path: Path to video 2
        
    Returns:
        tuple: (status_message, max_frames, video_1_path, video_2_path)
    """
    try:
        logger.info(f"Loading videos from upload: {video_1_path}, {video_2_path}")
        
        # Handle different scenarios
        if video_1_path and video_2_path:
            # Both videos available
            success, message = video_processor.load_videos(video_1_path, video_2_path)
            if success:
                video_info = video_processor.get_video_info()
                frame_count_1 = video_info['video_1']['frame_count']
                frame_count_2 = video_info['video_2']['frame_count']
                max_frames = max(0, min(frame_count_1, frame_count_2) - 1)
                return f"✅ {message} | Max frames: {max_frames}", max_frames, video_1_path, video_2_path
            else:
                return f"❌ {message}", 0, None, None
                
        elif video_1_path:
            # Only video 1 available
            success, message = video_processor.load_single_video(video_1_path, "video_1")
            if success:
                video_info = video_processor.get_video_info()
                frame_count = video_info.get('video_1', {}).get('frame_count', 0)
                max_frames = max(0, frame_count - 1)
                return f"✅ {message} | Max frames: {max_frames}", max_frames, video_1_path, None
            else:
                return f"❌ {message}", 0, None, None
                
        elif video_2_path:
            # Only video 2 available
            success, message = video_processor.load_single_video(video_2_path, "video_2")
            if success:
                video_info = video_processor.get_video_info()
                frame_count = video_info.get('video_2', {}).get('frame_count', 0)
                max_frames = max(0, frame_count - 1)
                return f"✅ {message} | Max frames: {max_frames}", max_frames, None, video_2_path
            else:
                return f"❌ {message}", 0, None, None
        else:
            return "Please upload at least one video", 0, None, None
            
    except Exception as e:
        logger.error(f"Error loading videos from upload: {str(e)}", exc_info=True)
        return f"❌ Error loading videos: {str(e)}", 0, None, None


def handle_file_deletion(video_processor: VideoProcessor, video_1_path, video_2_path, calibration_path) -> Tuple[str, int, Optional[str], Optional[str], str]:
    """
    Handle file deletion (when user clicks X on upload widget) and release specific videos
    
    Args:
        video_processor: VideoProcessor instance
        video_1_path: Path to video 1 (may be None if deleted)
        video_2_path: Path to video 2 (may be None if deleted)
        calibration_path: Path to calibration file
        
    Returns:
        tuple: (status_message, max_frames, video_1_path, video_2_path, upload_status)
    """
    try:
        logger.info(f"Handling file deletion: video_1={video_1_path}, video_2={video_2_path}, calibration={calibration_path}")
        
        # Check current state of video processor
        current_video_1 = video_processor.video_1_path if hasattr(video_processor, 'video_1_path') else None
        current_video_2 = video_processor.video_2_path if hasattr(video_processor, 'video_2_path') else None
        
        # Determine which specific video was deleted
        video_1_deleted = current_video_1 is not None and video_1_path is None
        video_2_deleted = current_video_2 is not None and video_2_path is None
        
        # Release only the specific video that was deleted
        if video_1_deleted:
            # Only video 1 deleted - release video 1 specifically
            if hasattr(video_processor, 'video_1_reader') and video_processor.video_1_reader:
                video_processor.video_1_reader.release()
                video_processor.video_1_reader = None
                video_processor.video_1_path = None
            logger.info("Video 1 deleted - released video 1 resources")
        elif video_2_deleted:
            # Only video 2 deleted - release video 2 specifically
            if hasattr(video_processor, 'video_2_reader') and video_processor.video_2_reader:
                video_processor.video_2_reader.release()
                video_processor.video_2_reader = None
                video_processor.video_2_path = None
            logger.info("Video 2 deleted - released video 2 resources")
        
        # Determine which files are still available
        available_files = []
        if video_1_path:
            available_files.append("Video 1")
        if video_2_path:
            available_files.append("Video 2")
        if calibration_path:
            available_files.append("Calibration")
        
        # Calculate max frames based on remaining videos
        max_frames = 0
        if video_1_path and video_2_path:
            # Both videos available - calculate max frames
            try:
                video_info = video_processor.get_video_info()
                frame_count_1 = video_info.get('video_1', {}).get('frame_count', 0)
                frame_count_2 = video_info.get('video_2', {}).get('frame_count', 0)
                max_frames = max(0, min(frame_count_1, frame_count_2) - 1)
            except:
                max_frames = 0
        elif video_1_path:
            # Only video 1 available
            try:
                video_info = video_processor.get_video_info()
                frame_count = video_info.get('video_1', {}).get('frame_count', 0)
                max_frames = max(0, frame_count - 1)
            except:
                max_frames = 0
        elif video_2_path:
            # Only video 2 available
            try:
                video_info = video_processor.get_video_info()
                frame_count = video_info.get('video_2', {}).get('frame_count', 0)
                max_frames = max(0, frame_count - 1)
            except:
                max_frames = 0
        
        if available_files:
            status_msg = f"📁 Files available: {', '.join(available_files)}"
            upload_msg = f"🗑️ Some files removed. Available: {', '.join(available_files)}"
        else:
            status_msg = "📁 No files loaded"
            upload_msg = "🗑️ All files removed"
        
        return status_msg, max_frames, video_1_path, video_2_path, upload_msg
            
    except Exception as e:
        logger.error(f"Error handling file deletion: {str(e)}", exc_info=True)
        return f"❌ Error: {str(e)}", 0, video_1_path, video_2_path, f"❌ Error handling deletion: {str(e)}"


def release_videos(video_processor: VideoProcessor) -> str:
    """
    Release loaded videos
    
    Args:
        video_processor: VideoProcessor instance
        
    Returns:
        str: Status message
    """
    try:
        video_processor.release()
        return "✅ Videos released successfully"
    except Exception as e:
        logger.error(f"Error releasing videos: {str(e)}")
        return f"❌ Error releasing videos: {str(e)}"
