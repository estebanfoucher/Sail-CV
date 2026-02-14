"""
File upload components for MVS web application
"""

import gradio as gr
from typing import Tuple, Optional


def create_file_upload_interface() -> Tuple[gr.File, gr.File, gr.File, gr.Button, gr.Textbox]:
    """
    Create the file upload interface components
    
    Returns:
        tuple: (video_1_upload, video_2_upload, calibration_upload, upload_btn, status_box)
    """
    
    with gr.Row():
        with gr.Column(scale=1):
            # Video uploads
            video_1_upload = gr.File(
                label="📹 Video 1 (Primary Camera)",
                file_types=[".mp4"],
                type="filepath",
                height=100
            )
            
            video_2_upload = gr.File(
                label="📹 Video 2 (Secondary Camera)", 
                file_types=[".mp4"],
                type="filepath",
                height=100
            )
            
            # Calibration upload
            calibration_upload = gr.File(
                label="📐 Calibration File",
                file_types=[".json"],
                type="filepath",
                height=100
            )
            
            # Upload button
            upload_btn = gr.Button(
                "📤 Upload & Validate Files", 
                variant="primary",
                size="lg"
            )
            
            # Status display
            status_box = gr.Textbox(
                label="📋 Upload Status",
                value="Ready to upload files...",
                interactive=False,
                lines=4,
                max_lines=8
            )
    
    return video_1_upload, video_2_upload, calibration_upload, upload_btn, status_box


def create_file_info_display() -> gr.Markdown:
    """
    Create file information display component
    
    Returns:
        gr.Markdown: File information display component
    """
    
    info_display = gr.Markdown(
        """
        ### 📊 File Information
        - **Video 1**: Not uploaded
        - **Video 2**: Not uploaded  
        - **Calibration**: Not uploaded
        
        *File details will appear here after upload*
        """,
        visible=True
    )
    
    return info_display


def update_file_info_display(video_1_info: dict, video_2_info: dict, calibration_info: dict) -> str:
    """
    Update file information display with uploaded file details
    
    Args:
        video_1_info: Video 1 information
        video_2_info: Video 2 information  
        calibration_info: Calibration information
        
    Returns:
        str: Updated markdown content
    """
    
    def format_video_info(info: dict, name: str) -> str:
        if not info:
            return f"- **{name}**: Not uploaded"
        
        return f"""
        - **{name}**: 
          - Resolution: {info.get('resolution', 'Unknown')}
          - FPS: {info.get('fps', 0):.2f}
          - Duration: {info.get('duration', 0):.1f}s
          - Frames: {info.get('frame_count', 0)}
        """
    
    def format_calibration_info(info: dict) -> str:
        if not info:
            return "- **Calibration**: Not uploaded"
        
        cameras = []
        if info.get('has_camera_1'):
            cameras.append("Camera 1")
        if info.get('has_camera_2'):
            cameras.append("Camera 2")
        
        camera_str = ", ".join(cameras) if cameras else "None"
        
        return f"""
        - **Calibration**: 
          - Cameras: {camera_str}
          - Camera Count: {info.get('camera_count', 0)}
        """
    
    content = f"""
    ### 📊 File Information
    
    {format_video_info(video_1_info, "Video 1")}
    
    {format_video_info(video_2_info, "Video 2")}
    
    {format_calibration_info(calibration_info)}
    """
    
    return content


def create_upload_help() -> gr.Markdown:
    """
    Create upload help and instructions
    
    Returns:
        gr.Markdown: Help content
    """
    
    help_content = """
    ### 📖 Upload Instructions
    
    **Required Files:**
    1. **Video 1**: Primary camera MP4 file
    2. **Video 2**: Secondary camera MP4 file (synchronized with Video 1)
    3. **Calibration**: JSON file containing stereo camera calibration parameters
    
    **File Requirements:**
    - Videos must be in MP4 format
    - Videos should be synchronized (same frame rate, similar duration)
    - Calibration file must be valid JSON with camera_1 and camera_2 sections
    
    **Tips:**
    - Ensure videos are properly synchronized before upload
    - Check that calibration file matches your camera setup
    - File sizes are displayed after upload for verification
    """
    
    return gr.Markdown(help_content)
