"""
UI Components Factory for MVS web application
"""

import gradio as gr
from typing import Tuple, Dict, Any
from constants import UI_LABELS, DEFAULT_VIDEO_HEIGHT, UPLOAD_WIDGET_HEIGHT, DEFAULT_FRAME_SLIDER_MAX, STATUS_MESSAGES
from components.processing_interface import ProcessingInterface
from components.batch_processing_interface import BatchProcessingInterface
from components.point_cloud_viz import PointCloudVisualizer
from components.sam_interface import SAMInterface


class UIFactory:
    """Factory class for creating UI components"""
    
    @staticmethod
    def create_video_upload_and_display() -> Tuple[gr.File, gr.File, gr.Video, gr.Video]:
        """
        Create integrated video upload and display components
        
        Returns:
            tuple: (video_1_upload, video_2_upload, video_1_display, video_2_display)
        """
        with gr.Row():
            with gr.Column(scale=1):
                # Video 1 - Integrated Upload + Display
                with gr.Group():
                    gr.Markdown(f"### {UI_LABELS['video_1_title']}")
                    video_1_upload = gr.File(
                        label=UI_LABELS['upload_video_1'],
                        file_types=[".mp4"],
                        type="filepath",
                        height=UPLOAD_WIDGET_HEIGHT,
                        container=True
                    )
                    video_1_display = gr.Video(
                        label=UI_LABELS['video_1_player'],
                        height=DEFAULT_VIDEO_HEIGHT,
                        interactive=False,
                        container=True
                    )
            
            with gr.Column(scale=1):
                # Video 2 - Integrated Upload + Display
                with gr.Group():
                    gr.Markdown(f"### {UI_LABELS['video_2_title']}")
                    video_2_upload = gr.File(
                        label=UI_LABELS['upload_video_2'],
                        file_types=[".mp4"],
                        type="filepath",
                        height=UPLOAD_WIDGET_HEIGHT,
                        container=True
                    )
                    video_2_display = gr.Video(
                        label=UI_LABELS['video_2_player'],
                        height=DEFAULT_VIDEO_HEIGHT,
                        interactive=False,
                        container=True
                    )
        
        return video_1_upload, video_2_upload, video_1_display, video_2_display
    
    
    @staticmethod
    def create_frame_selector() -> gr.Slider:
        """
        Create frame selector component
        
        Returns:
            gr.Slider: Frame slider component
        """
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📸 Frame Selection")
                
                # Frame controls - simplified single frame selector
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=0,  # Will be updated dynamically when videos are loaded
                    value=0,
                    step=1,
                    label="Frame Position (0 to max_frame)",
                    interactive=True
                )
        
        return frame_slider
    
    @staticmethod
    def create_selected_images() -> Tuple[gr.Image, gr.Image]:
        """
        Create selected images display components
        
        Returns:
            tuple: (selected_image_1, selected_image_2)
        """
        with gr.Row():
            with gr.Column(scale=1):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_image_1 = gr.Image(
                            label="Selected Frame - Video 1",
                            height=DEFAULT_VIDEO_HEIGHT,
                            interactive=False,
                            container=True
                        )
                    
                    with gr.Column(scale=1):
                        selected_image_2 = gr.Image(
                            label="Selected Frame - Video 2",
                            height=DEFAULT_VIDEO_HEIGHT,
                            interactive=False,
                            container=True
                        )
        
        return selected_image_1, selected_image_2
    
    @staticmethod
    def create_playback_timer() -> gr.Timer:
        """
        Create a timer for playback updates
        
        Returns:
            gr.Timer: Timer component for periodic updates
        """
        return gr.Timer(value=0.1, active=False)  # 100ms interval
    
    
    @staticmethod
    def create_calibration_upload() -> Tuple[gr.File, gr.Textbox]:
        """
        Create calibration upload component
        
        Returns:
            tuple: (calibration_upload, upload_status)
        """
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 Calibration File")
                calibration_upload = gr.File(
                    label=UI_LABELS['calibration_file'],
                    file_types=[".json"],
                    type="filepath"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 📋 Upload Status")
                upload_status = gr.Textbox(
                    label=UI_LABELS['upload_status'],
                    value=STATUS_MESSAGES['ready'],
                    interactive=False,
                    lines=8,
                    max_lines=12,
                    elem_classes=["status-box"]
                )
        
        return calibration_upload, upload_status
    
    
    @staticmethod
    def create_complete_interface() -> Dict[str, Any]:
        """
        Create complete UI interface
        
        Returns:
            dict: Dictionary containing all UI components
        """
        # Create all components
        video_1_upload, video_2_upload, video_1_display, video_2_display = UIFactory.create_video_upload_and_display()
        calibration_upload, upload_status = UIFactory.create_calibration_upload()
        frame_slider = UIFactory.create_frame_selector()
        selected_image_1, selected_image_2 = UIFactory.create_selected_images()
        process_pair_btn, download_ply, processing_status, update_status_btn, render_camera_toggle, subsample_param = ProcessingInterface.create_processing_controls()
        activate_sam_btn, deactivate_sam_btn, point_prompt_1, point_prompt_2, compute_masks_btn, sam_status = SAMInterface.create_sam_controls()
        model3d_viewer, viz_status = PointCloudVisualizer.create_point_cloud_viewer()
        folder_name, start_frame, end_frame, step, process_all_btn, batch_progress, batch_status, download_batch, update_batch_status_btn = BatchProcessingInterface.create_batch_processing_controls()
        
        return {
            'video_1_upload': video_1_upload,
            'video_2_upload': video_2_upload,
            'video_1_display': video_1_display,
            'video_2_display': video_2_display,
            'frame_slider': frame_slider,
            'selected_image_1': selected_image_1,
            'selected_image_2': selected_image_2,
            'calibration_upload': calibration_upload,
            'upload_status': upload_status,
            'process_pair_btn': process_pair_btn,
            'download_ply': download_ply,
            'processing_status': processing_status,
            'update_status_btn': update_status_btn,
            'render_camera_toggle': render_camera_toggle,
            'subsample_param': subsample_param,
            'activate_sam_btn': activate_sam_btn,
            'deactivate_sam_btn': deactivate_sam_btn,
            'point_prompt_1': point_prompt_1,
            'point_prompt_2': point_prompt_2,
            'compute_masks_btn': compute_masks_btn,
            'sam_status': sam_status,
            'model3d_viewer': model3d_viewer,
            'viz_status': viz_status,
            'folder_name': folder_name,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'step': step,
            'process_all_btn': process_all_btn,
            'batch_progress': batch_progress,
            'batch_status': batch_status,
            'download_batch': download_batch,
            'update_batch_status_btn': update_batch_status_btn
        }
