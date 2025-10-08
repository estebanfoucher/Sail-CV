"""
Event handler service for MVS web application
"""

import logging
import os
import threading
import json
from typing import Tuple, Optional, Dict, Any
import numpy as np
import gradio as gr
from services.video_service import VideoService
from services.processing_service import ProcessingService
from services.batch_processing_service import BatchProcessingService
from services.sam_service import SAMService
from components.point_cloud_viz import PointCloudVisualizer

logger = logging.getLogger(__name__)


class EventHandler:
    """Service class for handling UI events"""
    
    def __init__(self):
        self.video_service = VideoService()
        self.processing_service = ProcessingService()
        self.batch_processing_service = BatchProcessingService()
        self.sam_service = SAMService()
        self._status_file_path = "/tmp/mvs_model_status.txt"
        self._start_mast3r_loading()
    
    def _start_mast3r_loading(self):
        """Start MASt3R model loading in background thread"""
        def load_model_async():
            try:
                logger.info("Loading MASt3R model in background...")
                self._write_status("loading")
                
                success, _ = self.processing_service.load_mast3r_model()
                status = "loaded" if success else "failed"
                self._write_status(status)
                
                if success:
                    logger.info("MASt3R model loaded successfully in background")
                else:
                    logger.error("Failed to load MASt3R model in background")
                    
            except Exception as e:
                logger.error(f"Error loading MASt3R model in background: {str(e)}", exc_info=True)
                self._write_status("error")
        
        threading.Thread(target=load_model_async, daemon=True).start()
        logger.info("MASt3R model loading started in background thread")
    
    def _write_status(self, status: str):
        """Write status to file"""
        with open(self._status_file_path, 'w') as f:
            f.write(status)
    
    def _read_status(self) -> str:
        """Read status from file"""
        try:
            if os.path.exists(self._status_file_path):
                with open(self._status_file_path, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading status file: {str(e)}")
        return "loading"
    
    def handle_file_change_with_slider_update(
        self, 
        video_1_path: Optional[str], 
        video_2_path: Optional[str], 
        calibration_path: Optional[str]
    ) -> Tuple[int, Optional[str], Optional[str], str, gr.Slider]:
        """Handle file change events with frame slider update"""
        try:
            logger.info(f"Handling file change: video_1={video_1_path}, video_2={video_2_path}, calibration={calibration_path}")
            
            # Check if only calibration changed
            current_video_1 = self.video_service._current_video_1_path
            current_video_2 = self.video_service._current_video_2_path
            current_calibration = self.video_service._current_calibration_path
            
            if (calibration_path != current_calibration and 
                video_1_path == current_video_1 and 
                video_2_path == current_video_2):
                return self._handle_calibration_only_change(video_1_path, video_2_path, calibration_path)
            
            # Handle video changes
            if ((current_video_1 is not None and video_1_path is None) or 
                (current_video_2 is not None and video_2_path is None)):
                result = self.video_service.handle_file_deletion(video_1_path, video_2_path, calibration_path)
            else:
                result = self.video_service.validate_and_load_videos(video_1_path, video_2_path, calibration_path)
            
            max_frames = result[1]
            slider_update = gr.Slider(maximum=max_frames, value=0)
            
            return max_frames, result[2], result[3], result[4], slider_update
                
        except Exception as e:
            logger.error(f"Error in handle_file_change_with_slider_update: {str(e)}", exc_info=True)
            return 0, video_1_path, video_2_path, f"❌ Error: {str(e)}", gr.Slider(maximum=0, value=0)
    
    def _handle_calibration_only_change(
        self, 
        video_1_path: Optional[str], 
        video_2_path: Optional[str], 
        calibration_path: Optional[str]
    ) -> Tuple[int, Optional[str], Optional[str], str, gr.Slider]:
        """Handle calibration-only changes"""
        try:
            upload_messages = []
            
            # Add current video status
            if video_1_path:
                upload_messages.append("✅ Video 1: Valid MP4 file")
            if video_2_path:
                upload_messages.append("✅ Video 2: Valid MP4 file")
            
            # Validate calibration
            if calibration_path:
                is_valid = self.video_service._validate_calibration(calibration_path, upload_messages)
                if not is_valid:
                    return 0, video_1_path, video_2_path, "❌ Calibration validation failed", gr.Slider(maximum=0, value=0)
            else:
                upload_messages.append("⚠️ Calibration: No file uploaded")
            
            # Update calibration path
            self.video_service._current_calibration_path = calibration_path
            
            # Return current video state
            max_frames = self.video_service._calculate_max_frames(video_1_path, video_2_path)
            status_msg = f"✅ Calibration updated | Max frames: {max_frames}"
            slider_update = gr.Slider(maximum=max_frames, value=0)
            
            return max_frames, video_1_path, video_2_path, "\n".join(upload_messages), slider_update
                
        except Exception as e:
            logger.error(f"Error in _handle_calibration_only_change: {str(e)}")
            return 0, video_1_path, video_2_path, f"❌ Error: {str(e)}", gr.Slider(maximum=0, value=0)
    
    def handle_calibration_only_change(
        self, 
        video_1_path: Optional[str], 
        video_2_path: Optional[str], 
        calibration_path: Optional[str]
    ) -> str:
        """Handle calibration-only changes without affecting video displays"""
        try:
            upload_messages = []
            
            # Add current video status
            if video_1_path:
                upload_messages.append("✅ Video 1: Valid MP4 file")
            if video_2_path:
                upload_messages.append("✅ Video 2: Valid MP4 file")
            
            # Validate calibration
            if calibration_path:
                is_valid = self.video_service._validate_calibration(calibration_path, upload_messages)
                if not is_valid:
                    return "❌ Calibration validation failed"
            else:
                upload_messages.append("⚠️ Calibration: No file uploaded")
            
            # Update calibration path
            self.video_service._current_calibration_path = calibration_path
            
            return "\n".join(upload_messages)
                
        except Exception as e:
            logger.error(f"Error in handle_calibration_only_change: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def handle_select_frame_button(self, frame_number: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Handle frame selection for image display"""
        try:
            logger.debug(f"Handling frame selection: frame={frame_number}")
            frame_1_image, frame_2_image, _ = self.video_service.extract_frames_for_selection(frame_number, 0)
            return frame_1_image, frame_2_image
        except Exception as e:
            logger.error(f"Error in handle_select_frame_button: {str(e)}", exc_info=True)
            return None, None
    
    def handle_status_check(self) -> Tuple[str, gr.Button, gr.Button]:
        """Handle status check - reads status file and updates UI accordingly"""
        status_map = {
            "loaded": ("✅ MASt3R model loaded successfully! Ready to process pairs.", True),
            "loading": ("🔄 MASt3R model loading in background... Ready to process pairs once loaded.", False),
            "failed": ("❌ MASt3R model loading failed. Please check logs.", False),
            "error": ("❌ Error loading MASt3R model. Please check logs.", False)
        }
        
        file_status = self._read_status()
        status, model_loaded = status_map.get(file_status, status_map["loading"])
        
        process_button_update = gr.Button(
            interactive=model_loaded,
            variant="primary" if model_loaded else "secondary"
        )
        
        batch_button_update = gr.Button(
            interactive=model_loaded,
            variant="primary" if model_loaded else "secondary"
        )
        
        return status, process_button_update, batch_button_update
    
    def handle_process_pair(
        self, 
        frame_number: int, 
        calibration_file_path: Optional[str],
        render_cameras: bool = True,
        subsample: int = 8
    ) -> Tuple[str, gr.File, gr.Model3D, str]:
        """Handle stereo pair processing"""
        try:
            logger.info(f"Handling stereo pair processing for frame {frame_number}")
            
            # Check if model is loaded
            if self.processing_service.mast3r_engine is None:
                return ("⏳ MASt3R model is still loading... Please wait and try again.", 
                       gr.File(visible=False), gr.Model3D(visible=False), "Model not ready")
            
            if calibration_file_path is None:
                return ("❌ No calibration file uploaded", 
                       gr.File(visible=False), gr.Model3D(visible=False), "No point cloud available")
            
            # Read and parse calibration JSON file
            try:
                with open(calibration_file_path, 'r') as f:
                    calibration_data = json.load(f)
                logger.info(f"Calibration data loaded from: {calibration_file_path}")
            except Exception as e:
                logger.error(f"Error reading calibration file: {str(e)}")
                return (f"❌ Error reading calibration file: {str(e)}", 
                       gr.File(visible=False), gr.Model3D(visible=False), "Error loading calibration")
            
            # Get current frame images
            frame_1_image, frame_2_image, _ = self.video_service.extract_frames_for_selection(frame_number, 0)
            
            if frame_1_image is None or frame_2_image is None:
                return ("❌ No frame images available", 
                       gr.File(visible=False), gr.Model3D(visible=False), "No frame images available")
            
            # Process the stereo pair with SAM if available
            sam_instance = self.sam_service.sam_instance if self.sam_service.sam_instance else None
            point_prompt_1 = self.sam_service.point_prompt_1
            point_prompt_2 = self.sam_service.point_prompt_2
            
            success, status_msg, ply_file_path, download_file_path = self.processing_service.process_stereo_pair(
                frame_1_image, frame_2_image, calibration_data, frame_number, render_cameras, subsample, sam_instance, point_prompt_1, point_prompt_2
            )
            
            if success and ply_file_path:
                # Update download file component (use download_file_path which may be a zip)
                download_file_update = gr.File(value=download_file_path, visible=True)
                
                # Update 3D visualization (use original ply_file_path to find camera pyramids)
                obj_file_path, viz_status = PointCloudVisualizer.update_point_cloud_display(ply_file_path)
                
                # Update Model3D component
                if obj_file_path:
                    model3d_update = gr.Model3D(value=obj_file_path, visible=True)
                else:
                    model3d_update = gr.Model3D(visible=False)
                
                return status_msg, download_file_update, model3d_update, viz_status
            else:
                return (status_msg, gr.File(visible=False), gr.Model3D(visible=False), "Processing failed")
                
        except Exception as e:
            logger.error(f"Error in handle_process_pair: {str(e)}", exc_info=True)
            return (f"❌ Error processing pair: {str(e)}", 
                   gr.File(visible=False), gr.Model3D(visible=False), "Processing error")
    
    def handle_batch_processing(
        self,
        folder_name: str,
        start_frame: int,
        end_frame: int,
        step: int,
        calibration_file_path: Optional[str],
        render_cameras: bool = True,
        subsample: int = 8
    ) -> Tuple[str, str, gr.File]:
        """Handle batch processing of all frames"""
        try:
            logger.info(f"Handling batch processing: folder={folder_name}, frames={start_frame}-{end_frame}, step={step}")
            
            # Check if model is loaded
            if self.processing_service.mast3r_engine is None:
                return ("⏳ MASt3R model is still loading... Please wait and try again.", 
                       "Model not ready", gr.File(visible=False))
            
            if calibration_file_path is None:
                return ("❌ No calibration file uploaded", 
                       "No calibration file", gr.File(visible=False))
            
            # Validate frame range
            if start_frame >= end_frame:
                return ("❌ Start frame must be less than end frame", 
                       "Invalid frame range", gr.File(visible=False))
            
            if step <= 0:
                return ("❌ Step must be greater than 0", 
                       "Invalid step value", gr.File(visible=False))
            
            # Read and parse calibration JSON file
            try:
                with open(calibration_file_path, 'r') as f:
                    calibration_data = json.load(f)
                logger.info(f"Calibration data loaded from: {calibration_file_path}")
            except Exception as e:
                logger.error(f"Error reading calibration file: {str(e)}")
                return (f"❌ Error reading calibration file: {str(e)}", 
                       "Calibration error", gr.File(visible=False))
            
            # Set up batch processing service with the same model
            self.batch_processing_service.mast3r_engine = self.processing_service.mast3r_engine
            
            # Get SAM instance and point prompts if available
            sam_instance = self.sam_service.sam_instance if self.sam_service.sam_instance else None
            point_prompt_1 = self.sam_service.point_prompt_1
            point_prompt_2 = self.sam_service.point_prompt_2
            
            # Start batch processing in a separate thread
            def batch_process_async():
                try:
                    success, status_msg, zip_path = self.batch_processing_service.process_all_frames(
                        self.video_service,
                        calibration_data,
                        folder_name,
                        int(start_frame),
                        int(end_frame),
                        int(step),
                        render_cameras,
                        subsample,
                        sam_instance,
                        point_prompt_1,
                        point_prompt_2
                    )
                    
                    # Update UI components after processing
                    if success and zip_path:
                        # This would need to be handled through a callback or queue system
                        # For now, we'll return the result directly
                        pass
                        
                except Exception as e:
                    logger.error(f"Error in batch processing thread: {str(e)}", exc_info=True)
            
            # Start the batch processing thread
            batch_thread = threading.Thread(target=batch_process_async, daemon=True)
            batch_thread.start()
            
            # Return initial status
            total_frames = len(range(int(start_frame), int(end_frame) + 1, int(step)))
            initial_status = f"🚀 Starting batch processing of {total_frames} frames...\n"
            initial_status += f"📁 Output folder: {folder_name}\n"
            initial_status += f"🎬 Frame range: {start_frame} to {end_frame} (step: {step})\n"
            initial_status += f"📷 Camera rendering: {'Enabled' if render_cameras else 'Disabled'}"
            
            return (initial_status, "Starting batch processing...", gr.File(visible=False))
                
        except Exception as e:
            logger.error(f"Error in handle_batch_processing: {str(e)}", exc_info=True)
            return (f"❌ Error starting batch processing: {str(e)}", 
                   "Processing error", gr.File(visible=False))
    
    def handle_batch_progress_check(self) -> Tuple[str, str, gr.File]:
        """Check batch processing progress and update UI"""
        try:
            if not self.batch_processing_service.is_processing:
                # Processing completed, check for results
                if self.batch_processing_service.output_dir:
                    # Look for zip file
                    zip_files = list(self.batch_processing_service.output_dir.glob("*.zip"))
                    if zip_files:
                        zip_path = zip_files[0]
                        status_msg = f"✅ Batch processing completed!\n"
                        status_msg += f"📦 Results ready for download: {zip_path.name}"
                        
                        return (status_msg, "Processing completed", gr.File(value=str(zip_path), visible=True))
                    else:
                        return ("❌ Batch processing completed but no results found", 
                               "No results found", gr.File(visible=False))
                else:
                    return ("Ready to process frames...", 
                           "Ready", gr.File(visible=False))
            else:
                # Still processing
                current, total = self.batch_processing_service.get_progress()
                progress_value = current / total if total > 0 else 0
                
                status_msg = f"🔄 Processing frame {current + 1} of {total}...\n"
                status_msg += f"Progress: {progress_value:.1%}"
                
                return (status_msg, f"Progress: {progress_value:.1%}", gr.File(visible=False))
                
        except Exception as e:
            logger.error(f"Error in handle_batch_progress_check: {str(e)}", exc_info=True)
            return (f"❌ Error checking progress: {str(e)}", 
                   "Error", gr.File(visible=False))
    
    def handle_activate_sam(self) -> Tuple[str, gr.Button, gr.Button, gr.Textbox, gr.Textbox, gr.Button]:
        """Handle SAM activation"""
        try:
            logger.info("Activating SAM model...")
            success, status_msg = self.sam_service.activate_sam()
            
            if success:
                # Update UI components
                activate_btn = gr.Button(interactive=False)
                deactivate_btn = gr.Button(interactive=True)
                point_prompt_1 = gr.Textbox(interactive=True)
                point_prompt_2 = gr.Textbox(interactive=True)
                compute_masks_btn = gr.Button(interactive=True)
            else:
                # Keep current state on failure
                activate_btn = gr.Button(interactive=True)
                deactivate_btn = gr.Button(interactive=False)
                point_prompt_1 = gr.Textbox(interactive=False)
                point_prompt_2 = gr.Textbox(interactive=False)
                compute_masks_btn = gr.Button(interactive=False)
            
            return status_msg, activate_btn, deactivate_btn, point_prompt_1, point_prompt_2, compute_masks_btn
            
        except Exception as e:
            logger.error(f"Error activating SAM: {str(e)}", exc_info=True)
            return f"❌ Error activating SAM: {str(e)}", gr.Button(), gr.Button(), gr.Textbox(), gr.Textbox(), gr.Button()
    
    def handle_deactivate_sam(self, frame_number: int) -> Tuple[str, gr.Button, gr.Button, gr.Textbox, gr.Textbox, gr.Button, Optional[np.ndarray], Optional[np.ndarray]]:
        """Handle SAM deactivation and restore raw images"""
        try:
            logger.info("Deactivating SAM model...")
            success, status_msg = self.sam_service.deactivate_sam()
            
            # Update UI components
            activate_btn = gr.Button(interactive=True)
            deactivate_btn = gr.Button(interactive=False)
            point_prompt_1 = gr.Textbox(value="", interactive=False)
            point_prompt_2 = gr.Textbox(value="", interactive=False)
            compute_masks_btn = gr.Button(interactive=False)
            
            # Restore raw images (without SAM masks)
            frame_1_image, frame_2_image, _ = self.video_service.extract_frames_for_selection(frame_number, 0)
            
            return status_msg, activate_btn, deactivate_btn, point_prompt_1, point_prompt_2, compute_masks_btn, frame_1_image, frame_2_image
            
        except Exception as e:
            logger.error(f"Error deactivating SAM: {str(e)}", exc_info=True)
            return f"❌ Error deactivating SAM: {str(e)}", gr.Button(), gr.Button(), gr.Textbox(), gr.Textbox(), gr.Button(), None, None
    
    def handle_compute_masks(self, frame_number: int, point_prompt_1: str, point_prompt_2: str) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], str]:
        """Handle mask computation"""
        try:
            logger.info(f"Computing masks for frame {frame_number}")
            
            # Set point prompts
            success, status_msg = self.sam_service.set_point_prompts(point_prompt_1, point_prompt_2)
            if not success:
                return status_msg, None, None, self.sam_service.get_sam_status()
            
            # Get current frame images
            frame_1_image, frame_2_image, _ = self.video_service.extract_frames_for_selection(frame_number, 0)
            
            if frame_1_image is None and frame_2_image is None:
                return "❌ No frame images available", None, None, self.sam_service.get_sam_status()
            
            # Compute masks
            success, status_msg, rendered_image_1, rendered_image_2 = self.sam_service.compute_masks(frame_1_image, frame_2_image)
            
            return status_msg, rendered_image_1, rendered_image_2, self.sam_service.get_sam_status()
            
        except Exception as e:
            logger.error(f"Error computing masks: {str(e)}", exc_info=True)
            return f"❌ Error computing masks: {str(e)}", None, None, self.sam_service.get_sam_status()
    
    def setup_event_handlers(self, components: Dict[str, Any]) -> None:
        """Setup all event handlers for the UI components"""
        try:
            logger.info("Setting up event handlers")
            
            # File change events
            for video_upload in ['video_1_upload', 'video_2_upload']:
                components[video_upload].change(
                    fn=self.handle_file_change_with_slider_update,
                    inputs=[components['video_1_upload'], components['video_2_upload'], components['calibration_upload']],
                    outputs=[components['frame_slider'], components['video_1_display'], components['video_2_display'], 
                            components['upload_status'], components['frame_slider']]
                ).then(
                    fn=self.handle_status_check,
                    inputs=[],
                    outputs=[components['processing_status'], components['process_pair_btn'], components['process_all_btn']]
                )
            
            # Calibration upload change
            components['calibration_upload'].change(
                fn=self.handle_calibration_only_change,
                inputs=[components['video_1_upload'], components['video_2_upload'], components['calibration_upload']],
                outputs=[components['upload_status']]
            ).then(
                fn=self.handle_status_check,
                inputs=[],
                outputs=[components['processing_status'], components['process_pair_btn'], components['process_all_btn']]
            )
            
            # Frame slider change
            components['frame_slider'].change(
                fn=self.handle_select_frame_button,
                inputs=[components['frame_slider']],
                outputs=[components['selected_image_1'], components['selected_image_2']]
            ).then(
                fn=self.handle_status_check,
                inputs=[],
                outputs=[components['processing_status'], components['process_pair_btn'], components['process_all_btn']]
            )
            
            # Process pair button
            components['process_pair_btn'].click(
                fn=self.handle_process_pair,
                inputs=[components['frame_slider'], components['calibration_upload'], components['render_camera_toggle'], components['subsample_param']],
                outputs=[components['processing_status'], components['download_ply'], 
                        components['model3d_viewer'], components['viz_status']]
            )
            
            # Update status button
            components['update_status_btn'].click(
                fn=self.handle_status_check,
                inputs=[],
                outputs=[components['processing_status'], components['process_pair_btn'], components['process_all_btn']]
            )
            
            # Batch processing button
            components['process_all_btn'].click(
                fn=self.handle_batch_processing,
                inputs=[components['folder_name'], components['start_frame'], components['end_frame'], 
                       components['step'], components['calibration_upload'], components['render_camera_toggle'], components['subsample_param']],
                outputs=[components['batch_status'], components['batch_progress'], components['download_batch']]
            )
            
            # Update batch status button
            components['update_batch_status_btn'].click(
                fn=self.handle_batch_progress_check,
                inputs=[],
                outputs=[components['batch_status'], components['batch_progress'], components['download_batch']]
            )
            
            # SAM activation button
            components['activate_sam_btn'].click(
                fn=self.handle_activate_sam,
                inputs=[],
                outputs=[components['sam_status'], components['activate_sam_btn'], components['deactivate_sam_btn'], 
                        components['point_prompt_1'], components['point_prompt_2'], components['compute_masks_btn']]
            )
            
            # SAM deactivation button
            components['deactivate_sam_btn'].click(
                fn=self.handle_deactivate_sam,
                inputs=[components['frame_slider']],
                outputs=[components['sam_status'], components['activate_sam_btn'], components['deactivate_sam_btn'], 
                        components['point_prompt_1'], components['point_prompt_2'], components['compute_masks_btn'],
                        components['selected_image_1'], components['selected_image_2']]
            )
            
            # Compute masks button
            components['compute_masks_btn'].click(
                fn=self.handle_compute_masks,
                inputs=[components['frame_slider'], components['point_prompt_1'], components['point_prompt_2']],
                outputs=[components['sam_status'], components['selected_image_1'], components['selected_image_2'], components['sam_status']]
            )
            
            
            logger.info("Event handlers setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up event handlers: {str(e)}", exc_info=True)
            raise