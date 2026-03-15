"""
Batch processing service for MVS web application
"""

import logging
import tempfile
import os
import zipfile
import threading
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np
from PIL import Image

# Add reconstruction source to path for flat imports (process_pairs, stereo, etc.)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "reconstruction"))

from process_pairs import instantiate_mast3r_engine, process_pair
from stereo.convert_calibration import convert_calibration_parameters
from stereo.image import preprocess_image, resize_image

# Get project root for permanent output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEB_APP_OUTPUT_DIR = PROJECT_ROOT / "output" / "web_app"

logger = logging.getLogger(__name__)


class BatchProcessingService:
    """Service class for handling batch processing operations"""

    def __init__(self):
        self.mast3r_engine = None
        self.output_dir = None
        self.is_processing = False
        self.current_progress = 0
        self.total_frames = 0
        self.progress_callback = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress and call callback if set"""
        self.current_progress = current
        self.total_frames = total
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def process_all_frames(
        self,
        video_service,
        calibration_data: dict,
        folder_name: str,
        start_frame: int,
        end_frame: int,
        step: int,
        render_cameras: bool = True,
        subsample: int = 8,
        sam_instance = None,
        point_prompt_1 = None,
        point_prompt_2 = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Process all frames in the specified range

        Args:
            video_service: Video service instance for frame extraction
            calibration_data: Calibration parameters
            folder_name: Name of the output folder
            start_frame: Starting frame number
            end_frame: Ending frame number
            step: Step size between frames
            render_cameras: Whether to render camera pyramids
            subsample: Subsample parameter for point density control
            sam_instance: SAM instance for filtering (optional)
            point_prompt_1: Point prompt for image 1 (optional)
            point_prompt_2: Point prompt for image 2 (optional)

        Returns:
            tuple: (success, status_message, zip_file_path)
        """
        try:
            if self.mast3r_engine is None:
                return False, "❌ MASt3R model not loaded. Please load the model first.", None

            if self.is_processing:
                return False, "❌ Batch processing is already in progress.", None

            self.is_processing = True

            # Create output directory structure
            self.output_dir = Path(tempfile.mkdtemp(prefix=f"mvs_batch_{folder_name}_"))
            logger.info(f"Created batch output directory: {self.output_dir}")

            # Create folder structure
            main_folder = self.output_dir / folder_name
            main_folder.mkdir(exist_ok=True)

            camera_1_folder = main_folder / "camera_1"
            camera_2_folder = main_folder / "camera_2"
            camera_1_folder.mkdir(exist_ok=True)
            camera_2_folder.mkdir(exist_ok=True)

            # Calculate total frames to process
            frame_numbers = list(range(start_frame, end_frame + 1, step))
            total_frames = len(frame_numbers)

            # Log SAM filtering status
            sam_status = "with SAM filtering" if sam_instance is not None else "without SAM filtering"
            logger.info(f"Starting batch processing of {total_frames} frames {sam_status}")
            if sam_instance is not None:
                logger.info(f"SAM point prompts: Image 1: {point_prompt_1}, Image 2: {point_prompt_2}")

            self._update_progress(0, total_frames, f"Starting batch processing of {total_frames} frames {sam_status}...")

            # Convert calibration parameters (uses image_size from calibration_data if available)
            calibration_params = convert_calibration_parameters(calibration_data)

            processed_frames = 0
            failed_frames = 0

            for i, frame_number in enumerate(frame_numbers):
                try:
                    self._update_progress(
                        i, total_frames,
                        f"Processing frame {frame_number} ({i+1}/{total_frames})..."
                    )

                    # Extract frames
                    frame_1_image, frame_2_image, _ = video_service.extract_frames_for_selection(frame_number, 0)

                    if frame_1_image is None or frame_2_image is None:
                        logger.warning(f"Could not extract frames for frame {frame_number}")
                        failed_frames += 1
                        continue

                    # Convert numpy arrays to PIL Images
                    image_1_pil = Image.fromarray(frame_1_image)
                    image_2_pil = Image.fromarray(frame_2_image)

                    # Format frame number with leading zeros
                    frame_name = f"{frame_number:04d}"

                    # Process the pair
                    pair_name = f"frame_{frame_name}"

                    # Create permanent output directory for match renders
                    match_render_output_dir = WEB_APP_OUTPUT_DIR / "match_renders" / folder_name
                    match_render_output_dir.mkdir(parents=True, exist_ok=True)

                    process_pair(
                        image_1_pil,
                        image_2_pil,
                        self.mast3r_engine,
                        sam=sam_instance,  # Use SAM if available
                        calibration_params=calibration_params,
                        pair_name=pair_name,
                        render_cameras=render_cameras,
                        output_folder=main_folder,  # Output directly to main folder
                        point_prompt_1=point_prompt_1,  # Use point prompts if available
                        point_prompt_2=point_prompt_2,
                        save_resized_frames=False,  # Don't save resized frames for batch processing
                        save_obj_file=False,  # Don't save obj files for batch processing
                        subsample=subsample,
                        save_match_render=True,  # Save match correspondences visualization
                        match_render_output_folder=str(match_render_output_dir),  # Use permanent location
                    )

                    # Move and rename files according to structure
                    self._organize_output_files(main_folder, frame_name, render_cameras)

                    processed_frames += 1

                except Exception as e:
                    logger.error(f"Error processing frame {frame_number}: {str(e)}")
                    failed_frames += 1
                    continue

            # Create zip file
            zip_path = self._create_zip_file(folder_name)

            self._update_progress(
                total_frames, total_frames,
                f"Batch processing completed! Processed: {processed_frames}, Failed: {failed_frames}"
            )

            self.is_processing = False

            if processed_frames > 0:
                status_msg = f"✅ Batch processing completed!\n"
                status_msg += f"📊 Processed: {processed_frames} frames\n"
                status_msg += f"❌ Failed: {failed_frames} frames\n"
                status_msg += f"📁 Output folder: {folder_name}\n"
                status_msg += f"📦 Zip file ready for download"

                return True, status_msg, str(zip_path)
            else:
                return False, "❌ No frames were processed successfully.", None

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
            self.is_processing = False
            return False, f"❌ Error in batch processing: {str(e)}", None

    def _organize_output_files(self, main_folder: Path, frame_name: str, render_cameras: bool):
        """Organize output files according to the required structure"""
        try:
            # Look for the generated PLY file
            ply_file = main_folder / f"point_cloud_frame_{frame_name}.ply"

            if ply_file.exists():
                # Move to main folder with correct name
                new_ply_path = main_folder / f"frame_{frame_name}.ply"
                ply_file.rename(new_ply_path)
                logger.info(f"Moved PLY file to: {new_ply_path}")

            # Handle camera pyramids if enabled
            if render_cameras:
                camera_pyramid_dir = main_folder / f"camera_pyramids_frame_{frame_name}"
                if camera_pyramid_dir.exists():
                    # Move camera pyramid files to camera folders
                    for camera_file in camera_pyramid_dir.glob("camera_*_pyramid.ply"):
                        if "camera_1" in camera_file.name:
                            new_path = main_folder / "camera_1" / f"camera_{frame_name}.ply"
                        elif "camera_2" in camera_file.name:
                            new_path = main_folder / "camera_2" / f"camera_{frame_name}.ply"
                        else:
                            continue

                        camera_file.rename(new_path)
                        logger.info(f"Moved camera pyramid to: {new_path}")

                    # Remove empty camera pyramid directory
                    camera_pyramid_dir.rmdir()

        except Exception as e:
            logger.error(f"Error organizing output files for frame {frame_name}: {str(e)}")

    def _create_zip_file(self, folder_name: str) -> Path:
        """Create zip file of the entire output folder"""
        try:
            zip_path = self.output_dir / f"{folder_name}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the main folder
                main_folder = self.output_dir / folder_name
                for file_path in main_folder.rglob('*'):
                    if file_path.is_file():
                        # Add file to zip with relative path from main folder
                        arcname = file_path.relative_to(main_folder)
                        zipf.write(file_path, arcname)
                        logger.info(f"Added to zip: {arcname}")

            logger.info(f"Created zip file: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Error creating zip file: {str(e)}")
            raise

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.output_dir and self.output_dir.exists():
                import shutil
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up batch temp directory: {self.output_dir}")
                self.output_dir = None
        except Exception as e:
            logger.error(f"Error cleaning up batch temp files: {str(e)}")

    def get_progress(self) -> Tuple[int, int]:
        """Get current progress"""
        return self.current_progress, self.total_frames
