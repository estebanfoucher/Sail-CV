"""
Processing service for MVS web application
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
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


class ProcessingService:
    """Service class for handling MASt3R processing operations"""

    def __init__(self):
        self.mast3r_engine = None
        self.output_dir = None

    def load_mast3r_model(self) -> Tuple[bool, str]:
        """
        Load MASt3R model

        Returns:
            tuple: (success, status_message)
        """
        try:
            logger.info("Loading MASt3R model...")

            # Create temporary output directory
            self.output_dir = Path(tempfile.mkdtemp(prefix="mvs_processing_"))
            logger.info(f"Created temp output directory: {self.output_dir}")

            # Load MASt3R engine
            self.mast3r_engine = instantiate_mast3r_engine()

            logger.info("MASt3R model loaded successfully")
            return True, "✅ MASt3R model loaded successfully! Ready to process pairs."

        except Exception as e:
            logger.error(f"Error loading MASt3R model: {str(e)}", exc_info=True)
            return False, f"❌ Error loading MASt3R model: {str(e)}"

    def process_stereo_pair(
        self,
        image_1: np.ndarray,
        image_2: np.ndarray,
        calibration_data: dict,
        frame_number: int,
        render_cameras: bool = True,
        subsample: int = 8,
        sam_instance = None,
        point_prompt_1 = None,
        point_prompt_2 = None
    ) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Process a stereo pair with MASt3R

        Args:
            image_1: First image as numpy array
            image_2: Second image as numpy array
            calibration_data: Calibration parameters
            frame_number: Frame number for naming
            render_cameras: Whether to render camera pyramids
            subsample: Subsample parameter for point density control
            sam_instance: SAM instance for filtering (optional)
            point_prompt_1: Point prompt for image 1 (optional)
            point_prompt_2: Point prompt for image 2 (optional)

        Returns:
            tuple: (success, status_message, ply_file_path)
        """
        try:
            if self.mast3r_engine is None:
                return False, "❌ MASt3R model not loaded. Please load the model first.", None
            if self.output_dir is None:
                return False, "❌ Output directory not initialized.", None

            logger.info(f"Processing stereo pair for frame {frame_number}")

            # Convert calibration parameters (uses image_size from calibration_data if available)
            calibration_params = convert_calibration_parameters(calibration_data)

            # Convert numpy arrays to PIL Images
            if image_1 is not None:
                image_1_pil = Image.fromarray(image_1)
            else:
                return False, "❌ No image 1 available", None

            if image_2 is not None:
                image_2_pil = Image.fromarray(image_2)
            else:
                return False, "❌ No image 2 available", None

            # Process the pair with SAM if available
            pair_name = f"frame_{frame_number}"

            logger.info(f"Processing pair with render_cameras={render_cameras}")

            # Process the pair with SAM if available
            logger.info(f"Processing with SAM: {sam_instance is not None}, Point prompts: {point_prompt_1}, {point_prompt_2}")

            # Create permanent output directory for match renders
            match_render_output_dir = WEB_APP_OUTPUT_DIR / "match_renders"
            match_render_output_dir.mkdir(parents=True, exist_ok=True)

            process_pair(
                image_1_pil,
                image_2_pil,
                self.mast3r_engine,
                sam=sam_instance,
                calibration_params=calibration_params,
                pair_name=pair_name,
                render_cameras=render_cameras,
                output_folder=self.output_dir,
                point_prompt_1=point_prompt_1,
                point_prompt_2=point_prompt_2,
                save_resized_frames=True,  # Save resized frames for single frame processing
                save_obj_file=True,  # Save obj files for 3D rendering in single frame processing
                subsample=subsample,
                save_match_render=True,  # Save match correspondences visualization
                match_render_output_folder=str(match_render_output_dir),  # Use permanent location
            )

            # Check if PLY file was created
            ply_file_path = self.output_dir / f"point_cloud_{pair_name}.ply"
            if ply_file_path.exists():
                logger.info(f"Point cloud generated successfully: {ply_file_path}")

                # Collect all PLY files for download (point cloud + camera pyramids if enabled)
                all_ply_files = [str(ply_file_path)]
                if render_cameras:
                    # Look for camera pyramid PLY files in the subdirectory
                    camera_pyramid_dir = self.output_dir / f"camera_pyramids_{pair_name}"
                    if camera_pyramid_dir.exists():
                        for file in camera_pyramid_dir.glob("camera_*_pyramid.ply"):
                            all_ply_files.append(str(file))
                            logger.info(f"Found camera pyramid PLY file: {file}")

                # Create a zip file with all PLY files if there are multiple files
                if len(all_ply_files) > 1:
                    import zipfile
                    zip_path = self.output_dir / f"point_cloud_and_cameras_{pair_name}.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for ply_file in all_ply_files:
                            # Add file to zip with just the filename (not full path)
                            zipf.write(ply_file, os.path.basename(ply_file))
                    logger.info(f"Created zip file with all PLY files: {zip_path}")
                    download_file_path = str(zip_path)
                else:
                    download_file_path = str(ply_file_path)

                status_msg = f"✅ Point cloud generated successfully for frame {frame_number}!"
                if render_cameras and len(all_ply_files) > 1:
                    status_msg += f" ({len(all_ply_files)-1} camera pyramids included)"

                return True, status_msg, str(ply_file_path), download_file_path
            else:
                return False, "❌ Point cloud file was not generated", None, None

        except Exception as e:
            logger.error(f"Error processing stereo pair: {str(e)}", exc_info=True)
            return False, f"❌ Error processing stereo pair: {str(e)}", None, None

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.output_dir and self.output_dir.exists():
                import shutil
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up temp directory: {self.output_dir}")
                self.output_dir = None
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
