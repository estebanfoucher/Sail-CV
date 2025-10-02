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

# Import from src/ as requested
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from process_pairs import instantiate_mast3r_engine, process_pair
from stereo.convert_calibration import convert_calibration_parameters
from stereo.image import preprocess_image, resize_image

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
        frame_number: int
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Process a stereo pair with MASt3R
        
        Args:
            image_1: First image as numpy array
            image_2: Second image as numpy array  
            calibration_data: Calibration parameters
            frame_number: Frame number for naming
            
        Returns:
            tuple: (success, status_message, ply_file_path)
        """
        try:
            if self.mast3r_engine is None:
                return False, "❌ MASt3R model not loaded. Please load the model first.", None
            if self.output_dir is None:
                return False, "❌ Output directory not initialized.", None
            
            logger.info(f"Processing stereo pair for frame {frame_number}")
            
            # Convert calibration parameters
            calibration_params = convert_calibration_parameters(
                calibration_data, original_size=(1920, 1080)
            )
            
            # Convert numpy arrays to PIL Images
            if image_1 is not None:
                image_1_pil = Image.fromarray(image_1)
            else:
                return False, "❌ No image 1 available", None
                
            if image_2 is not None:
                image_2_pil = Image.fromarray(image_2)
            else:
                return False, "❌ No image 2 available", None
            
            # Process the pair (without SAM for now, as requested)
            pair_name = f"frame_{frame_number}"
            
            process_pair(
                image_1_pil,
                image_2_pil,
                self.mast3r_engine,
                sam=None,  # No SAM for now
                calibration_params=calibration_params,
                pair_name=pair_name,
                render_cameras=False,
                output_folder=self.output_dir,
                point_prompt_1=None,  # No point prompts for now
                point_prompt_2=None
            )
            
            # Check if PLY file was created
            ply_file_path = self.output_dir / f"point_cloud_{pair_name}.ply"
            if ply_file_path.exists():
                logger.info(f"Point cloud generated successfully: {ply_file_path}")
                return True, f"✅ Point cloud generated successfully for frame {frame_number}!", str(ply_file_path)
            else:
                return False, "❌ Point cloud file was not generated", None
                
        except Exception as e:
            logger.error(f"Error processing stereo pair: {str(e)}", exc_info=True)
            return False, f"❌ Error processing stereo pair: {str(e)}", None
    
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
