"""
SAM (Segment Anything Model) service for MVS web application
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2

# Add reconstruction source to path for flat imports (unitaries, etc.)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "reconstruction"))

from unitaries.sam import SAM

logger = logging.getLogger(__name__)


class SAMService:
    """Service class for handling SAM operations"""

    def __init__(self):
        self.sam_instance = None
        self.sam_model_path = None
        self.output_dir = None
        self.point_prompt_1 = None
        self.point_prompt_2 = None
        self.rendered_image_1 = None
        self.rendered_image_2 = None

    def activate_sam(self, model_path: str = None) -> Tuple[bool, str]:
        """
        Activate SAM model

        Args:
            model_path: Path to SAM model (optional, will use default if not provided)

        Returns:
            tuple: (success, status_message)
        """
        try:
            logger.info("Activating SAM model...")

            # Use default model path if not provided
            if model_path is None:
                project_root = Path(__file__).parent.parent.parent
                model_path = str(project_root / "checkpoints" / "FastSAM-x.pt")

            self.sam_model_path = model_path

            # Create temporary output directory for SAM operations
            self.output_dir = Path(tempfile.mkdtemp(prefix="sam_processing_"))
            logger.info(f"Created temp SAM output directory: {self.output_dir}")

            # Initialize SAM model
            self.sam_instance = SAM(model_path)

            logger.info("SAM model activated successfully")
            return True, "✅ SAM model activated successfully! Ready for point prompts."

        except Exception as e:
            logger.error(f"Error activating SAM model: {str(e)}", exc_info=True)
            return False, f"❌ Error activating SAM model: {str(e)}"

    def deactivate_sam(self) -> Tuple[bool, str]:
        """
        Deactivate SAM model and reset state

        Returns:
            tuple: (success, status_message)
        """
        try:
            logger.info("Deactivating SAM model...")

            # Reset SAM instance
            self.sam_instance = None
            self.sam_model_path = None
            self.point_prompt_1 = None
            self.point_prompt_2 = None
            self.rendered_image_1 = None
            self.rendered_image_2 = None

            # Clean up temporary files
            if self.output_dir and self.output_dir.exists():
                import shutil
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up SAM temp directory: {self.output_dir}")
                self.output_dir = None

            logger.info("SAM model deactivated successfully")
            return True, "🔴 SAM Filter: Inactive"

        except Exception as e:
            logger.error(f"Error deactivating SAM model: {str(e)}", exc_info=True)
            return False, f"❌ Error deactivating SAM model: {str(e)}"

    def set_point_prompts(self, prompt_1: str, prompt_2: str) -> Tuple[bool, str]:
        """
        Set point prompts for both images

        Args:
            prompt_1: Point prompt for image 1 (format: "x,y")
            prompt_2: Point prompt for image 2 (format: "x,y")

        Returns:
            tuple: (success, status_message)
        """
        try:
            if self.sam_instance is None:
                return False, "❌ SAM model not activated. Please activate SAM first."

            # Parse point prompts
            try:
                if prompt_1.strip():
                    x1, y1 = map(int, prompt_1.strip().split(','))
                    self.point_prompt_1 = (x1, y1)
                else:
                    self.point_prompt_1 = None

                if prompt_2.strip():
                    x2, y2 = map(int, prompt_2.strip().split(','))
                    self.point_prompt_2 = (x2, y2)
                else:
                    self.point_prompt_2 = None

            except ValueError:
                return False, "❌ Invalid point prompt format. Use 'x,y' format (e.g., '256,128')"

            logger.info(f"Point prompts set: Image 1: {self.point_prompt_1}, Image 2: {self.point_prompt_2}")
            return True, "✅ Point prompts set successfully"

        except Exception as e:
            logger.error(f"Error setting point prompts: {str(e)}", exc_info=True)
            return False, f"❌ Error setting point prompts: {str(e)}"

    def compute_masks(self, image_1: np.ndarray, image_2: np.ndarray) -> Tuple[bool, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute masks for both images using SAM and point prompts

        Args:
            image_1: First image as numpy array
            image_2: Second image as numpy array

        Returns:
            tuple: (success, status_message, rendered_image_1, rendered_image_2)
        """
        try:
            if self.sam_instance is None:
                return False, "❌ SAM model not activated. Please activate SAM first.", None, None

            if self.point_prompt_1 is None and self.point_prompt_2 is None:
                return False, "❌ No point prompts set. Please set point prompts first.", None, None

            logger.info("Computing masks with SAM...")

            rendered_image_1 = None
            rendered_image_2 = None

            # Process image 1 if point prompt is set
            if self.point_prompt_1 is not None and image_1 is not None:
                logger.info(f"Processing image 1 with point prompt: {self.point_prompt_1}")
                mask_result_1 = self.sam_instance.predict(image_1, point=self.point_prompt_1)
                if mask_result_1["mask"] is not None:
                    rendered_image_1 = self.sam_instance.render_result(
                        image_1, mask_result_1["mask"], [mask_result_1["point"]]
                    )
                    self.rendered_image_1 = rendered_image_1
                    logger.info("Image 1 mask computed successfully")
                else:
                    logger.warning("No mask found for image 1")

            # Process image 2 if point prompt is set
            if self.point_prompt_2 is not None and image_2 is not None:
                logger.info(f"Processing image 2 with point prompt: {self.point_prompt_2}")
                mask_result_2 = self.sam_instance.predict(image_2, point=self.point_prompt_2)
                if mask_result_2["mask"] is not None:
                    rendered_image_2 = self.sam_instance.render_result(
                        image_2, mask_result_2["mask"], [mask_result_2["point"]]
                    )
                    self.rendered_image_2 = rendered_image_2
                    logger.info("Image 2 mask computed successfully")
                else:
                    logger.warning("No mask found for image 2")

            status_msg = "✅ Masks computed successfully!"
            if rendered_image_1 is not None:
                status_msg += " Image 1: Mask applied."
            if rendered_image_2 is not None:
                status_msg += " Image 2: Mask applied."

            return True, status_msg, rendered_image_1, rendered_image_2

        except Exception as e:
            logger.error(f"Error computing masks: {str(e)}", exc_info=True)
            return False, f"❌ Error computing masks: {str(e)}", None, None

    def get_sam_status(self) -> str:
        """Get current SAM status"""
        if self.sam_instance is None:
            return "🔴 SAM Filter: Inactive"
        else:
            status = "🟢 SAM Filter: Active"
            if self.point_prompt_1 is not None:
                status += f" | Image 1: {self.point_prompt_1}"
            if self.point_prompt_2 is not None:
                status += f" | Image 2: {self.point_prompt_2}"
            return status

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.output_dir and self.output_dir.exists():
                import shutil
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up SAM temp directory: {self.output_dir}")
                self.output_dir = None
        except Exception as e:
            logger.error(f"Error cleaning up SAM temp files: {str(e)}")
