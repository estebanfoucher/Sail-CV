"""VPI-based background subtraction for foreground mask generation."""

import cv2
import numpy as np
from loguru import logger

try:
    import vpi
except ImportError:
    vpi = None
    logger.warning("VPI not available. BackgroundDetectorVPI will not work.")

from models.background_detector import BackgroundDetector


class BackgroundDetectorVPI(BackgroundDetector):
    """
    Background detector using NVIDIA VPI BackgroundSubtractor.

    Generates foreground (movement) masks by subtracting background from frames.
    """

    def __init__(self, image_size: tuple[int, int], backend: str = "cuda", learn_rate: float = 0.01):
        """
        Initialize VPI BackgroundSubtractor.

        Args:
            image_size: Tuple of (width, height) for image dimensions
            backend: Backend to use ("cuda" or "cpu")
            learn_rate: Learning rate for background model (0.0 to 1.0)
        """
        if vpi is None:
            raise ImportError("VPI is not installed. Cannot use BackgroundDetectorVPI.")
        
        self.image_size = image_size
        self.learn_rate = learn_rate
        
        # Select backend
        if backend == "cuda":
            self.backend = vpi.Backend.CUDA
        else:
            self.backend = vpi.Backend.CPU
        
        # Initialize background subtractor
        with self.backend:
            self.bgsub = vpi.BackgroundSubtractor(image_size, vpi.Format.BGR8)
        
        logger.info(f"Initialized BackgroundDetectorVPI with size {image_size}, backend={backend}, learn_rate={learn_rate}")

    def generate_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate binary foreground mask (movement mask) for a full image.

        Args:
            image: Full image as numpy array of shape (H, W, 3) in BGR format

        Returns:
            Binary foreground mask as numpy array of shape (H, W) with values 0/1.
        """
        h, w = image.shape[:2]
        
        # Verify image size matches expected size
        if (w, h) != self.image_size:
            logger.warning(
                f"Image size {(w, h)} doesn't match expected {self.image_size}, "
                f"resizing image"
            )
            image = cv2.resize(image, self.image_size)
        
        try:
            # Convert numpy array to VPI image
            vpi_image = vpi.asimage(image, vpi.Format.BGR8)
            
            # Get foreground mask and background image
            with self.backend:
                fgmask, _ = self.bgsub(vpi_image, learnrate=self.learn_rate)
                
                # Convert foreground mask to BGR8 format for CPU access
                fgmask_bgr = fgmask.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)
                
                # Copy to CPU as numpy array
                with fgmask_bgr.rlock_cpu():
                    mask_cpu = fgmask_bgr.cpu()
            
            # Convert to grayscale if needed (take first channel)
            if len(mask_cpu.shape) == 3:
                mask = mask_cpu[:, :, 0]
            else:
                mask = mask_cpu
            
            # Normalize to binary (0/1)
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)  # Threshold at 127 for 0-255 range
            else:
                mask = (mask > 0.5).astype(np.uint8)  # Already 0-1 range
            
            return mask
            
        except Exception as e:
            logger.error(f"Error generating foreground mask: {e}")
            # Return zero mask as fallback
            return np.zeros((h, w), dtype=np.uint8)
