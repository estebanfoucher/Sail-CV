"""Base class for background detection modules."""

from abc import ABC, abstractmethod

import numpy as np


class BackgroundDetector(ABC):
    """
    Abstract base class for generating foreground (movement) masks from images.

    Subclasses must implement generate_foreground_mask to create binary masks
    that indicate moving/foreground regions in the image.
    """

    @abstractmethod
    def generate_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate binary foreground mask (movement mask) for a full image.

        Args:
            image: Full image as numpy array of shape (H, W, 3) or (H, W)

        Returns:
            Binary foreground mask as numpy array of shape (H, W) with values 0/1 or 0/255.
            Pixels with value 1 (or 255) indicate foreground/movement regions.
        """
        pass
