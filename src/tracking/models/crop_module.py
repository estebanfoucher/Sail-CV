"""Base class for crop analysis modules."""

from abc import ABC, abstractmethod

import numpy as np

from .bounding_box import BoundingBox
from .image import Image


class CropModule(ABC):
    """
    Abstract base class for analyzing image crops.

    Subclasses must implement analyze_crop to process crops extracted from
    bounding boxes and return analysis results as numpy arrays.
    """

    @abstractmethod
    def analyze_crop(self, image: Image, bboxes: list[BoundingBox]) -> list[np.ndarray]:
        """
        Analyze crops from an image based on bounding boxes.

        Args:
            image: Image object containing the image data
            bboxes: List of bounding boxes defining crop regions

        Returns:
            List of numpy arrays, one per bounding box. Each array contains
            analysis results for the corresponding crop. The length of the
            returned list must equal the length of bboxes.
        """
        pass
