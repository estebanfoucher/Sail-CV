"""Base class for mask detection modules."""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class MaskDetector(ABC):
    """
    Abstract base class for generating masks from image crops.

    Subclasses must implement generate_mask to create binary masks
    that can be used to focus PCA analysis on object regions.
    """

    @abstractmethod
    def generate_masks(self, image: np.ndarray, bboxes: list) -> list[np.ndarray]:
        """
        Generate binary masks for multiple bounding boxes on a full image.

        Args:
            image: Full image as numpy array of shape (H, W, 3) or (H, W)
            bboxes: List of bounding boxes in XYXY format [x1, y1, x2, y2]

        Returns:
            List of binary masks, one per bounding box. Each mask is a numpy array
            of shape (H, W) with values 0/1 or 0/255. Pixels with value 1 (or 255)
            indicate the object region.
        """
        pass

    def generate_mask(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate binary mask for a crop (legacy method for backward compatibility).

        Args:
            crop: Numpy array of shape (H, W, 3) or (H, W) representing the crop

        Returns:
            Binary mask as numpy array of shape (H, W) with values 0/1 or 0/255.
        """
        # Default implementation: treat crop as full image with single bbox
        h, w = crop.shape[:2]
        bbox = [[0, 0, w, h]]
        masks = self.generate_masks(crop, bbox)
        return masks[0] if masks else np.ones((h, w), dtype=np.uint8)

    def render_masks(
        self, image: np.ndarray, masks: list[np.ndarray], alpha: float = 0.5
    ) -> np.ndarray:
        """
        Render multiple mask overlays on full image for visualization.

        Args:
            image: Full image
            masks: List of binary masks from generate_masks
            alpha: Transparency of mask overlay (0.0 to 1.0)

        Returns:
            Image with mask overlays applied
        """
        # Ensure image is 3-channel
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_rgb = image.copy()

        result = image_rgb.copy()

        # Apply each mask with different colors
        colors = [
            [0, 255, 0],  # Green
            [255, 0, 0],  # Blue
            [0, 0, 255],  # Red
            [255, 255, 0],  # Cyan
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Yellow
        ]

        for idx, mask in enumerate(masks):
            # Normalize mask to 0-1 range
            if mask.max() > 1:
                mask_normalized = mask.astype(np.float32) / 255.0
            else:
                mask_normalized = mask.astype(np.float32)

            # Get color (cycle through colors)
            color = colors[idx % len(colors)]

            # Create overlay for this mask
            overlay = result.copy()
            mask_indices = mask_normalized > 0.5
            overlay[mask_indices] = color

            # Blend overlay with result
            result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

        return result

    def render_mask(
        self, crop: np.ndarray, mask: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Render mask overlay on crop for visualization (legacy method).

        Args:
            crop: Original crop image
            mask: Binary mask from generate_mask
            alpha: Transparency of mask overlay (0.0 to 1.0)

        Returns:
            Image with mask overlay applied
        """
        return self.render_masks(crop, [mask], alpha)
