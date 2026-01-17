"""SAM2-based mask detection for crops using Ultralytics."""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from models.mask_detector import MaskDetector


class MaskDetectorSAM(MaskDetector):
    """
    Mask detector using Segment Anything Model 2 (SAM2) via Ultralytics.

    Uses Ultralytics SAM2 implementation for inference on full images with bbox prompts.
    """

    def __init__(self, model_path: Path | str | None = None, device: str = "cpu"):
        """
        Initialize SAM2 mask detector using Ultralytics.

        Args:
            model_path: Path to SAM2 model checkpoint. If None, uses Ultralytics SAM2 default (sam2_b.pt).
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = device
        self.model_path = Path(model_path) if model_path else None
        self._model = None

    @property
    def model(self):
        """Lazy initialization of Ultralytics SAM model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load Ultralytics SAM2 model."""
        try:
            from ultralytics import SAM

            if self.model_path is None:
                # Use Ultralytics SAM2 model (will download if needed)
                logger.debug("Loading Ultralytics SAM2 with default model")
                self._model = SAM("sam2_b.pt")  # sam2_b.pt is SAM2 base model
            else:
                logger.debug(f"Loading Ultralytics SAM2 from {self.model_path}")
                self._model = SAM(str(self.model_path))

            # Set device
            if hasattr(self._model, "to"):
                self._model.to(self.device)
            elif hasattr(self._model, "model"):
                self._model.model.to(self.device)

        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Please install: pip install ultralytics"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Ultralytics SAM2 model: {e}")
            raise

    def generate_masks(self, image: np.ndarray, bboxes: list) -> list[np.ndarray]:
        """
        Generate binary masks for multiple bounding boxes on a full image using Ultralytics SAM2.

        Args:
            image: Full image as numpy array of shape (H, W, 3) or (H, W)
            bboxes: List of bounding boxes in XYXY format [[x1, y1, x2, y2], ...]

        Returns:
            List of binary masks, one per bounding box. Each mask is (H, W) with values 0/1.
        """
        logger.debug(
            f"Generating masks for full image: shape={image.shape}, {len(bboxes)} bboxes"
        )

        h, w = image.shape[:2]

        try:
            # Ultralytics SAM expects BGR format
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 3:
                image_bgr = image
            else:
                image_bgr = image

            logger.debug(f"Image BGR shape: {image_bgr.shape}")

            # Convert bboxes to center points for SAM prompts
            # Use center of each bbox as a point prompt
            points = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                points.append([center_x, center_y])

            points_array = np.array(points)
            labels = np.ones(len(points), dtype=int)  # All foreground points

            logger.debug(f"Using {len(points)} point prompts from bbox centers")

            # Run Ultralytics SAM2 inference on full image with all points
            # SAM2 supports batch point prompts
            results = self.model.predict(
                image_bgr,
                points=points_array,
                labels=labels,
                verbose=False,
            )

            masks = []
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None and len(result.masks.data) > 0:
                    # Extract all masks
                    for i in range(len(result.masks.data)):
                        mask = result.masks.data[i].cpu().numpy()
                        # Resize to original image size if needed
                        if mask.shape != (h, w):
                            mask = cv2.resize(
                                mask, (w, h), interpolation=cv2.INTER_NEAREST
                            )
                        mask = mask.astype(np.uint8)
                        # Ensure binary (0/1)
                        mask = (mask > 0.5).astype(np.uint8)
                        masks.append(mask)
                        logger.debug(
                            f"Mask {i}: shape={mask.shape}, coverage={mask.mean() * 100:.2f}%"
                        )

            # If we got fewer masks than bboxes, pad with full masks
            while len(masks) < len(bboxes):
                logger.warning(
                    f"Got {len(masks)} masks but {len(bboxes)} bboxes, padding with full mask"
                )
                masks.append(np.ones((h, w), dtype=np.uint8))

            # Return only as many masks as we have bboxes
            return masks[: len(bboxes)]

        except Exception as e:
            logger.error(f"Error generating masks: {e}")
            # Return full masks as fallback
            return [np.ones((h, w), dtype=np.uint8) for _ in bboxes]

    def generate_mask(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate binary mask for a crop using Ultralytics SAM2.

        Args:
            crop: Numpy array of shape (H, W, 3) or (H, W) representing the crop

        Returns:
            Binary mask as numpy array of shape (H, W) with values 0/1.
        """
        logger.debug(f"Generating mask for crop: shape={crop.shape}")

        h, w = crop.shape[:2]

        try:
            # Ultralytics SAM2 expects BGR format (like other Ultralytics models)
            if len(crop.shape) == 2:
                # Grayscale to BGR
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            elif crop.shape[2] == 3:
                # Already 3-channel, assume BGR
                crop_bgr = crop
            else:
                crop_bgr = crop

            logger.debug(f"Crop BGR shape: {crop_bgr.shape}")

            # Use center point as prompt
            center_point = np.array([[w // 2, h // 2]])

            # Run Ultralytics SAM2 inference
            # Ultralytics SAM2 uses predict with point prompts
            results = self.model.predict(
                crop_bgr,
                points=center_point,
                labels=[1],  # 1 = foreground point
                verbose=False,
            )

            # Extract mask from results
            if results and len(results) > 0:
                result = results[0]
                # Ultralytics SAM returns masks in results.masks
                if result.masks is not None and len(result.masks.data) > 0:
                    # Get the first (best) mask
                    mask = result.masks.data[0].cpu().numpy()
                    # Resize to original crop size if needed
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(np.uint8)
                    logger.debug(
                        f"Generated mask: shape={mask.shape}, coverage={mask.mean() * 100:.2f}%"
                    )
                    # Ensure binary (0/1)
                    mask = (mask > 0.5).astype(np.uint8)
                    return mask
                else:
                    logger.warning("No masks found in SAM results, using full mask")
                    return np.ones((h, w), dtype=np.uint8)
            else:
                logger.warning("No results from SAM, using full mask")
                return np.ones((h, w), dtype=np.uint8)

        except Exception as e:
            logger.error(f"Error generating mask: {e}")
            # Return full mask as fallback
            return np.ones((h, w), dtype=np.uint8)

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
        return super().render_masks(image, masks, alpha)
