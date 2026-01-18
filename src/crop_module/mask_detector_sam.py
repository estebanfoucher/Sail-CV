"""SAM2-based mask detection for crops using Ultralytics."""

import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

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
            model_path: Path to SAM2 model checkpoint. If None, uses SAM2 tiny (sam2_t.pt) for fast inference.
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
                # Use Ultralytics SAM2 tiny model (fastest, smallest ~40MB)
                logger.info(f"Loading Ultralytics SAM2-tiny model on device: {self.device}")
                self._model = SAM("sam2_t.pt")  # sam2_t.pt is SAM2 tiny model (fastest)
            else:
                logger.info(f"Loading Ultralytics SAM2 from {self.model_path} on device: {self.device}")
                self._model = SAM(str(self.model_path))

            # Set device
            if hasattr(self._model, "to"):
                self._model.to(self.device)
                logger.info(f"✓ SAM2 model moved to {self.device}")
            elif hasattr(self._model, "model"):
                self._model.model.to(self.device)
                logger.info(f"✓ SAM2 model moved to {self.device}")

        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Please install: pip install ultralytics"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Ultralytics SAM2 model: {e}")
            raise

    def _compute_mask_bbox_iou(self, mask: np.ndarray, bbox: list) -> float:
        """Compute IoU between mask's bounding box and target bbox."""
        # Get mask bounding box
        mask_coords = np.where(mask > 0.5)
        if len(mask_coords[0]) == 0:
            return 0.0
        
        mask_x1, mask_y1 = mask_coords[1].min(), mask_coords[0].min()
        mask_x2, mask_y2 = mask_coords[1].max(), mask_coords[0].max()
        
        # Target bbox
        x1, y1, x2, y2 = bbox[:4]
        
        # Compute intersection
        inter_x1 = max(mask_x1, x1)
        inter_y1 = max(mask_y1, y1)
        inter_x2 = min(mask_x2, x2)
        inter_y2 = min(mask_y2, y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        mask_area = (mask_x2 - mask_x1) * (mask_y2 - mask_y1)
        bbox_area = (x2 - x1) * (y2 - y1)
        union_area = mask_area + bbox_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def generate_masks(self, image: np.ndarray, bboxes: list) -> list[np.ndarray]:
        """
        Generate binary masks for multiple bounding boxes on a full image using Ultralytics SAM2.

        Uses Hungarian assignment to match masks to bboxes based on IoU, and clips masks to bbox boundaries.

        Args:
            image: Full image as numpy array of shape (H, W, 3) or (H, W)
            bboxes: List of bounding boxes in XYXY format [[x1, y1, x2, y2], ...]

        Returns:
            List of binary masks, one per bounding box. Each mask is (H, W) with values 0/1,
            clipped to bbox boundaries. Unmatched bboxes get zero masks.
        """
        start_time = time.perf_counter()
        
        logger.debug(
            f"Generating masks for full image: shape={image.shape}, {len(bboxes)} bboxes"
        )

        h, w = image.shape[:2]

        try:
            # Ultralytics SAM expects BGR format
            prep_start = time.perf_counter()
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
            prep_time = (time.perf_counter() - prep_start) * 1000

            logger.debug(f"Using {len(points)} point prompts from bbox centers (prep: {prep_time:.1f}ms)")

            # Run Ultralytics SAM2 inference on full image with all points
            # SAM2 supports batch point prompts
            inference_start = time.perf_counter()
            results = self.model.predict(
                image_bgr,
                points=points_array,
                labels=labels,
                verbose=False,
            )
            inference_time = (time.perf_counter() - inference_start) * 1000
            logger.debug(f"SAM2 inference: {inference_time:.1f}ms for {len(bboxes)} bboxes")

            # Extract raw masks from SAM2 results
            raw_masks = []
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
                        raw_masks.append(mask)
                        logger.debug(
                            f"Raw mask {i}: shape={mask.shape}, coverage={mask.mean() * 100:.2f}%"
                        )

            # Hungarian assignment: match masks to bboxes using IoU
            num_bboxes = len(bboxes)
            num_masks = len(raw_masks)

            if num_masks > 0:
                # Build cost matrix: IoU (we want to maximize, so use negative for minimization)
                cost_matrix = np.zeros((num_bboxes, num_masks))
                for bbox_idx, bbox in enumerate(bboxes):
                    for mask_idx, mask in enumerate(raw_masks):
                        iou = self._compute_mask_bbox_iou(mask, bbox)
                        cost_matrix[bbox_idx, mask_idx] = -iou  # Negative for minimization
                
                # Hungarian algorithm: find optimal assignment
                bbox_indices, mask_indices = linear_sum_assignment(cost_matrix)
                
                # Assign masks to bboxes
                assigned_masks = [None] * num_bboxes
                for bbox_idx, mask_idx in zip(bbox_indices, mask_indices):
                    iou = -cost_matrix[bbox_idx, mask_idx]  # Convert back to IoU
                    if iou > 0.3:  # Threshold for valid match
                        assigned_masks[bbox_idx] = raw_masks[mask_idx]
                        logger.debug(f"Assigned mask {mask_idx} to bbox {bbox_idx} (IoU: {iou:.2f})")
                    else:
                        assigned_masks[bbox_idx] = np.zeros((h, w), dtype=np.uint8)
                        logger.warning(f"Mask {mask_idx} IoU too low ({iou:.2f}) for bbox {bbox_idx}, using zero mask")
                
                # Fill unmatched bboxes with zero masks
                for bbox_idx in range(num_bboxes):
                    if assigned_masks[bbox_idx] is None:
                        assigned_masks[bbox_idx] = np.zeros((h, w), dtype=np.uint8)
                        logger.warning(f"No mask assigned to bbox {bbox_idx}, using zero mask")
            else:
                # No masks returned, all get zero masks
                assigned_masks = [np.zeros((h, w), dtype=np.uint8) for _ in bboxes]
                logger.warning(f"No masks returned from SAM2, using zero masks for all {num_bboxes} bboxes")

            # Clip each mask to its bbox
            clipped_masks = []
            for i, (mask, bbox) in enumerate(zip(assigned_masks, bboxes)):
                x1, y1, x2, y2 = bbox[:4]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                # Zero out everything outside bbox in full mask
                clipped_mask = np.zeros((h, w), dtype=np.uint8)
                clipped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                
                clipped_masks.append(clipped_mask)

            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"SAM2 total: {total_time:.1f}ms ({len(clipped_masks)} masks generated and clipped)")
            return clipped_masks

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error generating masks: {e} (time: {total_time:.1f}ms)")
            # Return zero masks as fallback (not full masks)
            return [np.zeros((h, w), dtype=np.uint8) for _ in bboxes]

    def generate_mask(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate binary mask for a crop using Ultralytics SAM2.

        Args:
            crop: Numpy array of shape (H, W, 3) or (H, W) representing the crop

        Returns:
            Binary mask as numpy array of shape (H, W) with values 0/1.
        """
        start_time = time.perf_counter()
        
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
            inference_start = time.perf_counter()
            results = self.model.predict(
                crop_bgr,
                points=center_point,
                labels=[1],  # 1 = foreground point
                verbose=False,
            )
            inference_time = (time.perf_counter() - inference_start) * 1000
            logger.debug(f"SAM2 single crop inference: {inference_time:.1f}ms")

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
                    # Ensure binary (0/1)
                    mask = (mask > 0.5).astype(np.uint8)
                    total_time = (time.perf_counter() - start_time) * 1000
                    logger.debug(
                        f"Generated mask: shape={mask.shape}, coverage={mask.mean() * 100:.2f}%, total: {total_time:.1f}ms"
                    )
                    return mask
                else:
                    total_time = (time.perf_counter() - start_time) * 1000
                    logger.warning(f"No masks found in SAM results, using full mask (time: {total_time:.1f}ms)")
                    return np.ones((h, w), dtype=np.uint8)
            else:
                total_time = (time.perf_counter() - start_time) * 1000
                logger.warning(f"No results from SAM, using full mask (time: {total_time:.1f}ms)")
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
