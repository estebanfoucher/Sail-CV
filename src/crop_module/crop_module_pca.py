"""PCA-based crop analysis module."""

import cv2
import numpy as np
from loguru import logger

from models.bounding_box import BoundingBox
from models.crop_module import CropModule
from models.image import Image
from models.mask_detector import MaskDetector
from models.background_detector import BackgroundDetector

from .utils import extract_crop_from_bbox, validate_crop_coordinates


class CropModulePCA(CropModule):
    """
    Crop analysis module using Principal Component Analysis (PCA).

    Extracts crops from images and computes their principal axes using PCA.
    The principal axis represents the direction of maximum variance in the crop.
    """

    def __init__(
        self,
        n_components: int = 2,
        use_grayscale: bool = True,
        mask_detector: MaskDetector | None = None,
        background_detector: BackgroundDetector | None = None,
        layout_direction: tuple[float, float] | None = None,
        mask_fusion_alpha: float = 0.7,
        sam_fail_min_coverage: float = 0.01,
        sam_fail_max_coverage: float = 0.90,
        mask_fusion_eps: float = 1e-6,
        use_motion_in_pca_mask: bool = True,
        use_motion_for_direction: bool = False,
    ):
        """
        Initialize the PCA crop module.

        Args:
            n_components: Number of principal components to return (default: 2)
            use_grayscale: If True, convert crops to grayscale before PCA.
                          If False, use RGB channels (default: True)
            mask_detector: Optional mask detector to generate masks before PCA.
                          If provided, PCA will only consider masked regions.
            background_detector: Optional background detector for movement analysis.
                               Used to generate motion masks (for PCA mask fusion and/or direction).
            layout_direction: Optional 2D unitary vector indicating most likely direction.
                            Used as fallback for arrow direction determination.
            mask_fusion_alpha: Fusion weight α for SAM vs motion:
                              maskProb = α*SAM + (1-α)*MotionNorm.
            sam_fail_min_coverage: Minimum SAM mask coverage (mean over crop) before
                                   considering SAM as failed.
            sam_fail_max_coverage: Maximum SAM mask coverage (mean over crop) before
                                   considering SAM as failed (e.g. SAM fills bbox).
            mask_fusion_eps: Small epsilon for safe normalizations.
            use_motion_in_pca_mask: If True, incorporate motion mask into PCA masking.
            use_motion_for_direction: If True, use motion distribution to flip the PCA
                                      axis direction (legacy behavior). Default False.
        """
        self.n_components = n_components
        self.use_grayscale = use_grayscale
        self.mask_detector = mask_detector
        self.background_detector = background_detector
        self.layout_direction = layout_direction
        self.mask_fusion_alpha = float(mask_fusion_alpha)
        self.sam_fail_min_coverage = float(sam_fail_min_coverage)
        self.sam_fail_max_coverage = float(sam_fail_max_coverage)
        self.mask_fusion_eps = float(mask_fusion_eps)
        self.use_motion_in_pca_mask = bool(use_motion_in_pca_mask)
        self.use_motion_for_direction = bool(use_motion_for_direction)

    def _to_float01(self, mask: np.ndarray) -> np.ndarray:
        """Convert a mask to float32 in [0, 1]."""
        m = mask.astype(np.float32, copy=False)
        max_val = float(np.max(m)) if m.size else 0.0
        if max_val > 1.0:
            m = m / 255.0
        return np.clip(m, 0.0, 1.0)

    def _l1_normalize_mask01(
        self, mask01: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        L1-normalize a float mask in [0,1] so sum(mask)=1.

        Returns (mask_l1, mask_sum). If mask_sum is ~0, mask_l1 is None.
        """
        mask_sum = float(np.sum(mask01))
        if mask_sum <= self.mask_fusion_eps:
            return None, mask_sum
        # In-place scale to avoid an extra allocation.
        mask01 *= 1.0 / mask_sum
        return mask01, mask_sum

    def _determine_arrow_direction(
        self,
        pca_vector: np.ndarray,
        movement_mask_crop: np.ndarray | None,
        crop_center: tuple[int, int],
    ) -> np.ndarray:
        """
        Determine correct arrow direction by analyzing movement distribution.

        Projects movement mask pixels onto PCA axis and determines which side
        (tip vs root) has more movement. The tip moves more than the root.

        Args:
            pca_vector: PCA vector [dx, dy] from PCA computation
            movement_mask_crop: Binary movement mask for the crop region
            crop_center: Center coordinates (cx, cy) of the crop

        Returns:
            Corrected PCA vector with proper direction (may be flipped).
        """
        if movement_mask_crop is None or movement_mask_crop.sum() == 0:
            # No movement data, use layout direction if available
            if self.layout_direction is not None:
                logger.debug("No movement data, using layout direction prior")
                return np.array(self.layout_direction[: self.n_components])
            # No prior, return original
            return pca_vector

        # Normalize movement mask to 0-1
        if movement_mask_crop.max() > 1:
            movement_mask_crop = movement_mask_crop.astype(np.float32) / 255.0

        h, w = movement_mask_crop.shape
        cx, cy = crop_center

        # Get movement pixel coordinates
        y_coords, x_coords = np.where(movement_mask_crop > 0.5)
        
        if len(x_coords) == 0:
            # No movement pixels, use layout direction
            if self.layout_direction is not None:
                logger.debug("No movement pixels, using layout direction prior")
                return np.array(self.layout_direction[: self.n_components])
            return pca_vector

        # Project each movement pixel onto PCA axis
        # Projection = (x - cx) * dx + (y - cy) * dy
        dx, dy = pca_vector[0], pca_vector[1]
        
        # Center coordinates relative to crop center
        x_centered = x_coords - cx
        y_centered = y_coords - cy
        
        # Project onto PCA vector
        projections = x_centered * dx + y_centered * dy
        
        # Get movement intensities as weights
        movement_weights = movement_mask_crop[y_coords, x_coords]
        
        # Compute weighted sum on positive side (tip) vs negative side (root)
        positive_mask = projections > 0
        negative_mask = projections < 0
        
        tip_sum = np.sum(movement_weights[positive_mask] * projections[positive_mask])
        root_sum = np.sum(np.abs(movement_weights[negative_mask] * projections[negative_mask]))
        
        logger.debug(
            f"Movement analysis: tip_sum={tip_sum:.2f}, root_sum={root_sum:.2f}, "
            f"movement_pixels={len(x_coords)}"
        )
        
        # Determine direction: if tip has more movement, keep direction; else flip
        if tip_sum > root_sum:
            # Tip side has more movement, direction is correct
            return pca_vector
        elif root_sum > tip_sum:
            # Root side has more movement, flip direction
            logger.debug("Flipping PCA vector direction based on movement analysis")
            return -pca_vector
        else:
            # Inconclusive, use layout direction if available
            if self.layout_direction is not None:
                logger.debug("Inconclusive movement, using layout direction prior")
                return np.array(self.layout_direction[: self.n_components])
            # Keep original if no prior
            return pca_vector

    def analyze_crop(
        self,
        image: Image,
        bboxes: list[BoundingBox],
        precomputed_masks: list[np.ndarray] | None = None,
        precomputed_movement_mask: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Analyze crops using PCA to extract principal axes with direction determination.

        Args:
            image: Image object containing the image data
            bboxes: List of bounding boxes defining crop regions
            precomputed_masks: Optional list of pre-computed masks (avoids double inference).
                             Masks should be full-image size and already clipped to bbox boundaries.
            precomputed_movement_mask: Optional pre-computed movement mask for full image.

        Returns:
            List of numpy arrays, one per bounding box. Each array contains
            the principal axis/axes as a vector with correct direction determined
            by movement analysis. For n_components=2, returns a 2D vector [x, y].
        """
        results = []

        # Use precomputed masks if provided, otherwise generate
        all_masks = precomputed_masks
        if all_masks is None and self.mask_detector is not None:
            # Generate masks only if not provided
            try:
                # Convert all bboxes to XYXY format for mask generation
                bbox_list = [bbox.to_numpy() for bbox in bboxes]
                # Generate all masks at once on full image
                all_masks = self.mask_detector.generate_masks(image.image, bbox_list)
                logger.debug(
                    f"Generated {len(all_masks)} masks for {len(bboxes)} bboxes"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate masks: {e}, continuing without masks"
                )
                all_masks = None
        elif all_masks is not None:
            logger.debug(
                f"Using {len(all_masks)} precomputed masks for {len(bboxes)} bboxes"
            )

        # Generate movement mask if background detector is available
        movement_mask = precomputed_movement_mask
        if movement_mask is None and self.background_detector is not None:
            try:
                movement_mask = self.background_detector.generate_foreground_mask(image.image)
                logger.debug(f"Generated movement mask: shape={movement_mask.shape}")
            except Exception as e:
                logger.warning(f"Failed to generate movement mask: {e}")
                movement_mask = None

        for bbox_idx, bbox in enumerate(bboxes):
            # Validate and extract crop
            if not validate_crop_coordinates(image, bbox):
                # Return zero vector for invalid crops
                results.append(np.zeros(self.n_components))
                continue

            crop = extract_crop_from_bbox(image, bbox)

            # Handle empty or too small crops
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                logger.debug("Empty or too small crop, returning zero vector")
                results.append(np.zeros(self.n_components))
                continue

            crop_h, crop_w = crop.shape[:2]

            # Compute crop coordinates once (match extract_crop_from_bbox clipping)
            x1 = max(0, bbox.xyxy.x1)
            y1 = max(0, bbox.xyxy.y1)
            x2 = min(image.image.shape[1], bbox.xyxy.x2)
            y2 = min(image.image.shape[0], bbox.xyxy.y2)

            # Extract SAM mask crop (if available)
            sam_crop = None
            if all_masks is not None and bbox_idx < len(all_masks):
                full_mask = all_masks[bbox_idx]
                sam_crop = full_mask[y1:y2, x1:x2]

                # Ensure SAM crop matches crop size exactly
                if sam_crop.shape != (crop_h, crop_w):
                    logger.warning(
                        f"Mask shape {sam_crop.shape} doesn't match crop shape {(crop_h, crop_w)}, resizing"
                    )
                    sam_crop = cv2.resize(
                        sam_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )

                logger.debug(
                    f"Extracted mask region for bbox {bbox_idx}: crop_shape={crop.shape[:2]}, "
                    f"mask_shape={sam_crop.shape}, coverage={sam_crop.mean() * 100:.2f}%"
                )

            # Extract motion mask crop (if available/desired)
            motion_l1 = None
            if self.use_motion_in_pca_mask and movement_mask is not None:
                motion_crop = movement_mask[y1:y2, x1:x2]
                if motion_crop.shape != (crop_h, crop_w):
                    motion_crop = cv2.resize(
                        motion_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )
                motion_crop_f = self._to_float01(motion_crop)
                motion_l1, _motion_sum = self._l1_normalize_mask01(motion_crop_f)

            # Build fused mask probability for PCA
            mask_prob = None
            sam_ok = False
            if sam_crop is not None:
                sam_crop_f = self._to_float01(sam_crop)
                sam_l1, sam_sum = self._l1_normalize_mask01(sam_crop_f)
                # coverage is mean(sam_crop_f) but reuse sam_sum to avoid an extra pass
                coverage = sam_sum / float(crop_h * crop_w) if crop_h and crop_w else 0.0
                sam_ok = (
                    self.sam_fail_min_coverage
                    <= coverage
                    <= self.sam_fail_max_coverage
                )
                if sam_l1 is not None:
                    if motion_l1 is not None:
                        if sam_ok:
                            # L1-normalized fusion: both masks sum to 1, so α is meaningful.
                            a = float(np.clip(self.mask_fusion_alpha, 0.0, 1.0))
                            sam_l1 *= a
                            motion_l1 *= 1.0 - a
                            sam_l1 += motion_l1
                            mask_prob = sam_l1
                        else:
                            # Coverage-based SAM failure: fallback to motion-only.
                            mask_prob = motion_l1
                    else:
                        # No motion available: use SAM alone (even if coverage is suspicious).
                        mask_prob = sam_l1
                else:
                    # SAM is effectively empty: treat as missing.
                    mask_prob = motion_l1
            else:
                # No SAM available: fallback to motion-only (if any).
                mask_prob = motion_l1

            # Compute PCA (with soft mask if available)
            principal_axis = self._compute_pca(crop, mask=mask_prob)
            logger.debug(f"PCA result for bbox {bbox_idx}: {principal_axis}")

            # Determine arrow direction using movement mask
            if self.use_motion_for_direction and movement_mask is not None and self.n_components >= 2:
                # Extract movement mask crop
                movement_crop = movement_mask[y1:y2, x1:x2]
                
                # Ensure movement crop matches crop size
                if movement_crop.shape != (crop_h, crop_w):
                    movement_crop = cv2.resize(
                        movement_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )
                
                # Get crop center (relative to crop coordinates)
                crop_center = (crop_w // 2, crop_h // 2)
                
                # Determine correct arrow direction
                principal_axis = self._determine_arrow_direction(
                    principal_axis, movement_crop, crop_center
                )
                logger.debug(f"Corrected PCA direction for bbox {bbox_idx}: {principal_axis}")

            results.append(principal_axis)

        return results

    def _extract_crop(self, image: Image, bbox: BoundingBox) -> np.ndarray:
        """
        Extract crop from image based on bounding box.

        Args:
            image: Image object containing the image data
            bbox: BoundingBox defining the crop region

        Returns:
            Numpy array of the cropped region
        """
        return extract_crop_from_bbox(image, bbox)

    def _compute_pca(
        self, crop: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute principal components of a crop using PCA.

        For grayscale: computes PCA on pixel positions weighted by intensity.
        For RGB: computes PCA on pixel positions and color channels.
        If mask is provided, it is treated as a *soft probability mask* in [0,1]
        that down-weights pixels (rather than a hard include/exclude).

        Args:
            crop: Numpy array of shape (H, W, 3) or (H, W) representing the crop
            mask: Optional mask of shape (H, W). If provided, it is interpreted as
                  a soft probability mask in [0,1] (or uint8 0/255) and used to
                  weight pixel intensities.

        Returns:
            Numpy array of shape (n_components,) containing the principal axis/axes.
            For 2D crops, this represents the direction of maximum variance.
        """
        if self.use_grayscale:
            # Convert to grayscale if needed
            if len(crop.shape) == 3:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                crop_gray = crop

            h, w = crop_gray.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]

            # Flatten coordinates and intensities
            coords_full = np.column_stack([x_coords.flatten(), y_coords.flatten()])
            intensities_full = crop_gray.flatten().astype(np.float64)

            # Choose weights
            if mask is not None:
                mask_flat = mask.flatten().astype(np.float64)
                if mask_flat.max() > 1.0:
                    mask_flat = mask_flat / 255.0
                mask_flat = np.clip(mask_flat, 0.0, 1.0)

                # Soft weighting: multiply intensity by mask probability.
                pixel_weight_full = intensities_full * mask_flat
                keep = pixel_weight_full > 0

                if np.any(keep):
                    coords = coords_full[keep]
                    pixel_weight = pixel_weight_full[keep]
                    total_w = float(np.sum(pixel_weight))
                    if total_w > 0:
                        weights = pixel_weight / total_w
                    else:
                        weights = np.ones(len(coords), dtype=np.float64) / len(coords)
                    logger.debug(
                        f"Applied soft mask: kept {int(keep.sum())} pixels out of {len(mask_flat)}"
                    )
                else:
                    # Safe fallback: ignore mask if it annihilates all weights.
                    logger.warning(
                        "Mask produced zero total weight; falling back to unmasked PCA"
                    )
                    coords = coords_full
                    if intensities_full.sum() > 0:
                        weights = intensities_full / intensities_full.sum()
                    else:
                        weights = np.ones_like(intensities_full) / len(intensities_full)
            else:
                coords = coords_full
                if intensities_full.sum() > 0:
                    weights = intensities_full / intensities_full.sum()
                else:
                    weights = np.ones_like(intensities_full) / len(intensities_full)

            # Compute weighted mean (centroid)
            weighted_mean = np.sum(coords * weights[:, np.newaxis], axis=0)

            # Center coordinates
            centered_coords = coords - weighted_mean

            # Compute weighted covariance matrix
            # Cov = sum(weights[i] * (coords[i] - mean) * (coords[i] - mean)^T)
            cov = np.dot((centered_coords * weights[:, np.newaxis]).T, centered_coords)

            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov)

            # Sort by eigenvalue (descending)
            idx = eigenvals.argsort()[::-1]
            eigenvecs = eigenvecs[:, idx]

            # Return first n_components of first principal component
            if eigenvecs.shape[1] > 0:
                principal_axis = eigenvecs[:, 0][: self.n_components]
            else:
                principal_axis = np.zeros(self.n_components)

            # Pad if needed
            if len(principal_axis) < self.n_components:
                principal_axis = np.pad(
                    principal_axis,
                    (0, self.n_components - len(principal_axis)),
                    mode="constant",
                )

            return principal_axis

        else:
            # For RGB: use pixel positions and color channels as features
            if len(crop.shape) != 3:
                # If already grayscale, convert to 3-channel
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

            h, w = crop.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w]

            # Combine coordinates with color values as features
            features = np.column_stack(
                [
                    x_coords.flatten(),
                    y_coords.flatten(),
                    crop[:, :, 0].flatten(),
                    crop[:, :, 1].flatten(),
                    crop[:, :, 2].flatten(),
                ]
            ).astype(np.float64)

            # Center the data
            mean = np.mean(features, axis=0)
            centered = features - mean

            # Compute PCA using SVD
            if centered.shape[0] < 2:
                return np.zeros(self.n_components)

            _, _, Vt = np.linalg.svd(centered, full_matrices=False)

            # Vt has shape (min(n_samples, n_features), n_features)
            # First row is first principal component
            if Vt.shape[0] > 0:
                principal_axis = Vt[0, : self.n_components]
            else:
                principal_axis = np.zeros(self.n_components)

            # Pad if needed
            if len(principal_axis) < self.n_components:
                principal_axis = np.pad(
                    principal_axis,
                    (0, self.n_components - len(principal_axis)),
                    mode="constant",
                )

            return principal_axis
