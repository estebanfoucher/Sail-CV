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
                               Used to determine arrow direction.
            layout_direction: Optional 2D unitary vector indicating most likely direction.
                            Used as fallback for arrow direction determination.
        """
        self.n_components = n_components
        self.use_grayscale = use_grayscale
        self.mask_detector = mask_detector
        self.background_detector = background_detector
        self.layout_direction = layout_direction

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

            # Extract mask for current bbox if available
            mask = None
            if all_masks is not None and bbox_idx < len(all_masks):
                full_mask = all_masks[bbox_idx]
                # Extract mask region corresponding to this crop
                # Use the same coordinates as extract_crop_from_bbox to ensure exact match
                x1 = max(0, bbox.xyxy.x1)
                y1 = max(0, bbox.xyxy.y1)
                x2 = min(image.image.shape[1], bbox.xyxy.x2)
                y2 = min(image.image.shape[0], bbox.xyxy.y2)

                # Extract mask region
                mask = full_mask[y1:y2, x1:x2]

                # Ensure mask matches crop size exactly
                crop_h, crop_w = crop.shape[:2]
                if mask.shape != (crop_h, crop_w):
                    logger.warning(
                        f"Mask shape {mask.shape} doesn't match crop shape {(crop_h, crop_w)}, resizing"
                    )
                    mask = cv2.resize(
                        mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )

                logger.debug(
                    f"Extracted mask region for bbox {bbox_idx}: crop_shape={crop.shape[:2]}, "
                    f"mask_shape={mask.shape}, coverage={mask.mean() * 100:.2f}%"
                )

            # Compute PCA (with mask if available)
            principal_axis = self._compute_pca(crop, mask=mask)
            logger.debug(f"PCA result for bbox {bbox_idx}: {principal_axis}")

            # Determine arrow direction using movement mask
            if movement_mask is not None and self.n_components >= 2:
                # Extract movement mask crop
                x1 = max(0, bbox.xyxy.x1)
                y1 = max(0, bbox.xyxy.y1)
                x2 = min(image.image.shape[1], bbox.xyxy.x2)
                y2 = min(image.image.shape[0], bbox.xyxy.y2)
                
                movement_crop = movement_mask[y1:y2, x1:x2]
                
                # Ensure movement crop matches crop size
                crop_h, crop_w = crop.shape[:2]
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
        If mask is provided, only masked pixels are considered.

        Args:
            crop: Numpy array of shape (H, W, 3) or (H, W) representing the crop
            mask: Optional binary mask of shape (H, W). If provided, only pixels
                  where mask > 0 are used for PCA.

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
            coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])
            intensities = crop_gray.flatten().astype(np.float64)

            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.flatten().astype(np.float64)
                # Normalize mask to 0-1 if needed
                if mask_flat.max() > 1:
                    mask_flat = mask_flat / 255.0
                # Filter to only masked pixels
                mask_indices = mask_flat > 0.5
                coords = coords[mask_indices]
                intensities = intensities[mask_indices]
                logger.debug(
                    f"Applied mask: {mask_indices.sum()} pixels out of {len(mask_flat)}"
                )

                if len(coords) == 0:
                    logger.warning("No pixels in mask, returning zero vector")
                    return np.zeros(self.n_components)

            # Normalize intensities to use as weights
            if intensities.sum() > 0:
                weights = intensities / intensities.sum()
            else:
                weights = np.ones_like(intensities) / len(intensities)

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
