"""PCA-based crop analysis module."""

import cv2
import numpy as np

from models.bounding_box import BoundingBox
from models.crop_module import CropModule
from models.image import Image

from .utils import extract_crop_from_bbox, validate_crop_coordinates


class CropModulePCA(CropModule):
    """
    Crop analysis module using Principal Component Analysis (PCA).

    Extracts crops from images and computes their principal axes using PCA.
    The principal axis represents the direction of maximum variance in the crop.
    """

    def __init__(self, n_components: int = 2, use_grayscale: bool = True):
        """
        Initialize the PCA crop module.

        Args:
            n_components: Number of principal components to return (default: 2)
            use_grayscale: If True, convert crops to grayscale before PCA.
                          If False, use RGB channels (default: True)
        """
        self.n_components = n_components
        self.use_grayscale = use_grayscale

    def analyze_crop(self, image: Image, bboxes: list[BoundingBox]) -> list[np.ndarray]:
        """
        Analyze crops using PCA to extract principal axes.

        Args:
            image: Image object containing the image data
            bboxes: List of bounding boxes defining crop regions

        Returns:
            List of numpy arrays, one per bounding box. Each array contains
            the principal axis/axes as a vector. For n_components=2, returns
            a 2D vector [x, y] representing the principal direction.
        """
        results = []

        for bbox in bboxes:
            # Validate and extract crop
            if not validate_crop_coordinates(image, bbox):
                # Return zero vector for invalid crops
                results.append(np.zeros(self.n_components))
                continue

            crop = extract_crop_from_bbox(image, bbox)

            # Handle empty or too small crops
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                results.append(np.zeros(self.n_components))
                continue

            # Compute PCA
            principal_axis = self._compute_pca(crop)
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

    def _compute_pca(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute principal components of a crop using PCA.

        For grayscale: computes PCA on pixel positions weighted by intensity.
        For RGB: computes PCA on pixel positions and color channels.

        Args:
            crop: Numpy array of shape (H, W, 3) or (H, W) representing the crop

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
