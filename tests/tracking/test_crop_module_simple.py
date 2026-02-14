"""Simple test for PCA crop module with 1 axis."""

import cv2
import numpy as np

from models import BoundingBox, Detection, Image
from crop_module import CropModulePCA


def test_pca_1_axis():
    """Test PCA crop module with n_components=1."""
    # Create a simple synthetic image with a clear orientation
    # A white rectangle on black background (horizontal orientation)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 40), (180, 60), (255, 255, 255), -1)

    # Create Image object
    image = Image(image=img, rgb_bgr="BGR")

    # Create a bounding box around the rectangle
    bbox = BoundingBox.from_numpy(np.array([10, 30, 190, 70]))

    # Create detection
    detection = Detection(bbox=bbox, confidence=1.0, class_id=0)

    # Initialize PCA module with 1 component
    pca_module = CropModulePCA(n_components=1, use_grayscale=True)

    # Analyze crop
    results = pca_module.analyze_crop(image, [bbox])

    # Verify results
    assert len(results) == 1, "Should return one result for one bbox"
    result = results[0]
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.shape == (1,), f"Result should have shape (1,), got {result.shape}"
    assert not np.allclose(result, 0), "PCA result should not be zero for non-empty crop"

    print(f"✓ PCA 1-axis test passed. Result: {result}")
    print(f"  Principal axis value: {result[0]:.4f}")


def test_pca_1_axis_vertical():
    """Test PCA with vertical orientation."""
    # Create a vertical rectangle
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 20), (60, 180), (255, 255, 255), -1)

    image = Image(image=img, rgb_bgr="BGR")
    bbox = BoundingBox.from_numpy(np.array([30, 10, 70, 190]))

    pca_module = CropModulePCA(n_components=1, use_grayscale=True)
    results = pca_module.analyze_crop(image, [bbox])

    assert len(results) == 1
    result = results[0]
    assert result.shape == (1,)

    print(f"✓ PCA 1-axis vertical test passed. Result: {result}")
    print(f"  Principal axis value: {result[0]:.4f}")


if __name__ == "__main__":
    test_pca_1_axis()
    test_pca_1_axis_vertical()
    print("\n✓ All simple PCA tests passed!")
