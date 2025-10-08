#!/usr/bin/env python3
"""
Unit tests for image processing functions in stereo module.
Tests comparison between custom load+preprocess and dust3r load_images.
"""

import os
import sys
import numpy as np
import torch
import pytest
import PIL.Image

# Add the necessary directories to the path to import our modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
mast3r_path = os.path.join(project_root, "mast3r")
dust3r_path = os.path.join(project_root, "mast3r", "dust3r")

# Remove any existing paths that might interfere
sys.path = [p for p in sys.path if not p.endswith("/tests")]

sys.path.insert(0, src_path)
sys.path.insert(0, mast3r_path)
sys.path.insert(0, dust3r_path)

# Import after path setup
from stereo.image import load_image, preprocess_image, convert_image
from dust3r.utils.image import load_images as dust3r_load_images
from loguru import logger
import cv2
import tempfile


def load_images_pair(image1_path, image2_path, size=None):
    """Load and preprocess image pair for MASt3R inference."""
    logger.debug(f"Loading images: {image1_path}, {image2_path}")
    assert size is not None, "Size must be provided"
    images = dust3r_load_images([image1_path, image2_path], size=size)
    return images


@pytest.fixture
def test_images():
    """Set up test fixtures."""
    test_dir = "/home/estebanfoucher/Workspace/MVS_app/assets/extracted_pairs/scene_3/0"
    image1_path = os.path.join(test_dir, "frame_1.png")
    image2_path = os.path.join(test_dir, "frame_2.png")
    size = 512

    # Verify test images exist
    assert os.path.exists(image1_path), f"Test image 1 not found: {image1_path}"
    assert os.path.exists(image2_path), f"Test image 2 not found: {image2_path}"

    return {"image1_path": image1_path, "image2_path": image2_path, "size": size}


def test_load_and_preprocess_vs_dust3r_load_images(test_images):
    """Test that load_image_new + preprocess_image_new produces same output as dust3r load_images."""

    # Method 1: Using our custom functions
    image1 = load_image(test_images["image1_path"], size=test_images["size"])
    image2 = load_image(test_images["image2_path"], size=test_images["size"])

    processed1 = preprocess_image(image1, size=test_images["size"], idx=0)
    processed2 = preprocess_image(image2, size=test_images["size"], idx=1)

    custom_result = [processed1, processed2]

    # Method 2: Using dust3r load_images
    dust3r_result = dust3r_load_images(
        [test_images["image1_path"], test_images["image2_path"]],
        size=test_images["size"],
    )

    # Compare results
    assert len(custom_result) == len(dust3r_result), (
        "Number of processed images should match"
    )

    for i, (custom_img, dust3r_img) in enumerate(zip(custom_result, dust3r_result)):
        # Compare tensor shapes
        assert custom_img["img"].shape == dust3r_img["img"].shape, (
            f"Image {i} tensor shapes should match"
        )

        # Compare true_shape
        np.testing.assert_array_equal(
            custom_img["true_shape"],
            dust3r_img["true_shape"],
            f"Image {i} true_shape should match",
        )

        # Compare idx
        assert custom_img["idx"] == dust3r_img["idx"], f"Image {i} idx should match"

        # Compare instance
        assert custom_img["instance"] == dust3r_img["instance"], (
            f"Image {i} instance should match"
        )

        # Compare tensor values (with some tolerance for floating point differences)
        torch.testing.assert_close(
            custom_img["img"],
            dust3r_img["img"],
            rtol=1e-5,
            atol=1e-6,
            msg=f"Image {i} tensor values should be close",
        )


def test_load_images_pair_vs_dust3r_load_images(test_images):
    """Test that load_images_pair produces same output as dust3r load_images."""

    # Method 1: Using our load_images_pair function
    custom_result = load_images_pair(
        test_images["image1_path"], test_images["image2_path"], size=test_images["size"]
    )

    # Method 2: Using dust3r load_images
    dust3r_result = dust3r_load_images(
        [test_images["image1_path"], test_images["image2_path"]],
        size=test_images["size"],
    )

    # Compare results
    assert len(custom_result) == len(dust3r_result), (
        "Number of processed images should match"
    )

    for i, (custom_img, dust3r_img) in enumerate(zip(custom_result, dust3r_result)):
        # Compare tensor shapes
        assert custom_img["img"].shape == dust3r_img["img"].shape, (
            f"Image {i} tensor shapes should match"
        )

        # Compare true_shape
        np.testing.assert_array_equal(
            custom_img["true_shape"],
            dust3r_img["true_shape"],
            f"Image {i} true_shape should match",
        )

        # Compare idx
        assert custom_img["idx"] == dust3r_img["idx"], f"Image {i} idx should match"

        # Compare instance
        assert custom_img["instance"] == dust3r_img["instance"], (
            f"Image {i} instance should match"
        )

        # Compare tensor values (with some tolerance for floating point differences)
        torch.testing.assert_close(
            custom_img["img"],
            dust3r_img["img"],
            rtol=1e-5,
            atol=1e-6,
            msg=f"Image {i} tensor values should be close",
        )


@pytest.mark.parametrize("size", [224, 512, 1024])
def test_different_sizes(test_images, size):
    """Test with different size parameters."""
    # Test load_images_pair vs dust3r
    custom_result = load_images_pair(
        test_images["image1_path"], test_images["image2_path"], size=size
    )
    dust3r_result = dust3r_load_images(
        [test_images["image1_path"], test_images["image2_path"]], size=size
    )

    # Compare tensor shapes
    for i, (custom_img, dust3r_img) in enumerate(zip(custom_result, dust3r_result)):
        assert custom_img["img"].shape == dust3r_img["img"].shape, (
            f"Size {size}, Image {i} tensor shapes should match"
        )

        # Compare tensor values
        torch.testing.assert_close(
            custom_img["img"],
            dust3r_img["img"],
            rtol=1e-5,
            atol=1e-6,
            msg=f"Size {size}, Image {i} tensor values should be close",
        )


def test_video_frame_reading_vs_save_and_load():
    """Test that direct video frame reading is equivalent to cv2.imwrite + load_image."""
    # Use a video file from the assets
    video_path = (
        "/home/estebanfoucher/Workspace/MVS_app/assets/scene_3/camera_1/camera_1.mp4"
    )

    # Verify video file exists
    assert os.path.exists(video_path), f"Test video not found: {video_path}"

    # Import VideoReader from the video module
    sys.path.insert(0, os.path.join(project_root, "src"))
    from video import VideoReader

    # Create a temporary directory for saving the frame
    with tempfile.TemporaryDirectory() as temp_dir:
        # Method 1: Direct frame reading from video
        video_reader = VideoReader.open_video_file(video_path)
        ret, direct_frame = video_reader.read(frame_number=0)  # Read first frame
        video_reader.release()

        assert ret, "Failed to read frame from video"
        assert direct_frame is not None, "Direct frame is None"

        # Method 2: Save frame with cv2.imwrite and load with load_image
        temp_image_path = os.path.join(temp_dir, "temp_frame.png")
        cv2.imwrite(temp_image_path, direct_frame)

        # Load the saved image using load_image function
        loaded_image = load_image(temp_image_path, size=512)

        # Convert direct frame to PIL Image for comparison
        direct_pil = convert_image(direct_frame)

        # Compare the images
        # Both should be PIL Images in RGB format
        assert loaded_image.mode == direct_pil.mode, "Image modes should match"
        assert loaded_image.size == direct_pil.size, "Image sizes should match"

        # Convert to numpy arrays for pixel comparison
        loaded_array = np.array(loaded_image)
        direct_array = np.array(direct_pil)

        # Compare pixel values (should be identical)
        np.testing.assert_array_equal(
            loaded_array,
            direct_array,
            "Pixel values should be identical between direct reading and save+load",
        )

        print(f"✅ Video frame reading test passed:")
        print(f"   - Direct frame shape: {direct_frame.shape}")
        print(f"   - Loaded image size: {loaded_image.size}")
        print(f"   - Image mode: {loaded_image.mode}")
