#!/usr/bin/env python3
"""
MASt3R Core Inference Module
Handles model loading and raw inference operations
"""

import os
import time

import cv2
import numpy as np
from dust3r.inference import inference
from loguru import logger
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R


class MASt3RInferenceEngine:
    """Core inference engine for MASt3R"""

    def __init__(self, model_path=None, device=None):
        """Initialize MASt3R inference engine with model path and device."""
        self.device = device
        self.model = None
        self.model_path = model_path

        assert self.device is not None, "Device must be provided"
        assert self.model_path is not None, "Model path must be provided"

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load MASt3R model from pretrained checkpoint."""
        logger.info(f"Loading model from: {self.model_path}")

        # Control verbosity: only verbose if LOGURU_LEVEL is DEBUG
        log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
        verbose = log_level == "DEBUG"
        logger.debug(
            f"Model loading verbosity set to: {verbose} (LOGURU_LEVEL={log_level})"
        )

        self.model = AsymmetricMASt3R.from_pretrained(
            self.model_path, verbose=verbose
        ).to(self.device)
        logger.info("Model loaded successfully")
        return self.model

    def save_inference_images(self, images, output_dir):
        """Save preprocessed inference images to output directory."""
        logger.debug(f"Saving inference images to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "inference_images"), exist_ok=True)

        # Convert tensors to numpy arrays and handle the format properly
        img1_np = images[0]["img"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img2_np = images[1]["img"].squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Convert from [0,1] range to [0,255] and ensure uint8
        img1_np = (img1_np * 255).astype(np.uint8)
        img2_np = (img2_np * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_dir, "inference_images", "frame_1.png"), img1_np
        )
        cv2.imwrite(
            os.path.join(output_dir, "inference_images", "frame_2.png"), img2_np
        )

    def run_inference(self, images):
        """Run MASt3R inference on preprocessed image pair."""
        logger.debug("MASt3R inferring")

        start_time = time.time()
        output = inference(
            [tuple(images)], self.model, self.device, batch_size=1, verbose=False
        )
        end_time = time.time()

        logger.debug(f"MASt3R inference time: {end_time - start_time:.2f} seconds")

        return output

    def extract_raw_data(self, output, subsample=None):
        """Extract matches, 3D points, and colors from MASt3R inference output."""
        assert subsample is not None, "Subsample must be provided, set to 16,8,4,2,1"

        start_time = time.time()
        # Extract results
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        # Get descriptors and matches
        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=subsample,
            device=self.device,
            dist="dot",
            block_size=2**13,
        )

        # Get 3D points from the available keys
        pts3d_1 = pred1["pts3d"].squeeze(0).detach().cpu().numpy()
        pts3d_2 = pred2["pts3d_in_other_view"].squeeze(0).detach().cpu().numpy()

        # Get image colors for point cloud coloring
        img1_colors = view1["img"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img2_colors = view2["img"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        end_time = time.time()
        logger.debug(
            f"Extracted raw data in {end_time - start_time:.2f} seconds: {len(matches_im0)} matches found"
        )
        return {
            "view1": view1,
            "view2": view2,
            "pred1": pred1,
            "pred2": pred2,
            "matches_im0": matches_im0,
            "matches_im1": matches_im1,
            "pts3d_1": pts3d_1,
            "pts3d_2": pts3d_2,
            "img1_colors": img1_colors,
            "img2_colors": img2_colors,
            "num_matches": len(matches_im0),
        }

    def run_inference_with_images(self, images, subsample=None):
        """Run complete MASt3R pipeline: inference + data extraction."""

        # Run inference
        output = self.run_inference(images)

        # Extract raw data
        raw_data = self.extract_raw_data(output, subsample)

        return raw_data
