"""YOLO-based classifier for crop classification."""

import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from model_weights import resolve_model_path
from models.classifier import ClassifierConfig


class Classifier:
    """
    Classifier service class for crop classification using YOLO models.

    Input:
        config: ClassifierConfig object containing classifier configuration

    The classifier's classify_crop method:
        Input: Crop image as numpy array (BGR format)
        Output: Tuple of (class_id, confidence)
    """

    def __init__(
        self,
        config: ClassifierConfig,
        *,
        project_root: Path | None = None,
    ):
        """
        Initialize Classifier with model configuration.

        Args:
            config: ClassifierConfig object (Pydantic validated)
            project_root: Optional project root for resolving relative paths
                and for downloading from Hugging Face if not found locally.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Classifier using device: {self.device}")

        # Resolve path: local or download from Hugging Face (estefoucher/tell-tale-detector)
        model_path = resolve_model_path(
            config.model_path,
            project_root=project_root,
        )

        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        logger.info(f"✓ Classifier model loaded and moved to {self.device}")

    def classify_crop(self, crop: np.ndarray | None) -> tuple[int | None, float]:
        """
        Classify a crop image and return the predicted class.

        Args:
            crop: Crop image as numpy array (BGR format) or None if crop is invalid

        Returns:
            Tuple of (class_id, confidence). Returns (None, 0.0) if crop is None or empty.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        start_time = time.perf_counter()

        # Ensure crop is numpy array (should already be from pipeline)
        if isinstance(crop, torch.Tensor):
            crop = crop.cpu().numpy()

        # Handle grayscale or single channel crops
        if len(crop.shape) == 2 or (len(crop.shape) == 3 and crop.shape[2] == 1):
            import cv2

            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # Run inference (verbose=False to suppress output)
        inference_start = time.perf_counter()
        results = self.model(crop, verbose=False)
        inference_time = (time.perf_counter() - inference_start) * 1000  # ms

        logger.debug(f"Classifier inference: {inference_time:.1f}ms")

        # Extract results - handle both detection and classification models
        result = results[0]

        # Check if it's a classification model (has probs attribute)
        if hasattr(result, "probs") and result.probs is not None:
            # Classification model - use probs
            probs = result.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)

            # Apply confidence threshold if configured
            if (
                self.config.confidence_threshold > 0.0
                and confidence < self.config.confidence_threshold
            ):
                logger.debug(
                    f"Classification confidence {confidence:.3f} below threshold "
                    f"{self.config.confidence_threshold}"
                )
                return None, confidence

            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Classifier total: {total_time:.1f}ms -> class_id={class_id}, "
                f"confidence={confidence:.3f}"
            )
            return class_id, confidence

        # Otherwise, check for detection model (has boxes)
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Get the top prediction (highest confidence)
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()

            # Find index of highest confidence
            top_idx = int(confidences.argmax())
            class_id = int(class_ids[top_idx])
            confidence = float(confidences[top_idx])

            # Apply confidence threshold if configured
            if (
                self.config.confidence_threshold > 0.0
                and confidence < self.config.confidence_threshold
            ):
                logger.debug(
                    f"Classification confidence {confidence:.3f} below threshold "
                    f"{self.config.confidence_threshold}"
                )
                return None, confidence

            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Classifier total: {total_time:.1f}ms -> class_id={class_id}, "
                f"confidence={confidence:.3f}"
            )
            return class_id, confidence
        else:
            # No detections/classifications
            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Classifier total (no detections): {total_time:.1f}ms")
            return None, 0.0
