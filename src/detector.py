import json
import time
from pathlib import Path

import cv2
import torch
from loguru import logger

from models import BoundingBox, Detection, Image, ModelSpecs
from models.bounding_box import XYXY


class Model:
    """
    Model class for object detection

    Input:
        specs: ModelSpecs object containing model configuration

    The model's predict method:
        Input: Image object (Pydantic validated)
        Output: List of Detection objects (Pydantic validated)
    """

    def __init__(self, specs: ModelSpecs):
        self.specs = specs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Detector using device: {self.device}")

        # Load model based on architecture
        if specs.architecture == "rt-detr":
            from ultralytics import RTDETR

            self.model = RTDETR(str(specs.model_path))
        elif specs.architecture == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(str(specs.model_path))
        else:
            raise ValueError(f"Invalid architecture: {specs.architecture}")
        self.model.to(self.device)
        logger.info(f"✓ Model moved to {self.device}")

    def format_inference_results(self, results):
        """Format inference results exactly like the tracker"""
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            bboxes = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            return bboxes, confidences, class_ids
        else:
            return None, None, None

    def predict(self, image: Image) -> list[Detection]:
        """
        Predict tell-tale objects in the image

        Input:
            image: Image object (Pydantic validated)
            conf: Confidence threshold (not used, kept for compatibility)

        Output:
            List[Detection]: List of Detection objects (Pydantic validated)
        """
        start_time = time.perf_counter()

        # Convert Image to numpy array for model inference
        # Models typically expect BGR format
        image_array = image.to_bgr()

        # Run inference (verbose=False to suppress output)
        inference_start = time.perf_counter()
        results = self.model(image_array, verbose=False)
        inference_time = (time.perf_counter() - inference_start) * 1000  # ms

        logger.debug(f"Detector inference: {inference_time:.1f}ms")

        # Format results
        bboxes, confidences, class_ids = self.format_inference_results(results)

        # Handle None case (no detections)
        if bboxes is None or len(bboxes) == 0:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Detector total (no detections): {total_time:.1f}ms")
            return []

        # Convert to Detection objects
        detections = []
        for i in range(len(bboxes)):
            detection = Detection(
                bbox=BoundingBox.from_numpy(xyxy=bboxes[i]),
                confidence=float(confidences[i]),
                class_id=int(class_ids[i]),
            )
            detections.append(detection)

        total_time = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Detector total: {total_time:.1f}ms ({len(detections)} detections)"
        )

        return detections

    def render_result(
        self,
        image: Image,
        detections: list[Detection],
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> Image:
        """
        Visualize the detection results
        """
        result_image = image.image.copy()

        for detection in detections:
            xyxy = detection.bbox.xyxy
            x1, y1, x2, y2 = int(xyxy.x1), int(xyxy.y1), int(xyxy.x2), int(xyxy.y2)
            conf = detection.confidence
            class_id = detection.class_id

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label with confidence
            label = f"Class {class_id}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return Image(image=result_image, rgb_bgr=image.rgb_bgr)


class Detector:
    """
    Detector service class for object detection

    Input:
        specs: ModelSpecs object containing model configuration

    The detector's detect method:
        Input: Image object
        Output: List[Detection] objects
    """

    def __init__(self, specs: ModelSpecs):
        """
        Initialize Detector with model specifications

        Input:
            specs: ModelSpecs object (Pydantic validated)
        """
        self.specs = specs
        self._model: Model | None = None

    @property
    def model(self) -> Model:
        """Get the Model instance (lazy initialization)"""
        if self._model is None:
            self._model = Model(self.specs)
        return self._model

    def detect(self, image: Image) -> list[Detection]:
        """
        Detect objects in an image

        Input:
            image: Image object (Pydantic validated)

        Output:
            List[Detection]: List of Detection objects (Pydantic validated)
        """
        return self.model.predict(image)

    def render_result(
        self,
        image: Image,
        detections: list[Detection],
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> Image:
        """
        Render detection results on image

        Input:
            image: Image object (Pydantic validated)
            detections: List of Detection objects (Pydantic validated)
            color: BGR color tuple for bounding boxes
            thickness: Thickness of bounding box lines

        Output:
            Image: Rendered image with detection boxes drawn
        """
        return self.model.render_result(image, detections, color, thickness)


class FakeModel:
    """
    Fake model class that reads precomputed detections from JSON file.

    Used for testing to avoid running actual model inference.
    Automatically increments frame_number after each predict() call.
    """

    def __init__(self, precomputed_results_json_path: Path | str):
        """
        Initialize FakeModel with precomputed results JSON file.

        Args:
            precomputed_results_json_path: Path to JSON file with precomputed detections
        """
        self.precomputed_results_json_path = Path(precomputed_results_json_path)
        if not self.precomputed_results_json_path.exists():
            raise FileNotFoundError(
                f"Precomputed results file not found: {self.precomputed_results_json_path}"
            )
        self.frame_number = 0
        self._precomputed_results: dict | None = None
        logger.info(f"Initialized FakeModel with {self.precomputed_results_json_path}")

    def _load_results(self) -> dict:
        """Load precomputed results from JSON file (lazy loading)."""
        if self._precomputed_results is None:
            with self.precomputed_results_json_path.open() as f:
                self._precomputed_results = json.load(f)
            logger.debug(f"Loaded {len(self._precomputed_results)} frames from JSON")
        return self._precomputed_results

    def predict(self, image: Image) -> list[Detection]:
        """
        Predict tell-tale objects by reading from precomputed results.

        Args:
            image: Image object (not used, but kept for interface compatibility)

        Returns:
            List[Detection]: List of Detection objects from precomputed results
        """
        results = self._load_results()
        frame_key = str(self.frame_number)

        if frame_key not in results:
            logger.warning(
                f"Frame {self.frame_number} not found in precomputed results, returning empty list"
            )
            # Auto-increment even if frame not found
            self.frame_number += 1
            return []

        # Get detections for current frame
        detections_dict = results[frame_key]

        # Convert dict detections to Detection objects
        detections = []
        for det_dict in detections_dict:
            # Handle nested bbox structure
            bbox_dict = det_dict.get("bbox", {})
            xyxy_dict = bbox_dict.get("xyxy", {})

            xyxy = XYXY(
                x1=int(xyxy_dict["x1"]),
                y1=int(xyxy_dict["y1"]),
                x2=int(xyxy_dict["x2"]),
                y2=int(xyxy_dict["y2"]),
            )
            detection = Detection(
                bbox=BoundingBox(xyxy=xyxy),
                confidence=float(det_dict["confidence"]),
                class_id=int(det_dict["class_id"]),
            )
            detections.append(detection)

        # Auto-increment frame number for next call
        self.frame_number += 1

        logger.debug(
            f"FakeModel: Frame {self.frame_number - 1} -> {len(detections)} detections"
        )
        return detections

    def format_inference_results(self, results):
        """Not used in FakeModel, returns None for compatibility."""
        return None, None, None

    def render_result(
        self,
        image: Image,
        detections: list[Detection],
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> Image:
        """
        Visualize the detection results (same as Model class).

        Args:
            image: Image object
            detections: List of Detection objects
            color: BGR color tuple for bounding boxes
            thickness: Thickness of bounding box lines

        Returns:
            Image: Rendered image with detection boxes drawn
        """
        result_image = image.image.copy()

        for detection in detections:
            xyxy = detection.bbox.xyxy
            x1, y1, x2, y2 = int(xyxy.x1), int(xyxy.y1), int(xyxy.x2), int(xyxy.y2)
            conf = detection.confidence
            class_id = detection.class_id

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label with confidence
            label = f"Class {class_id}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return Image(image=result_image, rgb_bgr=image.rgb_bgr)

    def reset(self):
        """Reset frame_number to 0."""
        self.frame_number = 0
        logger.debug("FakeModel: Reset frame_number to 0")


class FakeDetector:
    """
    Fake detector service class that uses FakeModel for testing.

    Reads precomputed detections from JSON file instead of running model inference.
    """

    def __init__(self, precomputed_results_json_path: Path | str):
        """
        Initialize FakeDetector with precomputed results JSON file.

        Args:
            precomputed_results_json_path: Path to JSON file with precomputed detections
        """
        self.precomputed_results_json_path = Path(precomputed_results_json_path)
        self._model: FakeModel | None = None
        logger.info(
            f"Initialized FakeDetector with {self.precomputed_results_json_path}"
        )

    @property
    def model(self) -> FakeModel:
        """Get the FakeModel instance (lazy initialization)."""
        if self._model is None:
            self._model = FakeModel(self.precomputed_results_json_path)
        return self._model

    def detect(self, image: Image) -> list[Detection]:
        """
        Detect objects by reading from precomputed results.

        Args:
            image: Image object (Pydantic validated)

        Returns:
            List[Detection]: List of Detection objects from precomputed results
        """
        return self.model.predict(image)

    def render_result(
        self,
        image: Image,
        detections: list[Detection],
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> Image:
        """
        Render detection results on image.

        Args:
            image: Image object (Pydantic validated)
            detections: List of Detection objects (Pydantic validated)
            color: BGR color tuple for bounding boxes
            thickness: Thickness of bounding box lines

        Returns:
            Image: Rendered image with detection boxes drawn
        """
        return self.model.render_result(image, detections, color, thickness)

    def reset(self):
        """Reset frame counter to 0."""
        if self._model is not None:
            self._model.reset()
