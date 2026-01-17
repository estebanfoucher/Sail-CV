import cv2
import torch

from models import BoundingBox, Detection, Image, ModelSpecs


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
        # Convert Image to numpy array for model inference
        # Models typically expect BGR format
        image_array = image.to_bgr()

        # Run inference (verbose=False to suppress output)
        results = self.model(image_array, verbose=False)

        # Format results
        bboxes, confidences, class_ids = self.format_inference_results(results)

        # Handle None case (no detections)
        if bboxes is None or len(bboxes) == 0:
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
