import cv2
import numpy as np
import torch


class TellTaleDetector:
    def __init__(self, model_path, architecture):
        """
        Initialize RTDETR model for tell-tale detection

        Args:
            model_path (str): Path to model file
            architecture (str): Architecture of the model (yolo or rt-detr)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load RTDETR model
        if architecture == "rt-detr":
            from ultralytics import RTDETR

            self.model = RTDETR(model_path)
        elif architecture == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(model_path)
        else:
            raise ValueError(f"Invalid architecture: {architecture}")
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

    def predict(self, image, conf=0.1):
        """
        Predict tell-tale objects in the image

        Args:
            image: Input image as numpy array
            conf: Confidence threshold for predictions (not used, kept for compatibility)

        Returns:
            dict: Dictionary containing detections, boxes, scores, and class_ids
        """
        # Run inference exactly like the tracker: model(frame)
        results = self.model(image)

        # Use the same format_inference_results as the tracker
        bboxes, confidences, class_ids = self.format_inference_results(results)

        # Handle None case (no detections)
        if bboxes is None:
            bboxes = np.array([])
            confidences = np.array([])
            class_ids = np.array([])

        return {
            "boxes": bboxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "image_shape": image.shape,
            "num_detections": len(bboxes),
        }

    def render_result(self, image, detections, color=(255, 0, 0), thickness=2):
        """
        Visualize the detection results

        Args:
            image: Original image as numpy array
            detections: Detection results from predict()
            color: BGR color for bounding boxes
            thickness: Thickness of bounding box lines

        Returns:
            numpy array: Image with detection boxes drawn
        """
        result_image = image.copy()

        boxes = detections["boxes"]
        confidences = detections["confidences"]
        class_ids = detections["class_ids"]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            conf = confidences[i]
            class_id = int(class_ids[i])

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

        return result_image
