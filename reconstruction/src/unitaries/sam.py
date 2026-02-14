import cv2
import numpy as np
import torch
from ultralytics import FastSAM


class SAM:
    def __init__(self, model_path):
        """
        Initialize FastSAM model

        Args:
            model_path (str): Path to FastSAM model or model name (e.g., "FastSAM-x.pt").
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use FastSAM model - ultralytics will automatically download if needed
        self.model = FastSAM(model_path)

        self.model.to(self.device)

    def predict(self, image, point, label=1, conf=0.4):
        """
        Predict segmentation mask using a single point prompt

        Args:
            image: Input image as numpy array
            point: Point coordinates as (x, y) tuple/list
            label: Label for the point (1 for foreground, 0 for background)
            conf: Confidence threshold for predictions

        Returns:
            dict: Dictionary containing mask, score, and other results
        """
        # Run FastSAM inference
        results = self.model(image, device=self.device, retina_masks=True)

        # Get masks and scores
        masks = results[0].masks.data.cpu().numpy()  # shape [N, H, W]
        scores = results[0].boxes.conf.cpu().numpy()  # confidence scores, shape [N]

        # Find best mask index
        best_idx = scores.argmax()
        best_mask = masks[best_idx]

        # Check if the point is inside the best mask
        point_x, point_y = int(point[0]), int(point[1])

        if (
            point_y < best_mask.shape[0]
            and point_x < best_mask.shape[1]
            and best_mask[point_y, point_x] > 0
        ):
            selected_mask = best_mask
            selected_score = scores[best_idx]
        else:
            # Point not in best mask, find alternative
            selected_mask = None
            selected_score = 0
            for i, mask in enumerate(masks):
                if (
                    point_y < mask.shape[0]
                    and point_x < mask.shape[1]
                    and mask[point_y, point_x] > 0
                ) and scores[i] > selected_score:
                    selected_score = scores[i]
                    selected_mask = mask

        return {
            "mask": selected_mask,
            "score": selected_score,
            "point": point,
            "label": label,
            "all_masks": masks,
            "all_scores": scores,
            "image_shape": image.shape,
        }

    def render_result(self, image, mask, points=[]):
        """
        Visualize the segmentation result

        Args:
            image: Original image as numpy array
            mask: Segmentation mask
            points: Points to render

        Returns:
            numpy array: Image with mask overlay
        """
        if mask is None:
            return image

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green color for mask

        for point in points:
            cv2.circle(colored_mask, point, 5, (255, 0, 0), -1)

        # Blend image and mask only where mask is not 0
        result = image * 0.5 + colored_mask * 0.5
        result[mask == 0] = image[mask == 0]

        return result.astype(np.uint8)
