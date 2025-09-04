from ultralytics import FastSAM
import torch
import numpy as np
import cv2

class SAM:
    def __init__(self, model_path=None):
        """
        Initialize FastSAM model
        
        Args:
            model_path (str, optional): Path to FastSAM model. If None, uses default model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
       
        # Use default FastSAM model
        self.model = FastSAM('FastSAM-s.pt')
        
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
        masks = results[0].masks.data.cpu().numpy()        # shape [N, H, W]
        scores = results[0].boxes.conf.cpu().numpy()       # confidence scores, shape [N]
        
        # Find best mask index
        best_idx = scores.argmax()
        best_mask = masks[best_idx]
        
        # Check if the point is inside the best mask
        point_x, point_y = int(point[0]), int(point[1])
        
        if (point_y < best_mask.shape[0] and point_x < best_mask.shape[1] and 
            best_mask[point_y, point_x] > 0):
            selected_mask = best_mask
            selected_score = scores[best_idx]
        else:
            # Point not in best mask, find alternative
            selected_mask = None
            selected_score = 0
            for i, mask in enumerate(masks):
                if (point_y < mask.shape[0] and point_x < mask.shape[1] and 
                    mask[point_y, point_x] > 0):
                    if scores[i] > selected_score:
                        selected_score = scores[i]
                        selected_mask = mask
        
        return {
            'mask': selected_mask,
            'score': selected_score,
            'point': point,
            'label': label,
            'all_masks': masks,
            'all_scores': scores,
            'image_shape': image.shape
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
            cv2.circle(colored_mask, point, 5, (0, 0, 255), -1)
        
        # Blend image and mask
        result = image * 0.5 + colored_mask * 0.5
        
        return result.astype(np.uint8)

