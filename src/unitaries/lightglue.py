"""
Simplified LightGlue Feature Matcher for MVS Application

This module provides a streamlined wrapper around LightGlue for feature matching between image pairs.
"""

import torch
import numpy as np
import cv2
from typing import Dict, Union
from loguru import logger

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, match_pair
from lightglue.utils import load_image, rbd


class LightGlueFeatureMatcher:
    """
    A simplified wrapper class for LightGlue feature matching.
    
    This class provides a simple interface for matching features between stereo image pairs
    and returns the matched points in a format suitable for stereo vision processing.
    """
    
    def __init__(
        self, 
        feature_extractor: str = "superpoint",
        max_num_keypoints: int = 2048,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **lightglue_kwargs
    ):
        """
        Initialize the LightGlue feature matcher.
        
        Args:
            feature_extractor: Type of feature extractor to use ('superpoint', 'disk', 'sift', 'aliked', 'doghardnet')
            max_num_keypoints: Maximum number of keypoints to extract per image
            device: Device to run inference on ('cuda' or 'cpu')
            **lightglue_kwargs: Additional arguments for LightGlue configuration
        """
        self.device = device
        self.feature_extractor_name = feature_extractor
        self.max_num_keypoints = max_num_keypoints
        
        # Initialize feature extractor
        self.extractor = self._create_extractor(feature_extractor, max_num_keypoints)
        
        # Initialize LightGlue matcher
        self.matcher = self._create_matcher(feature_extractor, **lightglue_kwargs)
        
        logger.info(f"LightGlueFeatureMatcher initialized with {feature_extractor} extractor on {device}")
    
    def _create_extractor(self, extractor_type: str, max_keypoints: int):
        """Create the feature extractor based on the specified type."""
        extractors = {
            "superpoint": SuperPoint,
            "disk": DISK,
            "sift": SIFT,
            "aliked": ALIKED,
            "doghardnet": DoGHardNet
        }
        
        if extractor_type not in extractors:
            raise ValueError(f"Unknown feature extractor: {extractor_type}. "
                           f"Available options: {list(extractors.keys())}")
        
        # Follow official pattern: SuperPoint(max_num_keypoints=2048).eval().cuda()
        extractor = extractors[extractor_type](max_num_keypoints=max_keypoints)
        return extractor.eval().to(self.device)
    
    def _create_matcher(self, extractor_type: str, **kwargs):
        """Create the LightGlue matcher following official pattern."""
        default_config = {
            "features": extractor_type,
            "depth_confidence": 0.95,
            "width_confidence": 0.99,
            "filter_threshold": 0.1,
            "flash": True,
            "mp": False
        }
        
        # Update with user-provided kwargs
        default_config.update(kwargs)
        
        # Follow official pattern: LightGlue(features='superpoint').eval().cuda()
        matcher = LightGlue(**default_config)
        return matcher.eval().to(self.device)
    
    def _preprocess_image(self, image: Union[np.ndarray, str]) -> torch.Tensor:
        """
        Preprocess image for LightGlue inference.
        
        Args:
            image: Input image as numpy array (H, W, C) or path to image file
            
        Returns:
            Preprocessed image tensor (3, H, W) normalized to [0, 1]
        """
        if isinstance(image, str):
            # Load image from file
            image_tensor = load_image(image).to(self.device)
        else:
            # Convert numpy array to tensor
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB conversion
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def match_images(
        self, 
        image1: Union[np.ndarray, str], 
        image2: Union[np.ndarray, str]
    ) -> Dict[str, np.ndarray]:
        """
        Match features between two images using the match_pair convenience function.
        
        Args:
            image1: First image (numpy array or path to image file)
            image2: Second image (numpy array or path to image file)
            
        Returns:
            Dictionary containing:
                - 'matches': Array of shape (N, 2) with matched point indices
                - 'keypoints1': Keypoints from first image (N1, 2)
                - 'keypoints2': Keypoints from second image (N2, 2)
                - 'matched_points1': Matched points from first image (N, 2)
                - 'matched_points2': Matched points from second image (N, 2)
                - 'match_confidence': Confidence scores for matches (N,)
        """
        # Preprocess images
        img1_tensor = self._preprocess_image(image1)
        img2_tensor = self._preprocess_image(image2)
        
        # Use match_pair convenience function
        with torch.no_grad():
            feats1, feats2, matches = match_pair(self.extractor, self.matcher, img1_tensor, img2_tensor)
        
        # Extract matched points
        match_indices = matches['matches']  # Shape: (N, 2)
        # Handle different LightGlue versions - some use 'matching_scores', others use 'scores'
        match_confidence = matches.get('matching_scores', matches.get('scores', torch.ones(len(match_indices))))
        
        # Get keypoints
        keypoints1 = feats1['keypoints'].cpu().numpy()  # Shape: (N1, 2)
        keypoints2 = feats2['keypoints'].cpu().numpy()  # Shape: (N2, 2)
        
        # Get matched points
        matched_points1 = keypoints1[match_indices[:, 0]]  # Shape: (N, 2)
        matched_points2 = keypoints2[match_indices[:, 1]]  # Shape: (N, 2)
        
        result = {
            'matches': match_indices.cpu().numpy(),
            'keypoints1': keypoints1,
            'keypoints2': keypoints2,
            'matched_points1': matched_points1,
            'matched_points2': matched_points2,
            'match_confidence': match_confidence.cpu().numpy()
        }
        
        logger.info(f"Found {len(matched_points1)} matches between images")
        return result
    
    def visualize_matches(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray, 
        matches_result: Dict[str, np.ndarray],
        max_matches: int = 100
    ) -> np.ndarray:
        """
        Visualize matched features between two images.
        
        Args:
            image1: First image (numpy array)
            image2: Second image (numpy array)
            matches_result: Result from match_images method
            max_matches: Maximum number of matches to visualize
            
        Returns:
            Visualization image with matched features
        """
        matched_points1 = matches_result['matched_points1']
        matched_points2 = matches_result['matched_points2']
        match_confidence = matches_result['match_confidence']
        
        # Sort matches by confidence and take top matches
        sorted_indices = np.argsort(match_confidence)[::-1]
        top_matches = min(max_matches, len(sorted_indices))
        
        # Create visualization
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Create side-by-side visualization
        vis_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_image[:h1, :w1] = image1
        vis_image[:h2, w1:w1+w2] = image2
        
        # Draw matches
        colors = np.random.randint(0, 255, (top_matches, 3), dtype=np.uint8)
        
        for i in range(top_matches):
            idx = sorted_indices[i]
            pt1 = matched_points1[idx].astype(int)
            pt2 = matched_points2[idx].astype(int)
            pt2[0] += w1  # Offset for second image
            
            color = tuple(map(int, colors[i]))
            
            # Draw points
            cv2.circle(vis_image, tuple(pt1), 3, color, -1)
            cv2.circle(vis_image, tuple(pt2), 3, color, -1)
            
            # Draw line
            cv2.line(vis_image, tuple(pt1), tuple(pt2), color, 1)
        
        return vis_image
    
    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        if hasattr(self, 'extractor'):
            del self.extractor
        if hasattr(self, 'matcher'):
            del self.matcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()