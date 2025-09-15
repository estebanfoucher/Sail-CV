#!/usr/bin/env python3
"""
MASt3R Core Inference Module
Handles model loading and raw inference operations
"""

import torch
import time
import numpy as np
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
import cv2
import os

class MASt3RInferenceEngine:
    """Core inference engine for MASt3R"""
    
    def __init__(self, model_path=None, device=None):
        """Initialize the inference engine"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path or "/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the MASt3R model"""
        print(f"Loading model from: {self.model_path}")
        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(self.device)
        return self.model
    
    def load_images(self, image1_path, image2_path, size=512):
        """Load and preprocess images"""
        print(f"Loading images: {image1_path}, {image2_path}")
        images = load_images([image1_path, image2_path], size=size)
        return images
    
    def save_inference_images(self, image1_path, image2_path, output_dir):
        """Save inference images"""
        print(f"Saving inference images to {output_dir}/inference_images/")
        images = self.load_images(image1_path, image2_path)
        # save frame_1 and frame_2 to output_dir/inference_images/
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "inference_images"), exist_ok=True)
        
        # Convert tensors to numpy arrays and handle the format properly
        img1_np = images[0]['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img2_np = images[1]['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Convert from [0,1] range to [0,255] and ensure uint8
        img1_np = (img1_np * 255).astype(np.uint8)
        img2_np = (img2_np * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(output_dir, "inference_images", "frame_1.png"), img1_np)
        cv2.imwrite(os.path.join(output_dir, "inference_images", "frame_2.png"), img2_np)
    
    def run_inference(self, images, camera_poses=None):
        """Run MASt3R inference on image pair"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Running MASt3R inference...")
        start_time = time.time()
        
        # Add camera poses if provided
        if camera_poses is not None:
            print("Using camera pose priors for faster inference")
            # Convert to torch tensors and add to images
            for i, pose in enumerate(camera_poses):
                if pose is not None:
                    images[i]['cams2world'] = torch.tensor(pose, dtype=torch.float32, device=self.device)
        
        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)
        
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        
        return output
    
    def extract_raw_data(self, output, subsample=8):
        """Extract raw data from inference output"""
        # Extract results
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        # Get descriptors and matches
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=subsample, device=self.device, dist='dot', block_size=2**13
        )
        
        # Get 3D points from the available keys
        pts3d_1 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
        pts3d_2 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()
        
        # Get image colors for point cloud coloring
        img1_colors = view1['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img2_colors = view2['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Debug: Print available keys
        print("Available keys in pred1:", list(pred1.keys()))
        print("Available keys in pred2:", list(pred2.keys()))
        
        return {
            'view1': view1,
            'view2': view2,
            'pred1': pred1,
            'pred2': pred2,
            'matches_im0': matches_im0,
            'matches_im1': matches_im1,
            'pts3d_1': pts3d_1,
            'pts3d_2': pts3d_2,
            'img1_colors': img1_colors,
            'img2_colors': img2_colors,
            'num_matches': len(matches_im0)
        }
    
    def run_inference_with_images(self, images, camera_poses=None, subsample=8):
        """Run inference using already loaded images"""
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
                
        # Run inference
        output = self.run_inference(images, camera_poses)
        
        # Extract raw data
        raw_data = self.extract_raw_data(output, subsample)
        
        return raw_data

    def run_full_inference(self, image1_path, image2_path, camera_poses=None, subsample=8):
        """Run complete inference pipeline"""
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Load images
        images = self.load_images(image1_path, image2_path)
                
        # Run inference
        output = self.run_inference(images, camera_poses)
        
        # Extract raw data
        raw_data = self.extract_raw_data(output, subsample)
        
        return raw_data
