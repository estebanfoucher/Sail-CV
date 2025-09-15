#!/usr/bin/env python3
"""
Main orchestration script for MAST3R pipeline.
Runs the complete pipeline: inference -> FastSAM filtering -> calibration conversion -> triangulation.
"""

import os
import sys
import glob
import shutil
import json
from pathlib import Path
from loguru import logger
import time
import numpy as np
import cv2

# Import the modules directly
sys.path.append('/mast3r')
sys.path.append('/mast3r/docker/files')

# Import the functions we need
from run_mast3r import run_mast3r_pipeline
from convert_calibration import main as convert_calibration_main
from triangulate_matches import main as triangulate_matches_main
from image import resize_image
from mast3r_inference_core import MASt3RInferenceEngine
from sam import SAM


def find_images(directory, extensions=('*.jpg', '*.png', '*.jpeg')):
    """Find image files in directory."""
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(images)


def save_resized_frames(image1_path, image2_path, output_dir, size=512):
    """Save resized input frames to output_dir/resized_frame/"""
    logger.info(f"Saving resized frames to {output_dir}/resized_frame/")
    
    # Create resized_frame directory
    resized_dir = os.path.join(output_dir, "resized_frame")
    os.makedirs(resized_dir, exist_ok=True)
    
    # Resize and save first image
    resized_img1 = resize_image(image1_path, size)
    frame1_path = os.path.join(resized_dir, "frame_1.png")
    resized_img1.save(frame1_path)
    logger.info(f"Saved resized frame 1: {frame1_path} ({resized_img1.size})")
    
    # Resize and save second image
    resized_img2 = resize_image(image2_path, size)
    frame2_path = os.path.join(resized_dir, "frame_2.png")
    resized_img2.save(frame2_path)
    logger.info(f"Saved resized frame 2: {frame2_path} ({resized_img2.size})")
        

def run_mast3r_inference(image1_path, image2_path, output_dir, model_path, inference_engine=None):
    """Run MAST3R inference on image pair."""
    logger.info(f"Running MAST3R inference on: {image1_path} and {image2_path}")
    
    try:
        logger.info("DEBUG: About to call run_mast3r_pipeline")
        # Call the pipeline function directly
        results = run_mast3r_pipeline(
            image1_path,
            image2_path,
            output_dir,
            model_path,
            subsample=1,
            inference_engine=inference_engine,
        )
        logger.info("DEBUG: run_mast3r_pipeline returned successfully")
        logger.info("MAST3R inference completed successfully")
        return True
    except Exception as e:
        logger.error(f"MAST3R inference failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def run_calibration_conversion(input_calibration, output_calibration):
    """Run calibration parameter conversion."""
    logger.info("Converting calibration parameters...")
    
    # Set up sys.argv for the convert_calibration module
    original_argv = sys.argv.copy()
    sys.argv = ['convert_calibration.py', input_calibration, output_calibration]
    
    try:
        convert_calibration_main()
        logger.info("Calibration conversion completed successfully")
        return True
    except Exception as e:
        logger.error(f"Calibration conversion failed: {e}")
        return False
    finally:
        sys.argv = original_argv


def run_fastsam_filtering(resized_frame_path, output_dir):
    """Run FastSAM on resized frame with center point prompt and filter pairs.
    
    Args:
        resized_frame_path: Path to resized frame_1.png
        output_dir: Output directory for saving results
        
    Returns:
        tuple: (filtered_matches_file, mask_path, rendered_sam_path) or (None, None, None) if failed
    """
    logger.info("=== STARTING FastSAM filtering ===")
    logger.info(f"Resized frame path: {resized_frame_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        logger.info("Initializing FastSAM model...")
        # Initialize FastSAM
        sam = SAM()
        logger.info("FastSAM model initialized successfully")
        
        # Load the resized image
        image = cv2.imread(resized_frame_path)
        if image is None:
            logger.error(f"Could not load image: {resized_frame_path}")
            return None, None, None
            
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions for center point
        height, width = image_rgb.shape[:2]
        center_point = (width // 2, height // 2)
        
        logger.info(f"Running FastSAM with center point prompt: {center_point}")
        
        # Run FastSAM prediction
        result = sam.predict(image_rgb, center_point, label=1, conf=0.4)
        
        if result['mask'] is None:
            logger.warning("FastSAM did not find a valid mask, skipping filtering")
            return None, None, None
            
        mask = result['mask']
        logger.info(f"FastSAM mask found with score: {result['score']:.3f}")
        
        # Save the mask
        mask_dir = os.path.join(output_dir, "fastsam")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, "mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        # Render SAM result and save
        rendered_sam = sam.render_result(image_rgb, mask, [center_point])
        rendered_sam_path = os.path.join(mask_dir, "rendered_sam.png")
        cv2.imwrite(rendered_sam_path, cv2.cvtColor(rendered_sam, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved FastSAM mask to: {mask_path}")
        logger.info(f"Saved rendered SAM result to: {rendered_sam_path}")
        
        # Filter pixel pairs based on mask
        matches_file = os.path.join(output_dir, "matches", "pixel_pairs.json")
        filtered_matches_file = os.path.join(mask_dir, "filtered_pixel_pairs.json")
        
        if not os.path.exists(matches_file):
            logger.error(f"Matches file not found: {matches_file}")
            return None, None, None
            
        # Load original matches
        with open(matches_file, 'r') as f:
            matches_data = json.load(f)
        
        pixel_pairs = matches_data['pixel_pairs']
        filtered_pairs = []
        
        # Filter pairs based on mask
        for pair in pixel_pairs:
            x1, y1 = pair['image1_pixel']
            x2, y2 = pair['image2_pixel']
            
            # Check if both points are within image bounds and in mask
            if (0 <= x1 < width and 0 <= y1 < height and 
                0 <= x2 < width and 0 <= y2 < height):
                
                # Check if point 1 is in mask (we only filter based on frame_1)
                if mask[int(y1), int(x1)] > 0:
                    filtered_pairs.append(pair)
        
        # Save filtered matches
        filtered_data = {
            'pixel_pairs': filtered_pairs,
            'num_pairs': len(filtered_pairs),
            'format': 'dense_matching_results_filtered_by_fastsam',
            'description': f'Pixel correspondences filtered by FastSAM mask (original: {len(pixel_pairs)}, filtered: {len(filtered_pairs)})'
        }
        
        with open(filtered_matches_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        logger.info(f"Filtered {len(pixel_pairs)} pairs to {len(filtered_pairs)} pairs using FastSAM mask")
        logger.info(f"Saved filtered matches to: {filtered_matches_file}")
        
        return filtered_matches_file, mask_path, rendered_sam_path
        
    except Exception as e:
        logger.error(f"FastSAM filtering failed with exception: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None


def run_triangulation(matches_file, calibration_file, output_file_path, resized_frame_path):
    """Run triangulation of matches, saving into the given output directory.

    The output PLY will be written as `<output_dir>/triangulated_points.ply`.
    """
    logger.info("Running triangulation...")
    
    # Set up sys.argv for the triangulate_matches module
    original_argv = sys.argv.copy()
    sys.argv = ['triangulate_matches.py', matches_file, calibration_file, output_file_path, resized_frame_path]
    
    start_time = time.time()
    try:
        triangulate_matches_main()
        elapsed = time.time() - start_time
        logger.info(f"Triangulation completed successfully in {elapsed:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Triangulation failed: {e}")
        return False
    finally:
        sys.argv = original_argv


def main():
    """Main pipeline execution."""
    logger.info("Starting MAST3R pipeline...")
    
    # Default paths
    model = os.environ.get('MODEL', 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    model_path = f"/mast3r/checkpoints/{model}"
    test_dir = "/mast3r/tmp/mast3r_test"
    output_dir = "/mast3r/tmp/mast3r_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare pairs
    pairs_dir = os.path.join(test_dir, "pairs")
    if not os.path.isdir(pairs_dir):
        logger.error(f"Pairs directory not found: {pairs_dir}")
        sys.exit(1)
    pair_names = sorted([d for d in os.listdir(pairs_dir) if os.path.isdir(os.path.join(pairs_dir, d))])
    if not pair_names:
        logger.error(f"No pair subfolders found in {pairs_dir}")
        sys.exit(1)

    # Step 0: Convert calibration parameters once (resized extrinsics stay the same)
    input_calibration = os.path.join(test_dir, "extrinsics_calibration.json")
    output_calibration = os.path.join(output_dir, "extrinsics_calibration_512x288.json")
    run_calibration_conversion(input_calibration, output_calibration)

    # Prepare common triangulated output folder
    triangulated_root = os.path.join(output_dir, "triangulated_point_clouds")
    os.makedirs(triangulated_root, exist_ok=True)

    # Create a single inference engine (loads model to CUDA once)
    logger.info("Initializing shared MASt3R inference engine (single CUDA load)...")
    shared_engine = MASt3RInferenceEngine(model_path)
    # Defer actual model load to first inference to preserve behavior; it will load once

    # Iterate over each pair folder
    for pair_name in pair_names:
        # time the pair processing
        start_time = time.time()
        logger.info(f"Processing pair: {pair_name}")
        pair_dir = os.path.join(pairs_dir, pair_name)
        
        # Find frames inside the pair directory
        img1_candidates = sorted(glob.glob(os.path.join(pair_dir, "frame_1_*.png")))
        img2_candidates = sorted(glob.glob(os.path.join(pair_dir, "frame_2_*.png")))
        if not img1_candidates or not img2_candidates:
            logger.error(f"Missing frames in {pair_dir}")
            continue
        image1_path = img1_candidates[0]
        image2_path = img2_candidates[0]

        # Per-pair output directory
        pair_output_dir = os.path.join(output_dir, "pairs", pair_name)
        os.makedirs(pair_output_dir, exist_ok=True)

        # Save resized frames into the per-pair output
        save_resized_frames(image1_path, image2_path, pair_output_dir, size=512)

        # Run inference for this pair using the shared engine
        logger.info(f"DEBUG: About to call run_mast3r_inference for pair {pair_name}")
        inference_result = run_mast3r_inference(image1_path, image2_path, pair_output_dir, model_path, inference_engine=shared_engine)
        logger.info(f"DEBUG: run_mast3r_inference returned: {inference_result}")
        
        # Run FastSAM filtering on resized frame to filter pairs
        resized_frame_path = os.path.join(pair_output_dir, "resized_frame", "frame_1.png")
        logger.info(f"DEBUG: About to call FastSAM filtering for pair {pair_name}")
        logger.info(f"DEBUG: resized_frame_path exists: {os.path.exists(resized_frame_path)}")
        logger.info(f"DEBUG: pair_output_dir: {pair_output_dir}")
        filtered_matches_file, mask_path, rendered_sam_path = run_fastsam_filtering(resized_frame_path, pair_output_dir)
        logger.info(f"DEBUG: FastSAM filtering completed for pair {pair_name}")
        
        # Use filtered matches if available, otherwise fall back to original matches
        if filtered_matches_file is not None:
            matches_file = filtered_matches_file
            logger.info(f"Using FastSAM filtered matches: {matches_file}")
        else:
            matches_file = os.path.join(pair_output_dir, "matches", "pixel_pairs.json")
            logger.info(f"Using original matches (FastSAM filtering failed): {matches_file}")

        # Triangulate using filtered or original matches
        ply_path = os.path.join(triangulated_root, f"{pair_name}.ply")
        run_triangulation(matches_file, output_calibration, ply_path, resized_frame_path)



        # time the pair processing
        end_time = time.time()
        logger.info(f"Pair {pair_name} processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()