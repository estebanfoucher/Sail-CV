#!/usr/bin/env python3
"""
MASt3R Modular Pipeline

Orchestrates the complete MASt3R pipeline including inference, post-processing, and saving.
This module provides a clean interface to run the full MASt3R workflow with proper
error handling and logging.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

# Import the three modules
from mast3r_inference_core import MASt3RInferenceEngine
from mast3r_postprocess import process_mast3r_results
from mast3r_saver import save_mast3r_results

def run_mast3r_pipeline(
    image1_path: str,
    image2_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    camera_poses: Optional[list] = None,
    subsample: int = 8,
    inference_engine: Optional[MASt3RInferenceEngine] = None
) -> Dict:
    """
    Run the complete MASt3R pipeline using modular components.
    
    This function orchestrates the full MASt3R workflow including:
    1. Core inference using the MASt3R model
    2. Post-processing of raw inference results
    3. Saving results in various formats
    
    Args:
        image1_path: Path to the first input image
        image2_path: Path to the second input image
        output_dir: Directory where all results will be saved
        model_path: Optional path to model checkpoint. If None, uses default model.
    
    Returns:
        Dict containing processed results and summary statistics
        
    Raises:
        FileNotFoundError: If input images don't exist
        RuntimeError: If any pipeline step fails
    """
    logger.info("Starting MASt3R Modular Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Core Inference
    logger.info("Step 1: Running MASt3R Inference")
    # Reuse provided engine if available to ensure a single CUDA load
    if inference_engine is None:
        inference_engine = MASt3RInferenceEngine(model_path)
    
    # Load images once
    images = inference_engine.load_images(image1_path, image2_path)
    
    # Save inference images using loaded images
    logger.info(f"Saving inference images to {output_dir}/inference_images/")
    inference_engine.save_inference_images(image1_path, image2_path, output_dir)
    
    # Run inference using loaded images
    raw_data = inference_engine.run_inference_with_images(images, camera_poses, subsample)
    logger.success(f"Inference completed: {raw_data['num_matches']} matches found")
    
    # Step 2: Post-Processing
    logger.info("Step 2: Post-Processing Results")
    processed_data = process_mast3r_results(raw_data, image1_path, image2_path)
    logger.success("Post-processing completed")
    
    # Step 3: Save Results
    logger.info("Step 3: Saving Results")
    results = save_mast3r_results(processed_data, image1_path, image2_path, output_dir)
    logger.success(f"All results saved to {output_dir}/")
    
    logger.info("MASt3R Pipeline Completed Successfully!")
    logger.info("=" * 50)
    
    return results

def validate_inputs(image1_path: str, image2_path: str, output_dir: str) -> None:
    """
    Validate input parameters for the MASt3R pipeline.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        output_dir: Output directory path
        
    Raises:
        FileNotFoundError: If input images don't exist
        ValueError: If paths are invalid
    """
    # Validate image paths
    if not Path(image1_path).exists():
        raise FileNotFoundError(f"Image 1 not found: {image1_path}")
    
    if not Path(image2_path).exists():
        raise FileNotFoundError(f"Image 2 not found: {image2_path}")
    
    # Validate output directory (create if doesn't exist)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")


def print_usage() -> None:
    """Print usage information for the script."""
    logger.info("Usage: python run_mast3r.py <image1> <image2> <output_dir> [model_path] [camera_poses_json] [subsample]")
    logger.info("Example: python run_mast3r.py /data/image1.jpg /data/image2.jpg /output/")
    logger.info("Example: python run_mast3r.py /data/image1.jpg /data/image2.jpg /output/ /path/to/model.pth")
    logger.info("Example: python run_mast3r.py /data/image1.jpg /data/image2.jpg /output/ /path/to/model.pth /path/to/poses.json")
    logger.info("Example: python run_mast3r.py /data/image1.jpg /data/image2.jpg /output/ /path/to/model.pth /path/to/poses.json 1")
    logger.info("  subsample: 1=dense, 4=medium, 8=sparse (default)")


def print_summary(results: Dict) -> None:
    """Print a formatted summary of the pipeline results."""
    summary = results.get('summary', {})
    logger.info("Final Summary:")
    logger.info(f"  - Matches found: {summary.get('num_matches', 0):,}")
    logger.info(f"  - Camera validation: {'PASS' if summary.get('camera_validation') else 'FAIL'}")
    logger.info(f"  - Point cloud 1: {summary.get('point_cloud_1_shape', 'N/A')}")
    logger.info(f"  - Point cloud 2: {summary.get('point_cloud_2_shape', 'N/A')}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    output_dir = sys.argv[3]
    model_path = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != 'None' else None
    camera_poses_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != 'None' else None
    subsample = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    
    # Load camera poses if provided
    camera_poses = None
    if camera_poses_path:
        import json
        with open(camera_poses_path, 'r') as f:
            poses_data = json.load(f)
            camera_poses = [poses_data['cameras'][0]['matrix_world'], poses_data['cameras'][1]['matrix_world']]
    
    try:
        # Validate inputs
        validate_inputs(image1_path, image2_path, output_dir)
        
        # Run the pipeline
        results = run_mast3r_pipeline(image1_path, image2_path, output_dir, model_path, camera_poses, subsample)
        print_summary(results)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
