#!/usr/bin/env python3
"""
MASt3R Saver Module
Handles file output and project structure creation
"""

import os
import json
import time
import shutil
import numpy as np

# Import utility functions to avoid duplication
from mast3r_utils import (
    save_point_cloud_ply,
    save_pixel_pairs,
    verify_camera_data,
    create_project_structure
)




def save_mast3r_results(processed_data, image1_path, image2_path, output_dir):
    """
    Save all MASt3R results in organized project structure
    
    Args:
        processed_data: Processed results from mast3r_postprocess
        image1_path: Path to first image
        image2_path: Path to second image
        output_dir: Output directory
    
    Returns:
        dict: Summary of saved files
    """
    print(f"\n=== SAVING RESULTS TO {output_dir}/ ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create project structure first
    project_dirs = create_project_structure(output_dir, image1_path, image2_path)
    
    # Extract data
    cameras = processed_data['cameras']
    point_clouds = processed_data['point_clouds']
    matches = processed_data['matches']
    views = processed_data['views']
    
    # Save point clouds as PLY files in point_clouds directory
    save_point_cloud_ply(point_clouds['pts3d_1'], point_clouds['colors_1'], f'{project_dirs["point_clouds"]}/point_cloud_1.ply')
    save_point_cloud_ply(point_clouds['pts3d_2'], point_clouds['colors_2'], f'{project_dirs["point_clouds"]}/point_cloud_2.ply')
    
    # Save pixel pairs in matches directory
    save_pixel_pairs(matches['matches_im0'], matches['matches_im1'], f'{project_dirs["matches"]}/pixel_pairs.json')
    
    # Save cameras in MAST3R-compatible format in cameras directory
    camera_data = cameras['formatted']
    
    with open(f'{project_dirs["cameras"]}/camera_params.json', 'w') as f:
        json.dump(camera_data, f, indent=2)
    
    # Save summary
    results = {
        'summary': {
            'num_matches': matches['num_matches'],
            'camera_validation': cameras['valid'],
            'point_cloud_1_shape': list(point_clouds['pts3d_1'].shape),
            'point_cloud_2_shape': list(point_clouds['pts3d_2'].shape)
        },
        'files_generated': {
            'point_cloud_1_ply': 'point_clouds/point_cloud_1.ply',
            'point_cloud_2_ply': 'point_clouds/point_cloud_2.ply',
            'camera_params_json': 'camera_params.json',
            'pixel_pairs_json': 'matches/pixel_pairs.json'
        }
    }
    
    with open(f'{output_dir}/mast3r_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save full matches in matches directory
    with open(f'{project_dirs["matches"]}/matches_full.json', 'w') as f:
        json.dump({
            'image1_points': matches['matches_im0'].tolist(),
            'image2_points': matches['matches_im1'].tolist()
        }, f, indent=2)
    
    print(f"✅ Results saved successfully!")
    print(f"📁 Project structure created")
    print(f"🎯 Point clouds and camera data ready for use")
    
    return results
