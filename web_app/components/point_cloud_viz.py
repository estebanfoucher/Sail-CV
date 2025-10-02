"""
Point cloud visualization components for MVS web application
"""

import gradio as gr
from typing import Optional, Tuple
import os
from pathlib import Path


class PointCloudVisualizer:
    """Factory class for creating point cloud visualization components"""
    
    @staticmethod
    def create_point_cloud_viewer() -> Tuple[gr.Model3D, gr.Textbox]:
        """
        Create point cloud visualization components
        
        Returns:
            tuple: (model3d_viewer, viz_status)
        """
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📊 3D Point Cloud Visualization")
                
                # Interactive 3D Model viewer
                model3d_viewer = gr.Model3D(
                    label="🎮 Interactive 3D Point Cloud Viewer",
                    height=700,
                    interactive=True,
                    visible=False
                )
                
                # Visualization status
                viz_status = gr.Textbox(
                    label="Visualization Status",
                    value="No point cloud loaded",
                    interactive=False,
                    lines=2
                )
        
        return model3d_viewer, viz_status
    
    @staticmethod
    def update_point_cloud_display(ply_file_path: Optional[str]) -> Tuple[Optional[str], str]:
        """
        Update the point cloud display with new file
        
        Args:
            ply_file_path: Path to the PLY file (we'll convert to OBJ for better compatibility)
            
        Returns:
            tuple: (obj_file_path, status_message)
        """
        if ply_file_path is None or not os.path.exists(ply_file_path):
            return None, "No point cloud file available"
        
        try:
            # Convert PLY path to OBJ path (they should be generated together)
            obj_file_path = ply_file_path.replace('.ply', '.obj')
            
            if os.path.exists(obj_file_path):
                # Get file size for status
                file_size = os.path.getsize(obj_file_path) / 1024  # KB
                point_count = PointCloudVisualizer._count_points_in_obj(obj_file_path)
                
                status_msg = f"✅ Point cloud loaded: {point_count} points ({file_size:.1f} KB)"
                return obj_file_path, status_msg
            else:
                return None, f"❌ OBJ file not found: {obj_file_path}"
                
        except Exception as e:
            return None, f"❌ Error loading point cloud: {str(e)}"
    
    @staticmethod
    def _count_points_in_obj(obj_file_path: str) -> int:
        """
        Count the number of vertices in an OBJ file
        
        Args:
            obj_file_path: Path to the OBJ file
            
        Returns:
            int: Number of vertices
        """
        try:
            with open(obj_file_path, 'r') as f:
                count = 0
                for line in f:
                    if line.strip().startswith('v '):  # Vertex line
                        count += 1
            return count
        except Exception:
            return 0
