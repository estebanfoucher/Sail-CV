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
        Update the point cloud display with new file and camera pyramids

        Args:
            ply_file_path: Path to the PLY file (we'll convert to OBJ for better compatibility)

        Returns:
            tuple: (combined_obj_file_path, status_message)
        """
        if ply_file_path is None or not os.path.exists(ply_file_path):
            return None, "No point cloud file available"

        try:
            # Get the directory and base name
            ply_dir = os.path.dirname(ply_file_path)
            base_name = os.path.basename(ply_file_path).replace('.ply', '')

            # Convert PLY path to OBJ path (they should be generated together)
            point_cloud_obj = ply_file_path.replace('.ply', '.obj')

            if not os.path.exists(point_cloud_obj):
                return None, f"❌ Point cloud OBJ file not found: {point_cloud_obj}"

            # Look for camera pyramid OBJ files
            camera_obj_files = []
            print(f"Looking for camera pyramid files in: {ply_dir}")
            print(f"Base name: {base_name}")
            if os.path.exists(ply_dir):
                all_files = os.listdir(ply_dir)
                print(f"All files in directory: {all_files}")

                # Look for camera pyramid subdirectory
                # Extract frame number from base_name (e.g., "point_cloud_frame_26" -> "frame_26")
                frame_name = base_name.replace("point_cloud_", "")
                camera_pyramid_dir = os.path.join(ply_dir, f"camera_pyramids_{frame_name}")
                print(f"Looking for camera pyramid directory: {camera_pyramid_dir}")

                if os.path.exists(camera_pyramid_dir):
                    pyramid_files = os.listdir(camera_pyramid_dir)
                    print(f"Files in camera pyramid directory: {pyramid_files}")
                    # Look for camera pyramid files with the correct naming pattern
                    for file in pyramid_files:
                        if (file.startswith("camera_1_pyramid") or file.startswith("camera_2_pyramid")) and file.endswith('.obj'):
                            camera_obj_files.append(os.path.join(camera_pyramid_dir, file))
                            print(f"Found camera pyramid OBJ file: {file}")
                else:
                    print(f"Camera pyramid directory not found: {camera_pyramid_dir}")
            print(f"Camera pyramid OBJ files found: {len(camera_obj_files)}")

            # Combine all OBJ files into one
            combined_obj_path = os.path.join(ply_dir, f"combined_{base_name}.obj")
            PointCloudVisualizer._combine_obj_files([point_cloud_obj] + camera_obj_files, combined_obj_path)

            # Get file size and point count for status
            file_size = os.path.getsize(combined_obj_path) / 1024  # KB
            point_count = PointCloudVisualizer._count_points_in_obj(combined_obj_path)

            camera_info = f" + {len(camera_obj_files)} camera pyramids" if camera_obj_files else ""
            status_msg = f"✅ Combined 3D model loaded: {point_count} points{camera_info} ({file_size:.1f} KB)"

            return combined_obj_path, status_msg

        except Exception as e:
            return None, f"❌ Error loading point cloud: {str(e)}"

    @staticmethod
    def _combine_obj_files(obj_file_paths: list, output_path: str) -> None:
        """
        Combine multiple OBJ files into one

        Args:
            obj_file_paths: List of paths to OBJ files to combine
            output_path: Path for the combined OBJ file
        """
        vertex_offset = 0
        texture_offset = 0

        with open(output_path, 'w') as outfile:
            for obj_file in obj_file_paths:
                if not os.path.exists(obj_file):
                    continue

                with open(obj_file, 'r') as infile:
                    for line in infile:
                        line = line.strip()
                        if line.startswith('v '):  # Vertex
                            outfile.write(line + '\n')
                        elif line.startswith('vt '):  # Texture coordinate
                            outfile.write(line + '\n')
                        elif line.startswith('vn '):  # Normal
                            outfile.write(line + '\n')
                        elif line.startswith('f '):  # Face
                            # Adjust face indices by adding vertex_offset
                            parts = line.split()
                            adjusted_parts = [parts[0]]  # Keep 'f'
                            for part in parts[1:]:
                                if '/' in part:
                                    # Handle v/vt/vn format
                                    v, vt, vn = part.split('/')
                                    new_v = str(int(v) + vertex_offset)
                                    new_vt = str(int(vt) + texture_offset) if vt else ''
                                    new_vn = str(int(vn) + vertex_offset) if vn else ''
                                    adjusted_parts.append('/'.join([new_v, new_vt, new_vn]))
                                else:
                                    # Handle simple vertex index
                                    adjusted_parts.append(str(int(part) + vertex_offset))
                            outfile.write(' '.join(adjusted_parts) + '\n')
                        elif line.startswith('l '):  # Line (for wireframe)
                            # Adjust line indices by adding vertex_offset
                            parts = line.split()
                            adjusted_parts = [parts[0]]  # Keep 'l'
                            for part in parts[1:]:
                                adjusted_parts.append(str(int(part) + vertex_offset))
                            outfile.write(' '.join(adjusted_parts) + '\n')

                # Count vertices and texture coordinates in this file for offset
                with open(obj_file, 'r') as infile:
                    for line in infile:
                        if line.strip().startswith('v '):
                            vertex_offset += 1
                        elif line.strip().startswith('vt '):
                            texture_offset += 1

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
