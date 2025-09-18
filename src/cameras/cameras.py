#!/usr/bin/env python3
"""
Camera class for 3D pyramid representation and CloudCompare export.

This module provides a Camera class that represents a camera with focal length,
image, 3D position, and orientation. It can generate 3D pyramid representations
with point cloud bases and wireframe edges, and export them to CloudCompare format.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
import json
import os


class Camera:
    """
    A Camera class that represents a camera with focal length, image, 3D position, and orientation.
    Provides 3D pyramid representation and CloudCompare export functionality.
    """
    
    def __init__(self, 
                 name: str,
                 position: Union[List[float], np.ndarray],
                 rotation_matrix: Union[List[List[float]], np.ndarray],
                 intrinsics: Union[List[List[float]], np.ndarray],
                 image_size: Tuple[int, int],
                 image_path: Optional[str] = None,
                 focal_length: Optional[float] = None,
                 scale_factor: float = 0.00001):
        """
        Initialize a Camera object.
        
        Args:
            name: Camera identifier
            position: 3D position [x, y, z] in world coordinates (camera center in world frame)
            rotation_matrix: 3x3 rotation matrix representing camera-to-world rotation
                            (transforms points from camera coordinates to world coordinates)
            intrinsics: 3x3 camera intrinsic matrix
            image_size: (width, height) of the image
            image_path: Path to the camera image file
            focal_length: Focal length (if None, extracted from intrinsics)
            scale_factor: Scale factor to convert pixel focal length to 3D units (default: 0.001)
            
        Note:
            The Camera class expects camera-to-world transforms:
            - position: camera center in world coordinates
            - rotation_matrix: R such that P_world = R @ P_camera + position
            This is the inverse of world-to-camera transforms from stereo calibration
        """
        self.name = name
        self.position = np.array(position, dtype=np.float64)
        self.rotation_matrix = np.array(rotation_matrix, dtype=np.float64)
        self.intrinsics = np.array(intrinsics, dtype=np.float64)
        self.image_size = image_size
        self.image_path = image_path
        self.scale_factor = scale_factor
        
        # Extract focal length from intrinsics if not provided
        if focal_length is None:
            self.focal_length_pixels = (self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2.0
        else:
            self.focal_length_pixels = focal_length
            
        # Apply scale factor to get 3D focal length
        self.focal_length = self.focal_length_pixels * self.scale_factor
        
        # Load image if path is provided
        self.image = None
        if image_path and os.path.exists(image_path):
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def get_camera_center(self) -> np.ndarray:
        """Get the camera center (apex) in world coordinates."""
        return self.position
    
    def get_image_plane_corners(self, focal_length: Optional[float] = None) -> np.ndarray:
        """
        Get the 4 corners of the image plane at the specified focal length.
        
        Args:
            focal_length: Focal length (uses self.focal_length if None)
            
        Returns:
            4x3 array of corner coordinates in world space
        """
        if focal_length is None:
            focal_length = self.focal_length
            
        width, height = self.image_size
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        
        # Build pyramid in camera's own referential
        # Image corners in camera coordinates at focal length distance
        corners_cam = np.array([
            [(0 - cx) * focal_length / fx, (0 - cy) * focal_length / fy, focal_length],
            [(width - cx) * focal_length / fx, (0 - cy) * focal_length / fy, focal_length],
            [(width - cx) * focal_length / fx, (height - cy) * focal_length / fy, focal_length],
            [(0 - cx) * focal_length / fx, (height - cy) * focal_length / fy, focal_length]
        ])
        
        # Apply camera-to-world transform: P_world = R @ P_camera + T
        # where R is camera-to-world rotation and T is camera center in world coordinates
        corners_world = []
        for corner in corners_cam:
            world_corner = self.rotation_matrix @ corner + self.position.flatten()
            corners_world.append(world_corner.flatten())
            
        return np.array(corners_world)
    
    def get_pyramid_vertices(self, focal_length: Optional[float] = None) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Generate pyramid mesh vertices and edges (wireframe).
        
        Args:
            focal_length: Focal length (uses self.focal_length if None)
            
        Returns:
            Tuple of (vertices, edges) where vertices is Nx3 and edges is list of edge indices
        """
        if focal_length is None:
            focal_length = self.focal_length
            
        # Get camera center (apex)
        apex = self.get_camera_center()
        
        # Get image plane corners
        corners = self.get_image_plane_corners(focal_length)
        
        # Combine vertices: apex + 4 corners
        vertices = np.vstack([apex.reshape(1, 3), corners])
        
        # Define edges (wireframe)
        edges = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # apex to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # base perimeter
        ]
        
        return vertices, edges
    
    def get_pyramid_with_texture_coords(self, focal_length: Optional[float] = None, pixel_sampling: int = 1, image_sampling: int = 1) -> Tuple[np.ndarray, List[List[int]], List[List[int]], np.ndarray, List[List[int]]]:
        """
        Generate pyramid with wireframe edges and point cloud base (each pixel as a point).
        
        Args:
            focal_length: Focal length (uses self.focal_length if None)
            pixel_sampling: Sample every Nth pixel for wireframe corners (default: 1)
            image_sampling: Sample every Nth pixel for image base (default: 1 for full resolution)
            
        Returns:
            Tuple of (vertices, edges, base_faces, texture_coords, colors)
        """
        if focal_length is None:
            focal_length = self.focal_length
            
        # Get camera center (apex)
        apex = self.get_camera_center()
        
        # Get image plane corners
        corners = self.get_image_plane_corners(focal_length)
        
        # Create point cloud base from image pixels
        base_vertices = []
        base_texture_coords = []
        base_faces = []
        
        if self.image is not None:
            height, width = self.image.shape[:2]
            
            # Sample pixels with high resolution for image base
            rows = height // image_sampling
            cols = width // image_sampling
            
            for row in range(rows):
                for col in range(cols):
                    # Calculate image coordinates
                    i = (rows - 1 - row) * image_sampling  # Start from bottom row
                    j = col * image_sampling
                    
                    # Convert pixel coordinates to UV coordinates
                    u = j / (width - 1)
                    v = i / (height - 1)
                    
                    # Bilinear interpolation between the 4 corners
                    corner_weights = [
                        (1 - u) * (1 - v),  # bottom-left
                        u * (1 - v),        # bottom-right
                        u * v,              # top-right
                        (1 - u) * v         # top-left
                    ]
                    
                    # Interpolate vertex position
                    vertex = np.zeros(3)
                    for k, weight in enumerate(corner_weights):
                        vertex += weight * corners[k]
                    
                    base_vertices.append(vertex)
                    base_texture_coords.append([u, v])
        else:
            # Fallback: use the 4 corners
            base_vertices = corners.tolist()
            base_texture_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        
        # Combine vertices: apex + base vertices
        vertices = np.vstack([apex.reshape(1, 3), np.array(base_vertices)])
        
        # Create wireframe as point clouds (linspaces of 100 points per edge)
        wireframe_vertices = []
        wireframe_colors = []
        
        if self.image is not None and len(base_vertices) >= 4:
            height, width = self.image.shape[:2]
            wireframe_rows = height // pixel_sampling
            wireframe_cols = width // pixel_sampling
            image_rows = height // image_sampling
            image_cols = width // image_sampling
            
            # Find the 4 corner indices in the high-resolution point cloud
            bottom_left = 1  # First base vertex (after apex)
            bottom_right = image_cols  # Last point in first row
            top_right = len(base_vertices)  # Last base vertex
            top_left = len(base_vertices) - image_cols + 1  # First point in last row
            
            # Get apex position
            apex_pos = vertices[0]
            
            # Create linspace points for apex to corner edges
            corner_indices = [bottom_left, bottom_right, top_right, top_left]
            for corner_idx in corner_indices:
                if corner_idx < len(vertices):
                    corner_pos = vertices[corner_idx]
                    for t in np.linspace(0, 1, 100):
                        point = apex_pos + t * (corner_pos - apex_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
            
            # Create linspace points for base perimeter edges
            # Bottom edge (left to right)
            for j in range(image_cols - 1):
                if bottom_left + j + 1 < len(vertices):
                    start_pos = vertices[bottom_left + j]
                    end_pos = vertices[bottom_left + j + 1]
                    for t in np.linspace(0, 1, 100):
                        point = start_pos + t * (end_pos - start_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
            
            # Top edge (left to right)
            for j in range(image_cols - 1):
                if top_left + j + 1 < len(vertices):
                    start_pos = vertices[top_left + j]
                    end_pos = vertices[top_left + j + 1]
                    for t in np.linspace(0, 1, 100):
                        point = start_pos + t * (end_pos - start_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
            
            # Left edge (bottom to top)
            for i in range(image_rows - 1):
                if bottom_left + (i + 1) * image_cols < len(vertices):
                    start_pos = vertices[bottom_left + i * image_cols]
                    end_pos = vertices[bottom_left + (i + 1) * image_cols]
                    for t in np.linspace(0, 1, 100):
                        point = start_pos + t * (end_pos - start_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
            
            # Right edge (bottom to top)
            for i in range(image_rows - 1):
                if bottom_right + (i + 1) * image_cols < len(vertices):
                    start_pos = vertices[bottom_right + i * image_cols]
                    end_pos = vertices[bottom_right + (i + 1) * image_cols]
                    for t in np.linspace(0, 1, 100):
                        point = start_pos + t * (end_pos - start_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
        else:
            # Fallback for simple 4-corner case
            if len(base_vertices) >= 4:
                apex_pos = vertices[0]
                # Create linspace points for apex to corner edges
                for i in range(1, 5):
                    corner_pos = vertices[i]
                    for t in np.linspace(0, 1, 100):
                        point = apex_pos + t * (corner_pos - apex_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
                
                # Create linspace points for base perimeter edges
                corner_pairs = [(1, 2), (2, 3), (3, 4), (4, 1)]
                for start_idx, end_idx in corner_pairs:
                    start_pos = vertices[start_idx]
                    end_pos = vertices[end_idx]
                    for t in np.linspace(0, 1, 100):
                        point = start_pos + t * (end_pos - start_pos)
                        wireframe_vertices.append(point)
                        wireframe_colors.append([255, 255, 255, 255])  # White wireframe
        
        # Combine all vertices: apex + base vertices + wireframe vertices
        if len(wireframe_vertices) > 0:
            all_vertices = np.vstack([vertices, np.array(wireframe_vertices)])
        else:
            all_vertices = vertices
        
        # Adjust face indices to account for apex vertex
        base_faces = [[face[0] + 1, face[1] + 1, face[2] + 1] for face in base_faces]
        
        # Texture coordinates: apex + base texture coords
        texture_coords = np.vstack([[0.0, 0.0], np.array(base_texture_coords)])
        
        # Create color array for all vertices
        all_colors = []
        # Apex color (white)
        all_colors.append([255, 255, 255, 255])
        
        # Base vertex colors (from image)
        for i in range(1, len(vertices)):
            if self.image is not None:
                u, v = texture_coords[i]
                height, width = self.image.shape[:2]
                x = int(u * (width - 1))  # Convert back to image coordinates (no flip)
                y = int(v * (height - 1))  # Convert back to image coordinates (no flip)
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                
                if len(self.image.shape) == 3:
                    r, g, b = self.image[y, x]
                else:
                    r = g = b = self.image[y, x]
                
                all_colors.append([r, g, b, 77])  # 30% transparency
            else:
                all_colors.append([255, 255, 255, 77])
        
        # Add wireframe colors
        all_colors.extend(wireframe_colors)
        
        # No edges needed since we're using point clouds
        edges = []
        
        return all_vertices, edges, base_faces, texture_coords, all_colors
    
    def export_to_cloudcompare_ply(self, output_path: str, focal_length: Optional[float] = None, pixel_sampling: int = 1, image_sampling: int = 1) -> bool:
        """
        Export camera pyramid to PLY format for CloudCompare with wireframe point clouds and image point cloud base.
        
        Args:
            output_path: Path to save the PLY file
            focal_length: Focal length (uses self.focal_length if None)
            pixel_sampling: Sample every Nth pixel for wireframe corners (default: 1)
            image_sampling: Sample every Nth pixel for image base (default: 1 for full resolution)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            vertices, edges, base_faces, texture_coords, colors = self.get_pyramid_with_texture_coords(focal_length, pixel_sampling, image_sampling)
            
            with open(output_path, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("property uchar alpha\n")
                f.write("end_header\n")
                
                # Write vertices with colors
                for i, vertex in enumerate(vertices):
                    if i < len(colors):
                        r, g, b, alpha = colors[i]
                        f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {r} {g} {b} {alpha}\n")
                    else:
                        # Fallback color
                        f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} 255 255 255 255\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting to PLY: {e}")
            return False
    
    def export_to_cloudcompare_obj(self, output_path: str, focal_length: Optional[float] = None) -> bool:
        """
        Export camera pyramid to OBJ format for CloudCompare with wireframe and transparent base.
        
        Args:
            output_path: Path to save the OBJ file
            focal_length: Focal length (uses self.focal_length if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            vertices, edges, base_faces, texture_coords = self.get_pyramid_with_texture_coords(focal_length)
            
            with open(output_path, 'w') as f:
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                # Write texture coordinates
                for tex_coord in texture_coords:
                    f.write(f"vt {tex_coord[0]:.6f} {tex_coord[1]:.6f}\n")
                
                # Write lines (wireframe)
                for edge in edges:
                    f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting to OBJ: {e}")
            return False
    
    def export_to_cloudcompare_json(self, output_path: str) -> bool:
        """
        Export camera parameters to JSON format.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            camera_data = {
                "name": self.name,
                "position": self.position.tolist(),
                "rotation_matrix": self.rotation_matrix.tolist(),
                "intrinsics": self.intrinsics.tolist(),
                "image_size": self.image_size,
                "focal_length_pixels": self.focal_length_pixels,
                "focal_length_3d": self.focal_length,
                "scale_factor": self.scale_factor
            }
            
            with open(output_path, 'w') as f:
                json.dump(camera_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    @classmethod
    def from_camera_params(cls, camera_data: dict, image_path: Optional[str] = None, scale_factor: float = 0.001) -> 'Camera':
        """
        Create a Camera instance from camera parameters dictionary.
        
        Args:
            camera_data: Dictionary containing camera parameters
            image_path: Path to the camera image file
            scale_factor: Scale factor to convert pixel focal length to 3D units
            
        Returns:
            Camera instance
        """
        return cls(
            name=camera_data.get("name", "camera"),
            position=camera_data["position"],
            rotation_matrix=camera_data["rotation_matrix"],
            intrinsics=camera_data["intrinsics"],
            image_size=camera_data["image_size"],
            image_path=image_path,
            focal_length=camera_data.get("focal_length"),
            scale_factor=scale_factor
        )


def load_cameras_from_json(json_path: str, images_dir: Optional[str] = None, scale_factor: float = 0.001) -> List[Camera]:
    """
    Load cameras from a JSON file containing camera parameters.
    
    Args:
        json_path: Path to the JSON file
        images_dir: Directory containing camera images
        scale_factor: Scale factor to convert pixel focal length to 3D units
        
    Returns:
        List of Camera instances
        
    Note:
        The JSON file should contain camera-to-world transforms:
        - "position": camera center in world coordinates [x, y, z]
        - "rotation_matrix": 3x3 matrix such that P_world = R @ P_camera + position
        - "intrinsics": 3x3 camera intrinsic matrix
        - "image_size": [width, height]
        
        If the JSON contains world-to-camera transforms (e.g., from stereo calibration),
        they must be converted to camera-to-world before calling this function.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cameras = []
    for camera_data in data:
        image_path = None
        if images_dir and "name" in camera_data:
            image_path = os.path.join(images_dir, f"{camera_data['name']}.png")
        
        camera = Camera.from_camera_params(camera_data, image_path, scale_factor)
        cameras.append(camera)
    
    return cameras


def convert_world_to_camera_to_camera_to_world(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert world-to-camera transforms to camera-to-world transforms.
    
    Args:
        rotation_matrix: 3x3 rotation matrix R such that X_camera = R @ X_world + T
        translation_vector: 3x1 translation vector T such that X_camera = R @ X_world + T
        
    Returns:
        Tuple of (camera_to_world_rotation, camera_center_in_world)
        where:
        - camera_to_world_rotation: R_cam_to_world such that X_world = R_cam_to_world @ X_camera + camera_center
        - camera_center_in_world: camera center position in world coordinates
    """
    # Convert world-to-camera to camera-to-world
    # If X_camera = R @ X_world + T, then X_world = R.T @ (X_camera - T) = R.T @ X_camera - R.T @ T
    camera_to_world_rotation = rotation_matrix.T
    camera_center_in_world = -camera_to_world_rotation @ translation_vector.reshape(3,)
    
    return camera_to_world_rotation, camera_center_in_world


def create_cameras_from_stereo_calibration(calibration: dict, 
                                         image_1: Union[str, np.ndarray],
                                         image_2: Union[str, np.ndarray],
                                         scale_factor: float = 0.001) -> Tuple[Camera, Camera]:
    """
    Create two Camera instances from stereo calibration data.
    
    Args:
        calibration: Stereo calibration data containing world-to-camera transforms
        image_1: Path to first camera image (str) or numpy array of image data
        image_2: Path to second camera image (str) or numpy array of image data
        scale_factor: Scale factor to convert pixel focal length to 3D units
        
    Returns:
        Tuple of (camera1, camera2)
        
    Note:
        The calibration data contains world-to-camera transforms from cv2.stereoCalibrate:
        - rotation_matrix: R such that X_camera2 = R @ X_camera1 + T
        - translation_vector: T such that X_camera2 = R @ X_camera1 + T
        
        This function converts them to camera-to-world transforms for the Camera class.
    """
    data = calibration
    
    # Extract camera matrices and calibration data
    camera_matrix1 = np.array(data["camera_matrix1"])
    camera_matrix2 = np.array(data["camera_matrix2"])
    rotation_matrix = np.array(data["rotation_matrix"])  # world-to-camera2 rotation
    translation_vector = np.array(data["translation_vector"])  # world-to-camera2 translation
    image_size = tuple(data["image_size"])
    
    # Camera 1 (reference camera at origin in world coordinates)
    camera1 = Camera(
        name="camera_1",
        position=[0.0, 0.0, 0.0],  # camera center at world origin
        rotation_matrix=np.eye(3),  # identity rotation (camera1 = world frame)
        intrinsics=camera_matrix1,
        image_size=image_size,
        image_path=image_1 if isinstance(image_1, str) else None,
        scale_factor=scale_factor
    )
    
    # If image_1 is a numpy array, set it directly
    if isinstance(image_1, np.ndarray):
        camera1.image = image_1
    
    # Camera 2: Convert world-to-camera transforms to camera-to-world transforms
    # The stereo calibration gives us: X_camera2 = R @ X_camera1 + T
    # We need: X_world = R_cam2_to_world @ X_camera2 + camera2_center_in_world
    camera2_rotation_world, camera2_position_world = convert_world_to_camera_to_camera_to_world(
        rotation_matrix, translation_vector
    )
    
    camera2 = Camera(
        name="camera_2",
        position=camera2_position_world,  # camera2 center in world coordinates
        rotation_matrix=camera2_rotation_world,  # camera2-to-world rotation
        intrinsics=camera_matrix2,
        image_size=image_size,
        image_path=image_2 if isinstance(image_2, str) else None,
        scale_factor=scale_factor
    )
    
    # If image_2 is a numpy array, set it directly
    if isinstance(image_2, np.ndarray):
        camera2.image = image_2
    
    return camera1, camera2


def export_cameras_to_cloudcompare(cameras: List[Camera], output_dir: str, format: str = "ply", pixel_sampling: int = 4, image_sampling: int = 4) -> bool:
    """
    Export multiple cameras to CloudCompare format.
    
    Args:
        cameras: List of Camera instances
        output_dir: Directory to save the exported files
        format: Export format ("ply", "obj", or "json")
        pixel_sampling: Sample every Nth pixel for wireframe corners (default: 4)
        image_sampling: Sample every Nth pixel for image base (default: 4)
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for camera in cameras:
        if format == "ply":
            output_path = os.path.join(output_dir, f"{camera.name}_pyramid.ply")
            success = camera.export_to_cloudcompare_ply(output_path, pixel_sampling=pixel_sampling, image_sampling=image_sampling)
        elif format == "obj":
            output_path = os.path.join(output_dir, f"{camera.name}_pyramid.obj")
            success = camera.export_to_cloudcompare_obj(output_path)
        elif format == "json":
            output_path = os.path.join(output_dir, f"{camera.name}_params.json")
            success = camera.export_to_cloudcompare_json(output_path)
        else:
            print(f"Unsupported format: {format}")
            continue
        
        if success:
            success_count += 1
        else:
            print(f"Failed to export {camera.name}")
    
    return success_count == len(cameras)


def main(calibration_path: str, image_1_path: str, image_2_path: str, output_dir: str) -> None:
    calibration = json.load(open(calibration_path))
    image_1 = cv2.imread(image_1_path)
    image_2 = cv2.imread(image_2_path)
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration, image_1, image_2)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--image_1", type=str, required=True)
    parser.add_argument("--image_2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.calibration, args.image_1, args.image_2, args.output_dir)