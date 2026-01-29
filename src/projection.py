"""
3D-to-2D projection module for sail tracking.

This module handles the transformation of 3D telltale positions on the sail
to 2D pixel coordinates in the camera image, accounting for sail rotation.

Coordinate system:
- X-axis: Boat axis (bow is +X)
- Y-axis: Lateral (starboard is +Y)
- Z-axis: Vertical (up is +Z)
"""

import numpy as np

from models.sail_3d import CameraConfig, SailGeometry, TelltalePoint


def sail_to_world(
    telltale: TelltalePoint,
    sail: SailGeometry,
    angle_deg: float,
) -> np.ndarray:
    """
    Convert sail-relative coordinates to world 3D coordinates.

    The sail rotates around the Z-axis at the mast position.
    At angle=0, the sail is aligned with the X-axis (boat axis).
    At angle=45, the sail is opened 45 degrees to starboard.

    Args:
        telltale: Telltale point with normalized (u, v) coordinates
        sail: Sail geometry with dimensions and mast position
        angle_deg: Sail angle in degrees (0 = aligned with boat, positive = opened to starboard)

    Returns:
        3D point in world coordinates as numpy array [x, y, z]
    """
    # Convert normalized coordinates to sail-local coordinates
    # u: along width (0 = mast/luff, 1 = leech)
    # v: along height (0 = foot, 1 = head)
    local_x = telltale.u * sail.width  # Distance from mast along sail
    local_y = 0.0  # Sail is a flat surface, no Y offset in sail frame
    local_z = telltale.v * sail.height  # Height on sail

    # Rotation matrix around Z-axis
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotate the sail point around the mast (Z-axis)
    # At angle=0: sail extends along +X
    # At angle>0: sail rotates toward +Y (starboard)
    rotated_x = local_x * cos_a - local_y * sin_a
    rotated_y = local_x * sin_a + local_y * cos_a
    rotated_z = local_z

    # Add mast position offset
    mast_x, mast_y, mast_z = sail.mast_position
    world_point = np.array([
        mast_x + rotated_x,
        mast_y + rotated_y,
        mast_z + rotated_z,
    ])

    return world_point


def compute_camera_matrix(camera: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera rotation matrix and translation vector.

    Uses look-at formulation to construct the camera coordinate frame.

    Args:
        camera: Camera configuration with position and look_at target

    Returns:
        Tuple of (rotation_matrix, translation_vector)
        - rotation_matrix: 3x3 matrix transforming world to camera frame
        - translation_vector: 3x1 vector (camera position in world)
    """
    # Camera position and target
    cam_pos = np.array(camera.position)
    look_at = np.array(camera.look_at)
    up = np.array(camera.up_vector)

    # Compute camera coordinate axes
    # Z-axis: pointing from camera toward target (forward)
    z_axis = look_at - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    # X-axis: right vector (perpendicular to Z and up)
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y-axis: down vector (perpendicular to Z and X)
    # Note: In image coordinates, Y points down
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Rotation matrix: rows are camera axes in world coordinates
    # This transforms world coordinates to camera coordinates
    rotation_matrix = np.array([x_axis, y_axis, z_axis])

    return rotation_matrix, cam_pos


def world_to_camera(
    point_3d: np.ndarray,
    camera: CameraConfig,
) -> np.ndarray:
    """
    Transform world coordinates to camera frame coordinates.

    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        camera: Camera configuration

    Returns:
        3D point in camera coordinates [x_cam, y_cam, z_cam]
        where z_cam is depth (distance along viewing direction)
    """
    rotation_matrix, cam_pos = compute_camera_matrix(camera)

    # Translate point relative to camera position, then rotate
    point_relative = point_3d - cam_pos
    point_camera = rotation_matrix @ point_relative

    return point_camera


def camera_to_pixel(
    point_cam: np.ndarray,
    camera: CameraConfig,
) -> tuple[float, float]:
    """
    Project camera-frame 3D point to 2D pixel coordinates.

    Uses pinhole camera model:
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

    Args:
        point_cam: 3D point in camera coordinates [x, y, z]
        camera: Camera configuration with intrinsics

    Returns:
        Tuple (u, v) pixel coordinates
    """
    x, y, z = point_cam

    # Avoid division by zero for points behind camera
    if z <= 0:
        return (float('nan'), float('nan'))

    # Pinhole projection (assuming fx = fy = focal_length)
    fx = fy = camera.focal_length
    cx, cy = camera.principal_point

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    return (u, v)


def project_point(
    point_3d: np.ndarray,
    camera: CameraConfig,
) -> tuple[float, float]:
    """
    Project a 3D world point to 2D pixel coordinates.

    Combines world_to_camera and camera_to_pixel.

    Args:
        point_3d: 3D point in world coordinates
        camera: Camera configuration

    Returns:
        Tuple (u, v) pixel coordinates
    """
    point_cam = world_to_camera(point_3d, camera)
    return camera_to_pixel(point_cam, camera)


def project_telltales(
    sail: SailGeometry,
    camera: CameraConfig,
    angle_deg: float,
) -> list[tuple[float, float]]:
    """
    Project all telltales on the sail to 2D pixel coordinates.

    Args:
        sail: Sail geometry with telltale positions
        camera: Camera configuration
        angle_deg: Sail angle in degrees

    Returns:
        List of (u, v) pixel coordinates for each telltale,
        in the same order as sail.telltales
    """
    projected = []
    for telltale in sail.telltales:
        world_point = sail_to_world(telltale, sail, angle_deg)
        pixel = project_point(world_point, camera)
        projected.append(pixel)
    return projected


def get_sail_corners_world(
    sail: SailGeometry,
    angle_deg: float,
) -> np.ndarray:
    """
    Get the 4 corners of the sail rectangle in world coordinates.

    Useful for visualization.

    Args:
        sail: Sail geometry
        angle_deg: Sail angle in degrees

    Returns:
        4x3 numpy array with corner positions:
        [bottom-luff, bottom-leech, top-leech, top-luff]
    """
    # Create temporary telltale points at corners
    corners_uv = [
        (0.0, 0.0),  # bottom-luff (foot at mast)
        (1.0, 0.0),  # bottom-leech (foot at leech)
        (1.0, 1.0),  # top-leech (head at leech)
        (0.0, 1.0),  # top-luff (head at mast)
    ]

    corners_world = []
    for u, v in corners_uv:
        temp_point = TelltalePoint(id="corner", name="corner", u=u, v=v)
        world_pos = sail_to_world(temp_point, sail, angle_deg)
        corners_world.append(world_pos)

    return np.array(corners_world)


def get_telltales_world(
    sail: SailGeometry,
    angle_deg: float,
) -> list[tuple[str, np.ndarray]]:
    """
    Get all telltale positions in world coordinates.

    Args:
        sail: Sail geometry with telltale positions
        angle_deg: Sail angle in degrees

    Returns:
        List of (telltale_id, world_position) tuples
    """
    result = []
    for telltale in sail.telltales:
        world_pos = sail_to_world(telltale, sail, angle_deg)
        result.append((telltale.id, world_pos))
    return result
