"""
3D Sail tracking models for perspective-aware telltale tracking.

This module defines the geometry of the sail, camera configuration,
and telltale positions for 3D-to-2D projection and tracking.
"""

from pydantic import BaseModel, Field


class TelltalePoint(BaseModel):
    """
    A telltale point on the sail surface.

    Coordinates are normalized (0-1) relative to sail dimensions.
    u = 0 is at the mast (luff), u = 1 is at the leech.
    v = 0 is at the foot, v = 1 is at the head.
    """

    id: str = Field(..., description="Unique identifier for this telltale (e.g., 'TL', 'TR')")
    name: str = Field(..., description="Human-readable name (e.g., 'top left')")
    u: float = Field(..., ge=0.0, le=1.0, description="Normalized position along sail width [0=luff, 1=leech]")
    v: float = Field(..., ge=0.0, le=1.0, description="Normalized position along sail height [0=foot, 1=head]")


class SailGeometry(BaseModel):
    """
    Geometry of the sail as a rectangle in 3D space.

    The sail is modeled as a rectangle attached to the mast:
    - Width extends from the mast (luff) toward the leech
    - Height extends from the foot toward the head
    - The sail rotates around the Z-axis (vertical) at the mast position
    """

    width: float = Field(..., gt=0, description="Sail width in meters (luff to leech)")
    height: float = Field(..., gt=0, description="Sail height in meters (foot to head)")
    mast_position: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Mast base position (x, y, z) in world coordinates"
    )
    telltales: list[TelltalePoint] = Field(
        default_factory=list,
        description="List of telltale points on the sail"
    )

    @classmethod
    def from_json_dict(cls, data: dict) -> "SailGeometry":
        """Create SailGeometry from JSON dictionary."""
        telltales_data = data.get("telltales", [])
        telltales = [TelltalePoint(**t) for t in telltales_data]
        return cls(
            width=data["width"],
            height=data["height"],
            mast_position=tuple(data.get("mast_position", [0.0, 0.0, 0.0])),
            telltales=telltales,
        )


class CameraConfig(BaseModel):
    """
    Camera configuration for 3D-to-2D projection.

    The camera is defined by its position, orientation (look_at target),
    and intrinsic parameters (focal length, principal point).
    """

    position: tuple[float, float, float] = Field(
        ..., description="Camera position (x, y, z) in world coordinates"
    )
    look_at: tuple[float, float, float] = Field(
        ..., description="Point the camera is looking at (x, y, z) in world coordinates"
    )
    focal_length: float = Field(
        ..., gt=0, description="Focal length in pixels"
    )
    principal_point: tuple[float, float] = Field(
        ..., description="Principal point (cx, cy) in pixels"
    )
    image_size: tuple[int, int] = Field(
        ..., description="Image size (width, height) in pixels"
    )
    up_vector: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0),
        description="Up vector for camera orientation (default: Z-up)"
    )

    @classmethod
    def from_json_dict(cls, data: dict) -> "CameraConfig":
        """Create CameraConfig from JSON dictionary."""
        return cls(
            position=tuple(data["position"]),
            look_at=tuple(data["look_at"]),
            focal_length=data["focal_length"],
            principal_point=tuple(data["principal_point"]),
            image_size=tuple(data["image_size"]),
            up_vector=tuple(data.get("up_vector", [0.0, 0.0, 1.0])),
        )


class Sail3DConfig(BaseModel):
    """
    Complete configuration for 3D sail tracking.

    Combines sail geometry, camera configuration, and optimization parameters.
    Supports both 1-DOF (angle only) and 2-DOF (angle + twist) tracking.
    """

    sail: SailGeometry = Field(..., description="Sail geometry and telltale positions")
    camera: CameraConfig = Field(..., description="Camera configuration")
    
    # Angle (rotation) parameters
    angle_min: float = Field(default=0.0, description="Minimum sail angle in degrees")
    angle_max: float = Field(default=45.0, description="Maximum sail angle in degrees")
    coarse_steps: int = Field(
        default=10, ge=1, description="Number of discrete angle steps for coarse search"
    )
    
    # Twist parameters (2-DOF tracking)
    twist_min: float = Field(default=0.0, description="Minimum twist angle in degrees")
    twist_max: float = Field(default=30.0, description="Maximum twist angle in degrees")
    coarse_steps_twist: int = Field(
        default=5, ge=1, description="Number of discrete twist steps for coarse search"
    )

    @classmethod
    def from_json_dict(cls, data: dict) -> "Sail3DConfig":
        """Create Sail3DConfig from JSON dictionary."""
        sail = SailGeometry.from_json_dict(data["sail"])
        camera = CameraConfig.from_json_dict(data["camera"])
        return cls(
            sail=sail,
            camera=camera,
            angle_min=data.get("angle_min", 0.0),
            angle_max=data.get("angle_max", 45.0),
            coarse_steps=data.get("coarse_steps", 10),
            twist_min=data.get("twist_min", 0.0),
            twist_max=data.get("twist_max", 30.0),
            coarse_steps_twist=data.get("coarse_steps_twist", 5),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "Sail3DConfig":
        """Load Sail3DConfig from a JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.from_json_dict(data)
