"""
Layout position models for layout-based tracking.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator


class LayoutPosition(BaseModel):
    """
    A single layout position representing an expected object location.

    Coordinates are normalized (0-1) relative to image dimensions.
    Origin (0, 0) is top-left corner.
    """

    id: str = Field(
        ..., description="Unique identifier for this layout position (e.g., 'TL', 'TR')"
    )
    name: str = Field(..., description="Human-readable name (e.g., 'top left')")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate (0-1)")

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return int(self.x * width), int(self.y * height)


class Layout(BaseModel):
    """
    A collection of layout positions with optional direction prior.

    The direction field is a 2D unitary vector indicating the most likely
    direction for objects in this layout (e.g., (1.0, 0.0) for rightward).
    """

    positions: list[LayoutPosition] = Field(
        default_factory=list, description="List of layout positions"
    )
    direction: tuple[float, float] | None = Field(
        default=None,
        description="Optional 2D unitary vector indicating most likely direction (dx, dy). Normalized to unit length.",
    )

    @field_validator("direction")
    @classmethod
    def normalize_direction(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Normalize direction vector to unit length."""
        if v is None:
            return None
        dx, dy = v
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude == 0:
            return None
        return (dx / magnitude, dy / magnitude)

    @classmethod
    def from_json_dict(cls, data: dict) -> "Layout":
        """
        Create Layout from JSON dictionary.

        Supports format: {"layout": [...], "direction": [1.0, 0.0]}
        """
        layout_list = data.get("layout", data)
        if isinstance(layout_list, list):
            positions = [LayoutPosition(**item) for item in layout_list]
            direction = data.get("direction")
            if direction is not None:
                direction = tuple(direction)
            return cls(positions=positions, direction=direction)
        raise ValueError(
            "Invalid layout format: expected 'layout' key with list of positions"
        )
