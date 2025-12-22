"""
Layout position models for layout-based tracking.
"""

from pydantic import BaseModel, Field


class LayoutPosition(BaseModel):
    """
    A single layout position representing an expected object location.

    Coordinates are normalized (0-1) relative to image dimensions.
    Origin (0, 0) is top-left corner.
    """

    id: str = Field(..., description="Unique identifier for this layout position (e.g., 'TL', 'TR')")
    name: str = Field(..., description="Human-readable name (e.g., 'top left')")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate (0-1)")

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return int(self.x * width), int(self.y * height)


class Layout(BaseModel):
    """
    A collection of layout positions.
    """

    positions: list[LayoutPosition] = Field(
        default_factory=list, description="List of layout positions"
    )

    @classmethod
    def from_json_dict(cls, data: dict) -> "Layout":
        """
        Create Layout from JSON dictionary.

        Supports format: {"layout": [{"id": "TL", "name": "top left", "x": 0.1, "y": 0.3}, ...]}
        """
        layout_list = data.get("layout", data)
        if isinstance(layout_list, list):
            positions = [LayoutPosition(**item) for item in layout_list]
            return cls(positions=positions)
        raise ValueError("Invalid layout format: expected 'layout' key with list of positions")

