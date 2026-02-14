import numpy as np
from pydantic import BaseModel, Field, model_validator


class XYXY(BaseModel):
    """
    Bounding box in XYXY format (x1, y1, x2, y2)
    where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
    """

    x1: int = Field(..., description="Left coordinate")
    y1: int = Field(..., description="Top coordinate")
    x2: int = Field(..., description="Right coordinate")
    y2: int = Field(..., description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self):
        if self.x2 <= self.x1:
            raise ValueError(
                f"x2 must be greater than x1, got x1={self.x1}, x2={self.x2}"
            )
        if self.y2 <= self.y1:
            raise ValueError(
                f"y2 must be greater than y1, got y1={self.y1}, y2={self.y2}"
            )
        return self

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]"""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @classmethod
    def from_numpy(cls, xyxy: np.ndarray) -> "XYXY":
        """Create from numpy array [x1, y1, x2, y2]"""
        if len(xyxy) != 4:
            raise ValueError(f"xyxy must have 4 elements, got {len(xyxy)}")
        return cls(x1=int(xyxy[0]), y1=int(xyxy[1]), x2=int(xyxy[2]), y2=int(xyxy[3]))


class BoundingBox(BaseModel):
    """
    Bounding box with coordinates and class information

    Input:
        xyxy: XYXY object with coordinates

    Output:
        Validated BoundingBox object
    """

    xyxy: XYXY = Field(..., description="Bounding box coordinates in XYXY format")

    def to_numpy(self) -> np.ndarray:
        """Convert xyxy coordinates to numpy array"""
        return self.xyxy.to_numpy()

    @classmethod
    def from_numpy(cls, xyxy: np.ndarray) -> "BoundingBox":
        """Create from numpy array"""
        return cls(xyxy=XYXY.from_numpy(xyxy))
