from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .bounding_box import BoundingBox


class Detection(BaseModel):
    """
    Detection result containing bounding box, confidence, and class information

    bbox: BoundingBox with coordinates and class_id
    confidence: Confidence score between 0 and 1
    class_id: Class ID (must be >= 0)
    """

    bbox: BoundingBox = Field(
        ..., description="Bounding box with coordinates and class_id"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    class_id: int = Field(..., ge=0, description="Class ID (must be >= 0)")


class ModelSpecs(BaseModel):
    """
    Model configuration and specifications

    Input:
        model_path: Path to model file
        architecture: Architecture type ("yolo" or "rt-detr")

    Output:
        Validated ModelSpecs object
    """

    model_path: Path = Field(..., description="Path to model file")
    architecture: Literal["yolo", "rt-detr"] = Field(
        ..., description="Model architecture"
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        path = v
        if not path.exists():
            raise ValueError(f"Model file does not exist: {path}")
        return path
