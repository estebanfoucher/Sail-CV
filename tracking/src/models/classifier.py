"""Classifier configuration models."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClassifierConfig(BaseModel):
    """Configuration for crop classifier."""

    model_path: Path = Field(..., description="Path to classifier checkpoint file")
    padding_factor: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Padding factor to extend bbox before cropping (0-1)",
    )
    confidence_threshold: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence threshold for classification (0-1)"
    )

    @field_validator("model_path", mode="before")
    @classmethod
    def validate_model_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v
