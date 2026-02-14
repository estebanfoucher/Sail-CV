from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Image(BaseModel):
    """
    Image representation with color space information

    Input:
        image: numpy array of shape (H, W, 3) representing an image
        rgb_bgr: color space, either "RGB" or "BGR"

    Output:
        Validated Image object with conversion methods
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(..., description="Image as numpy array (H, W, 3)")
    rgb_bgr: Literal["RGB", "BGR"] = Field(..., description="Color space: RGB or BGR")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: np.ndarray) -> np.ndarray:
        if len(v.shape) != 3:
            raise ValueError(f"image must be 3D array (H, W, C), got shape {v.shape}")
        if v.shape[2] != 3:
            raise ValueError(f"image must have 3 channels, got {v.shape[2]} channels")
        return v

    def to_rgb(self) -> np.ndarray:
        """Convert image to RGB color space"""
        if self.rgb_bgr == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.image.copy()

    def to_bgr(self) -> np.ndarray:
        """Convert image to BGR color space"""
        if self.rgb_bgr == "RGB":
            return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        return self.image.copy()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get image shape (H, W, C)"""
        return self.image.shape
