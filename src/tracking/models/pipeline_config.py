"""Pipeline configuration models for crop module tracking."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .classifier import ClassifierConfig


class DetectorConfig(BaseModel):
    """Configuration for object detector."""

    model_path: Path = Field(..., description="Path to model checkpoint file")
    architecture: str = Field(..., description="Model architecture (rt-detr or yolo)")

    @field_validator("model_path", mode="before")
    @classmethod
    def validate_model_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v


class LayoutTrackerConfig(BaseModel):
    """Configuration for layout-based tracker."""

    alpha: float = Field(
        0.7, ge=0.0, le=1.0, description="Weight for distance component"
    )
    beta: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight for confidence component"
    )
    max_distance: float = Field(
        0.2, ge=0.0, description="Maximum normalized distance for valid match"
    )
    confidence_thresh: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )


class CropModuleConfig(BaseModel):
    """Configuration for PCA-based crop module."""

    n_components: int = Field(
        2, ge=1, le=3, description="Number of principal components"
    )
    use_grayscale: bool = Field(
        True, description="Convert crops to grayscale before PCA"
    )
    mask_fusion_alpha: float = Field(
        0.7, ge=0.0, le=1.0, description="Fusion weight for SAM vs motion"
    )
    sam_fail_min_coverage: float = Field(
        0.01, ge=0.0, le=1.0, description="Minimum SAM mask coverage"
    )
    sam_fail_max_coverage: float = Field(
        0.90, ge=0.0, le=1.0, description="Maximum SAM mask coverage"
    )
    mask_fusion_eps: float = Field(
        1e-6, ge=0.0, description="Small epsilon for safe normalizations"
    )
    use_motion_in_pca_mask: bool = Field(
        True, description="Incorporate motion mask into PCA masking"
    )
    use_motion_for_direction: bool = Field(
        False, description="Use motion distribution to flip PCA axis direction"
    )


class ArrowSenseConfig(BaseModel):
    """Configuration for arrow sense temporal smoothing (EMA)."""

    tau_seconds: float = Field(0.2, gt=0.0, description="EMA time constant in seconds")
    eps: float = Field(
        1e-6, ge=0.0, description="Small epsilon for numerical stability"
    )
    fusion_threshold: float = Field(
        0.0, ge=0.0, description="Threshold for fusion weights (0 = use all pixels)"
    )
    purge_state_after_seconds: float = Field(
        0.5, gt=0.0, description="Drop per-track EMA state after inactivity"
    )


class BackgroundDetectorConfig(BaseModel):
    """Configuration for background detector."""

    type: str = Field("ocv", description="Background detector type: 'ocv' or 'vpi'")
    backend: str = Field("cuda", description="Backend to use (cuda or cpu)")
    learn_rate: float = Field(
        0.01, ge=0.0, le=1.0, description="Learning rate for background model"
    )


class MaskDetectorConfig(BaseModel):
    """Configuration for mask detector."""

    type: str = Field(
        "sam",
        description="Mask detector type: 'sam', 'morphological_snake', or 'grabcut'",
    )
    iterations: int = Field(
        50, ge=1, le=1000, description="Number of iterations for mask refinement"
    )
    init_scale: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Morph snake initialization scale as a fraction of bbox size "
            "(used to build an init mask within the bbox crop)"
        ),
    )
    model_path: Path | None = Field(
        None, description="Path to SAM2 model (only for SAM type)"
    )

    @field_validator("model_path", mode="before")
    @classmethod
    def validate_model_path(cls, v: Any) -> Path | None:
        """Convert string to Path or None."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v


class VisualizationConfig(BaseModel):
    """Configuration for PCA vector visualization."""

    arrow_scale: float = Field(
        50.0, gt=0.0, description="Scale factor for arrow length"
    )
    arrow_thickness: int = Field(3, ge=1, description="Thickness of arrow lines")
    arrow_color: tuple[int, int, int] = Field(
        (0, 255, 255), description="Color of arrows in BGR format"
    )

    @field_validator("arrow_color")
    @classmethod
    def validate_arrow_color(
        cls, v: tuple[int, int, int] | list[int]
    ) -> tuple[int, int, int]:
        """Ensure arrow color is a tuple of 3 integers."""
        if isinstance(v, list):
            return tuple(v)
        return v


class OutputConfig(BaseModel):
    """Configuration for output generation."""

    generate_fgmask_video: bool = Field(
        True, description="Generate foreground mask video"
    )
    generate_pca_visualization: bool = Field(
        True, description="Generate PCA vector visualization video"
    )
    output_tracking_video: bool = Field(
        True,
        description="Produce main tracking video (colored bboxes + class labels). Overlays controlled by render_masks and render_arrows.",
    )
    render_masks: bool = Field(
        False, description="Overlay masks on tracking video (memory intensive)"
    )
    render_arrows: bool = Field(
        False, description="Overlay PCA arrows on tracking video (memory intensive)"
    )


class PipelineConfig(BaseModel):
    """Main pipeline configuration containing all component configs."""

    detector: DetectorConfig
    layout_tracker: LayoutTrackerConfig
    crop_module: CropModuleConfig
    arrow_sense: ArrowSenseConfig
    background_detector: BackgroundDetectorConfig
    mask_detector: MaskDetectorConfig
    visualization: VisualizationConfig
    output: OutputConfig
    classifier: ClassifierConfig | None = None

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "PipelineConfig":
        """
        Load pipeline configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PipelineConfig instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, yaml_path: Path | str) -> None:
        """
        Save pipeline configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings for YAML serialization
        data = self.model_dump(mode="json")
        # Convert Path objects to strings
        if "detector" in data and "model_path" in data["detector"]:
            data["detector"]["model_path"] = str(data["detector"]["model_path"])

        with yaml_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
