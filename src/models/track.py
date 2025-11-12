from pydantic import BaseModel, Field

from .detector import Detection


class Track(BaseModel):
    """
    Track result containing a Detection and tracking information

    detection: Detection object that this track represents
    track_id: Unique track identifier
    frame_id: Optional frame ID for tracking state
    """

    detection: Detection = Field(..., description="The detection this track represents")
    track_id: int = Field(
        ..., ge=0, description="Unique track identifier (must be >= 0)"
    )
    frame_id: int | None = Field(
        None, ge=0, description="Optional frame ID for tracking state"
    )


class TrackerConfig(BaseModel):
    """
    Tracker configuration and specifications

    Input:
        track_thresh: Confidence threshold for track initialization
        track_buffer: Number of frames to keep lost tracks
        match_thresh: IoU threshold for matching tracks to detections
    """

    track_thresh: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for track initialization (0-1)",
    )
    track_buffer: int = Field(
        ..., ge=1, description="Number of frames to keep lost tracks (must be >= 1)"
    )
    match_thresh: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="IoU threshold for matching tracks to detections (0-1)",
    )
