from pathlib import Path

import numpy as np
import pytest
from detector import Detector
from pydantic import ValidationError
from tracker import Tracker

from model_weights import resolve_model_path
from models import (
    XYXY,
    BoundingBox,
    Detection,
    Image,
    ModelSpecs,
    Track,
    TrackerConfig,
)


def test_track_model():
    """Test Track Pydantic model creation and validation"""
    xyxy = XYXY(x1=10, y1=20, x2=30, y2=40)
    bbox = BoundingBox(xyxy=xyxy)
    detection = Detection(bbox=bbox, confidence=0.9, class_id=0)

    track = Track(detection=detection, track_id=1, frame_id=5)

    assert track.detection == detection
    assert track.track_id == 1
    assert track.frame_id == 5
    assert track.detection.bbox.xyxy.x1 == 10


def test_tracker_config_model():
    """Test TrackerConfig Pydantic model creation and validation"""
    config = TrackerConfig(track_thresh=0.5, track_buffer=30, match_thresh=0.8)

    assert config.track_thresh == 0.5
    assert config.track_buffer == 30
    assert config.match_thresh == 0.8


def test_tracker_config_validation():
    """Test TrackerConfig validation constraints"""
    # Test invalid track_thresh (should be 0-1)
    with pytest.raises(ValidationError):
        TrackerConfig(
            track_thresh=1.5,  # Invalid: > 1
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30.0,
        )

    # Test invalid track_buffer (should be >= 1)
    with pytest.raises(ValidationError):
        TrackerConfig(
            track_thresh=0.5,
            track_buffer=0,  # Invalid: < 1
            match_thresh=0.8,
            frame_rate=30.0,
        )


def test_tracker_initialization():
    """Test Tracker service class initialization"""
    config = TrackerConfig(
        track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30.0
    )

    tracker = Tracker(config)

    assert tracker.config == config
    assert tracker._byte_tracker is None  # Lazy initialization


def test_tracker_update():
    """Test Tracker.update() with Detection objects"""
    config = TrackerConfig(
        track_thresh=0.3,  # Low threshold for testing
        track_buffer=30,
        match_thresh=0.5,
        frame_rate=30.0,
    )

    tracker = Tracker(config)

    # Create test detections
    detections = []
    for i in range(3):
        xyxy = XYXY(x1=10 + i * 50, y1=20, x2=30 + i * 50, y2=40)
        bbox = BoundingBox(xyxy=xyxy)
        detection = Detection(bbox=bbox, confidence=0.8, class_id=0)
        detections.append(detection)

    # First update - should create new tracks
    tracks1 = tracker.update(detections)
    assert len(tracks1) > 0
    assert all(isinstance(track, Track) for track in tracks1)
    assert all(track.detection in detections for track in tracks1)

    # Second update with same detections - should match existing tracks
    tracks2 = tracker.update(detections)
    assert len(tracks2) > 0
    # Track IDs should be consistent (same detections should match)
    assert all(track.track_id is not None for track in tracks2)


def test_tracker_integration_with_detector():
    """Test integration between Detector and Tracker (uses HF-backed rt-detr)."""
    project_root = Path(__file__).resolve().parents[2]
    model_path = resolve_model_path(
        project_root / "checkpoints" / "sailcv-rtdetrl640.pt",
        project_root=project_root,
    )

    # Initialize Detector
    specs = ModelSpecs(model_path=model_path, architecture="rt-detr")
    detector = Detector(specs)

    # Initialize Tracker
    tracker_config = TrackerConfig(
        track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30.0
    )
    tracker = Tracker(tracker_config)

    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    image = Image(image=test_image, rgb_bgr="BGR")

    # Run detection
    detections = detector.detect(image)

    # Run tracking
    tracks = tracker.update(detections)

    # Verify output
    assert isinstance(tracks, list)
    assert all(isinstance(track, Track) for track in tracks)

    # If there are detections, there should be tracks (if above threshold)
    if len(detections) > 0:
        # Tracks should have valid track_ids
        for track in tracks:
            assert track.track_id is not None
            assert track.track_id >= 0
            assert isinstance(track.detection, Detection)


def test_track_serialization():
    """Test Track model serialization (for JSON export)"""
    xyxy = XYXY(x1=10, y1=20, x2=30, y2=40)
    bbox = BoundingBox(xyxy=xyxy)
    detection = Detection(bbox=bbox, confidence=0.9, class_id=0)

    track = Track(detection=detection, track_id=1, frame_id=5)

    # Test model_dump
    track_dict = track.model_dump()
    assert isinstance(track_dict, dict)
    assert track_dict["track_id"] == 1
    assert track_dict["frame_id"] == 5
    assert "detection" in track_dict
