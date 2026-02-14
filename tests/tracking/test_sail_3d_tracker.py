"""
Tests for 3D sail tracking components.

Tests cover:
- Projection math (3D to 2D)
- Sail3DTracker with synthetic detections
- Angle recovery accuracy
"""

import math

import numpy as np
import pytest

from models import Detection, CameraConfig, Sail3DConfig, SailGeometry, TelltalePoint
from models.bounding_box import XYXY, BoundingBox
from projection import (
    sail_to_world,
    world_to_camera,
    camera_to_pixel,
    project_point,
    project_telltales,
    get_sail_corners_world,
)
from sail_3d_tracker import Sail3DTracker


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_sail() -> SailGeometry:
    """Create a simple sail geometry for testing."""
    return SailGeometry(
        width=3.0,
        height=5.0,
        mast_position=(0.0, 0.0, 0.0),
        telltales=[
            TelltalePoint(id="TL", name="top left", u=0.1, v=0.9),
            TelltalePoint(id="TR", name="top right", u=0.9, v=0.9),
            TelltalePoint(id="BL", name="bottom left", u=0.1, v=0.1),
            TelltalePoint(id="BR", name="bottom right", u=0.9, v=0.1),
        ],
    )


@pytest.fixture
def simple_camera() -> CameraConfig:
    """Create a simple camera configuration for testing.
    
    Camera is positioned to the side of the boat, looking at the sail.
    The configuration ensures telltales project within the image frame.
    """
    return CameraConfig(
        position=(0.0, -4.0, 2.5),  # 4m to port side, 2.5m height
        look_at=(1.5, 0.0, 2.5),    # Looking at middle of sail
        focal_length=500.0,          # Wider FOV for testing
        principal_point=(640.0, 360.0),
        image_size=(1280, 720),
        up_vector=(0.0, 0.0, 1.0),
    )


@pytest.fixture
def sail_3d_config(simple_sail: SailGeometry, simple_camera: CameraConfig) -> Sail3DConfig:
    """Create a complete 3D config for testing."""
    return Sail3DConfig(
        sail=simple_sail,
        camera=simple_camera,
        angle_min=0.0,
        angle_max=45.0,
        coarse_steps=10,
    )


def make_detection(cx: float, cy: float, size: float = 20.0, conf: float = 0.9) -> Detection:
    """Create a detection centered at (cx, cy) with given size and confidence."""
    half = size / 2
    return Detection(
        bbox=BoundingBox(
            xyxy=XYXY(
                x1=int(cx - half),
                y1=int(cy - half),
                x2=int(cx + half),
                y2=int(cy + half)
            )
        ),
        confidence=conf,
        class_id=0,
    )


# ============================================================================
# Projection Tests
# ============================================================================

class TestSailToWorld:
    """Tests for sail_to_world transformation."""

    def test_origin_telltale_at_zero_angle(self, simple_sail: SailGeometry):
        """Telltale at (0, 0) should be at mast position regardless of angle."""
        telltale = TelltalePoint(id="origin", name="origin", u=0.0, v=0.0)
        
        for angle in [0.0, 15.0, 30.0, 45.0]:
            world_pos = sail_to_world(telltale, simple_sail, angle)
            np.testing.assert_array_almost_equal(
                world_pos, [0.0, 0.0, 0.0], decimal=6
            )

    def test_telltale_position_at_zero_angle(self, simple_sail: SailGeometry):
        """At angle=0, sail extends along +X axis."""
        # Telltale at u=1 (full width), v=0 (bottom)
        telltale = TelltalePoint(id="leech_bottom", name="leech bottom", u=1.0, v=0.0)
        world_pos = sail_to_world(telltale, simple_sail, 0.0)
        
        # Should be at x=width, y=0, z=0
        np.testing.assert_array_almost_equal(
            world_pos, [3.0, 0.0, 0.0], decimal=6
        )

    def test_telltale_position_at_90_angle(self, simple_sail: SailGeometry):
        """At angle=90, sail extends along +Y axis."""
        telltale = TelltalePoint(id="leech_bottom", name="leech bottom", u=1.0, v=0.0)
        world_pos = sail_to_world(telltale, simple_sail, 90.0)
        
        # Should be at x~0, y=width, z=0
        np.testing.assert_array_almost_equal(
            world_pos, [0.0, 3.0, 0.0], decimal=6
        )

    def test_telltale_height(self, simple_sail: SailGeometry):
        """Telltale v coordinate maps to Z height."""
        telltale = TelltalePoint(id="top", name="top", u=0.0, v=1.0)
        world_pos = sail_to_world(telltale, simple_sail, 0.0)
        
        # Should be at mast, full height
        np.testing.assert_array_almost_equal(
            world_pos, [0.0, 0.0, 5.0], decimal=6
        )

    def test_mast_offset(self):
        """Mast position offset should be applied."""
        sail = SailGeometry(
            width=2.0,
            height=3.0,
            mast_position=(1.0, 2.0, 0.5),
            telltales=[],
        )
        telltale = TelltalePoint(id="test", name="test", u=0.0, v=0.0)
        world_pos = sail_to_world(telltale, sail, 0.0)
        
        np.testing.assert_array_almost_equal(
            world_pos, [1.0, 2.0, 0.5], decimal=6
        )


class TestWorldToCamera:
    """Tests for world_to_camera transformation."""

    def test_point_in_front_of_camera(self, simple_camera: CameraConfig):
        """Point in front of camera should have positive Z in camera frame."""
        # Point at look_at target
        point = np.array(simple_camera.look_at)
        cam_point = world_to_camera(point, simple_camera)
        
        # Z should be positive (in front of camera)
        assert cam_point[2] > 0

    def test_camera_position_maps_to_origin(self, simple_camera: CameraConfig):
        """Camera position in world should map to origin in camera frame."""
        point = np.array(simple_camera.position)
        cam_point = world_to_camera(point, simple_camera)
        
        np.testing.assert_array_almost_equal(
            cam_point, [0.0, 0.0, 0.0], decimal=6
        )


class TestCameraToPixel:
    """Tests for camera_to_pixel projection."""

    def test_point_on_optical_axis(self, simple_camera: CameraConfig):
        """Point on optical axis should project to principal point."""
        # Point directly in front of camera (on optical axis)
        cam_point = np.array([0.0, 0.0, 1.0])
        u, v = camera_to_pixel(cam_point, simple_camera)
        
        cx, cy = simple_camera.principal_point
        assert abs(u - cx) < 1e-6
        assert abs(v - cy) < 1e-6

    def test_point_behind_camera(self, simple_camera: CameraConfig):
        """Point behind camera should return NaN."""
        cam_point = np.array([0.0, 0.0, -1.0])
        u, v = camera_to_pixel(cam_point, simple_camera)
        
        assert math.isnan(u)
        assert math.isnan(v)


class TestProjectTelltales:
    """Tests for full projection pipeline."""

    def test_all_telltales_project(self, simple_sail: SailGeometry, simple_camera: CameraConfig):
        """All telltales should project to valid pixel coordinates."""
        projected = project_telltales(simple_sail, simple_camera, 0.0)
        
        assert len(projected) == len(simple_sail.telltales)
        
        for u, v in projected:
            assert not math.isnan(u)
            assert not math.isnan(v)

    def test_projection_changes_with_angle(
        self, simple_sail: SailGeometry, simple_camera: CameraConfig
    ):
        """Projected positions should change as sail angle changes."""
        proj_0 = project_telltales(simple_sail, simple_camera, 0.0)
        proj_45 = project_telltales(simple_sail, simple_camera, 45.0)
        
        # At least some points should have different projections
        differences = [
            abs(p0[0] - p45[0]) + abs(p0[1] - p45[1])
            for p0, p45 in zip(proj_0, proj_45, strict=False)
        ]
        assert max(differences) > 1.0  # At least 1 pixel difference


class TestGetSailCornersWorld:
    """Tests for sail corner computation."""

    def test_corner_count(self, simple_sail: SailGeometry):
        """Should return 4 corners."""
        corners = get_sail_corners_world(simple_sail, 0.0)
        assert corners.shape == (4, 3)

    def test_corners_at_zero_angle(self, simple_sail: SailGeometry):
        """Corners at angle=0 should be along X axis."""
        corners = get_sail_corners_world(simple_sail, 0.0)
        
        # Bottom-luff at origin
        np.testing.assert_array_almost_equal(corners[0], [0, 0, 0], decimal=6)
        # Bottom-leech at (width, 0, 0)
        np.testing.assert_array_almost_equal(corners[1], [3, 0, 0], decimal=6)
        # Top-leech at (width, 0, height)
        np.testing.assert_array_almost_equal(corners[2], [3, 0, 5], decimal=6)
        # Top-luff at (0, 0, height)
        np.testing.assert_array_almost_equal(corners[3], [0, 0, 5], decimal=6)


# ============================================================================
# Tracker Tests
# ============================================================================

class TestSail3DTracker:
    """Tests for Sail3DTracker."""

    def test_initialization(self, sail_3d_config: Sail3DConfig):
        """Tracker should initialize without errors."""
        tracker = Sail3DTracker(sail_3d_config)
        assert tracker.config == sail_3d_config
        assert tracker._last_angle is None

    def test_empty_detections(self, sail_3d_config: Sail3DConfig):
        """Empty detections should return empty tracks."""
        tracker = Sail3DTracker(sail_3d_config)
        tracks, angle = tracker.update([])
        
        assert len(tracks) == 0
        assert angle == sail_3d_config.angle_min

    def test_single_detection_assignment(self, sail_3d_config: Sail3DConfig):
        """Single detection should be assigned to nearest telltale."""
        tracker = Sail3DTracker(sail_3d_config)
        
        # Project telltales at angle=0
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 0.0
        )
        
        # Create detection near first projected position
        det_x, det_y = projected[0]
        detection = make_detection(det_x, det_y)
        
        tracks, angle = tracker.update([detection])
        
        assert len(tracks) == 1
        assert tracks[0].track_id == sail_3d_config.sail.telltales[0].id

    def test_multiple_detections_assignment(self, sail_3d_config: Sail3DConfig):
        """Multiple detections should be assigned optimally."""
        tracker = Sail3DTracker(sail_3d_config)
        
        # Project telltales at angle=0
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 0.0
        )
        
        # Create detections near projected positions (first 3)
        detections = []
        for i in range(3):
            det_x, det_y = projected[i]
            detections.append(make_detection(det_x + 5, det_y + 5))  # Small offset
        
        tracks, angle = tracker.update(detections)
        
        assert len(tracks) == 3
        track_ids = {t.track_id for t in tracks}
        expected_ids = {sail_3d_config.sail.telltales[i].id for i in range(3)}
        assert track_ids == expected_ids

    def test_angle_estimation_at_zero(self, sail_3d_config: Sail3DConfig):
        """Tracker should estimate angle close to 0 when detections match angle=0."""
        tracker = Sail3DTracker(sail_3d_config)
        
        # Project telltales at angle=0
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 0.0
        )
        
        # Create detections exactly at projected positions
        detections = [make_detection(x, y) for x, y in projected]
        
        tracks, estimated_angle = tracker.update(detections)
        
        # Estimated angle should be close to 0
        assert abs(estimated_angle - 0.0) < 5.0  # Within 5 degrees

    def test_angle_estimation_at_30(self, sail_3d_config: Sail3DConfig):
        """Tracker should estimate angle close to 30 when detections match angle=30."""
        tracker = Sail3DTracker(sail_3d_config)
        
        # Project telltales at angle=30
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 30.0
        )
        
        # Create detections exactly at projected positions
        detections = [make_detection(x, y) for x, y in projected]
        
        tracks, estimated_angle = tracker.update(detections)
        
        # Estimated angle should be close to 30
        assert abs(estimated_angle - 30.0) < 5.0  # Within 5 degrees

    def test_confidence_filtering(self, sail_3d_config: Sail3DConfig):
        """Low confidence detections should be filtered."""
        tracker = Sail3DTracker(sail_3d_config, confidence_thresh=0.5)
        
        # Project telltales at angle=0
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 0.0
        )
        
        # Create one high-conf and one low-conf detection
        det_high = make_detection(projected[0][0], projected[0][1], conf=0.9)
        det_low = make_detection(projected[1][0], projected[1][1], conf=0.3)
        
        tracks, angle = tracker.update([det_high, det_low])
        
        # Only high-conf detection should be tracked
        assert len(tracks) == 1

    def test_max_distance_filtering(self, sail_3d_config: Sail3DConfig):
        """Detections far from all telltales should be rejected."""
        tracker = Sail3DTracker(sail_3d_config, max_distance=0.05)  # Very strict
        
        # Create detection far from any telltale
        detection = make_detection(100, 100)  # Far corner of image
        
        tracks, angle = tracker.update([detection])
        
        assert len(tracks) == 0

    def test_get_projected_positions(self, sail_3d_config: Sail3DConfig):
        """get_projected_positions should return correct format."""
        tracker = Sail3DTracker(sail_3d_config)
        
        positions = tracker.get_projected_positions(15.0)
        
        assert len(positions) == len(sail_3d_config.sail.telltales)
        for telltale_id, x, y in positions:
            assert isinstance(telltale_id, str)
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_last_estimated_angle(self, sail_3d_config: Sail3DConfig):
        """last_estimated_angle should be updated after update()."""
        tracker = Sail3DTracker(sail_3d_config)
        
        assert tracker.last_estimated_angle is None
        
        # Project telltales at angle=20
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 20.0
        )
        detections = [make_detection(x, y) for x, y in projected]
        
        tracker.update(detections)
        
        assert tracker.last_estimated_angle is not None
        assert 0.0 <= tracker.last_estimated_angle <= 45.0
