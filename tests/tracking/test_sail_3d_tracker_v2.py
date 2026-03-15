"""
Tests for 2-DOF 3D sail tracking with twist.

Tests cover:
- Projection math with twist
- Sail3DTrackerV2 with synthetic detections
- 2-DOF (angle + twist) recovery accuracy
- Latency benchmarks
"""

import math
import time

import numpy as np
import pytest
from projection import (
    get_sail_mesh_world,
    project_telltales,
    sail_to_world,
)
from sail_3d_tracker_v2 import Sail3DTrackerV2

from models import CameraConfig, Detection, Sail3DConfig, SailGeometry, TelltalePoint
from models.bounding_box import XYXY, BoundingBox

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
            TelltalePoint(id="ML", name="middle left", u=0.1, v=0.5),
            TelltalePoint(id="MR", name="middle right", u=0.9, v=0.5),
            TelltalePoint(id="BL", name="bottom left", u=0.1, v=0.1),
            TelltalePoint(id="BR", name="bottom right", u=0.9, v=0.1),
        ],
    )


@pytest.fixture
def simple_camera() -> CameraConfig:
    """Create a simple camera configuration for testing."""
    return CameraConfig(
        position=(0.0, -4.0, 2.5),
        look_at=(1.5, 0.0, 2.5),
        focal_length=500.0,
        principal_point=(640.0, 360.0),
        image_size=(1280, 720),
        up_vector=(0.0, 0.0, 1.0),
    )


@pytest.fixture
def sail_3d_config(
    simple_sail: SailGeometry, simple_camera: CameraConfig
) -> Sail3DConfig:
    """Create a complete 3D config for testing."""
    return Sail3DConfig(
        sail=simple_sail,
        camera=simple_camera,
        angle_min=0.0,
        angle_max=45.0,
        coarse_steps=7,
        twist_min=0.0,
        twist_max=30.0,
        coarse_steps_twist=5,
    )


def make_detection(
    cx: float, cy: float, size: float = 20.0, conf: float = 0.9
) -> Detection:
    """Create a detection centered at (cx, cy) with given size and confidence."""
    half = size / 2
    return Detection(
        bbox=BoundingBox(
            xyxy=XYXY(
                x1=int(cx - half),
                y1=int(cy - half),
                x2=int(cx + half),
                y2=int(cy + half),
            )
        ),
        confidence=conf,
        class_id=0,
    )


# ============================================================================
# Projection Tests with Twist
# ============================================================================


class TestSailToWorldWithTwist:
    """Tests for sail_to_world transformation with twist."""

    def test_no_twist_same_as_before(self, simple_sail: SailGeometry):
        """With twist=0, results should match original behavior."""
        telltale = TelltalePoint(id="test", name="test", u=0.5, v=0.5)

        # Without twist parameter (defaults to 0)
        pos_no_twist = sail_to_world(telltale, simple_sail, 20.0, 0.0)

        # The position should be consistent
        assert pos_no_twist is not None
        assert len(pos_no_twist) == 3

    def test_twist_affects_top_more_than_bottom(self, simple_sail: SailGeometry):
        """With twist, top telltales should rotate more than bottom ones."""
        top_telltale = TelltalePoint(id="top", name="top", u=0.5, v=1.0)
        bottom_telltale = TelltalePoint(id="bottom", name="bottom", u=0.5, v=0.0)

        base_angle = 10.0
        twist = 20.0

        # Top telltale at v=1.0 should have effective angle = base + twist = 30°
        top_pos = sail_to_world(top_telltale, simple_sail, base_angle, twist)

        # Bottom telltale at v=0.0 should have effective angle = base = 10°
        bottom_pos = sail_to_world(bottom_telltale, simple_sail, base_angle, twist)

        # Top should have more Y displacement (more rotated toward starboard)
        assert top_pos[1] > bottom_pos[1]

    def test_twist_linear_interpolation(self, simple_sail: SailGeometry):
        """Twist should interpolate linearly between foot and head."""
        base_angle = 0.0
        twist = 30.0

        # Middle telltale at v=0.5 should have effective angle = 15°
        middle_telltale = TelltalePoint(id="mid", name="mid", u=1.0, v=0.5)
        middle_pos = sail_to_world(middle_telltale, simple_sail, base_angle, twist)

        # At v=0.5, effective angle = 0 + 0.5 * 30 = 15°
        # So Y displacement should be sin(15°) * width
        expected_y = np.sin(np.deg2rad(15)) * simple_sail.width
        np.testing.assert_almost_equal(middle_pos[1], expected_y, decimal=5)


class TestGetSailMeshWorld:
    """Tests for sail mesh generation."""

    def test_mesh_strip_count(self, simple_sail: SailGeometry):
        """Mesh should have the requested number of strips."""
        num_strips = 5
        mesh = get_sail_mesh_world(simple_sail, 10.0, 15.0, num_strips=num_strips)
        assert len(mesh) == num_strips

    def test_mesh_quad_shape(self, simple_sail: SailGeometry):
        """Each mesh element should be a 4x3 array (4 corners, 3D)."""
        mesh = get_sail_mesh_world(simple_sail, 10.0, 15.0, num_strips=3)
        for quad in mesh:
            assert quad.shape == (4, 3)

    def test_mesh_covers_sail_height(self, simple_sail: SailGeometry):
        """Mesh should cover from foot to head."""
        mesh = get_sail_mesh_world(simple_sail, 0.0, 0.0, num_strips=4)

        # First quad bottom should be at z=0
        assert abs(mesh[0][0, 2]) < 0.01  # bottom-luff z

        # Last quad top should be at z=height
        assert abs(mesh[-1][2, 2] - simple_sail.height) < 0.01  # top-leech z


class TestProjectTelltalesWithTwist:
    """Tests for projection with twist."""

    def test_projection_with_twist_valid(
        self, simple_sail: SailGeometry, simple_camera: CameraConfig
    ):
        """All telltales should project to valid coordinates with twist."""
        projected = project_telltales(simple_sail, simple_camera, 15.0, 10.0)

        assert len(projected) == len(simple_sail.telltales)
        for u, v in projected:
            assert not math.isnan(u)
            assert not math.isnan(v)

    def test_twist_changes_projection(
        self, simple_sail: SailGeometry, simple_camera: CameraConfig
    ):
        """Adding twist should change projected positions, especially at top."""
        proj_no_twist = project_telltales(simple_sail, simple_camera, 15.0, 0.0)
        proj_with_twist = project_telltales(simple_sail, simple_camera, 15.0, 20.0)

        # Top telltales (high v) should move more than bottom telltales
        top_indices = [i for i, t in enumerate(simple_sail.telltales) if t.v > 0.7]
        bottom_indices = [i for i, t in enumerate(simple_sail.telltales) if t.v < 0.3]

        top_movements = [
            abs(proj_with_twist[i][0] - proj_no_twist[i][0])
            + abs(proj_with_twist[i][1] - proj_no_twist[i][1])
            for i in top_indices
        ]
        bottom_movements = [
            abs(proj_with_twist[i][0] - proj_no_twist[i][0])
            + abs(proj_with_twist[i][1] - proj_no_twist[i][1])
            for i in bottom_indices
        ]

        # Top should move more on average
        assert np.mean(top_movements) > np.mean(bottom_movements)


# ============================================================================
# Tracker V2 Tests
# ============================================================================


class TestSail3DTrackerV2:
    """Tests for Sail3DTrackerV2."""

    def test_initialization(self, sail_3d_config: Sail3DConfig):
        """Tracker should initialize without errors."""
        tracker = Sail3DTrackerV2(sail_3d_config)
        assert tracker.config == sail_3d_config
        assert tracker._last_angle is None
        assert tracker._last_twist is None

    def test_empty_detections(self, sail_3d_config: Sail3DConfig):
        """Empty detections should return empty tracks."""
        tracker = Sail3DTrackerV2(sail_3d_config)
        tracks, angle, twist = tracker.update([])

        assert len(tracks) == 0
        assert angle == sail_3d_config.angle_min
        assert twist == sail_3d_config.twist_min

    def test_2dof_recovery_no_twist(self, sail_3d_config: Sail3DConfig):
        """Tracker should recover angle accurately when twist=0."""
        tracker = Sail3DTrackerV2(sail_3d_config)

        ground_truth_angle = 20.0
        ground_truth_twist = 0.0

        # Project telltales at ground truth
        projected = project_telltales(
            sail_3d_config.sail,
            sail_3d_config.camera,
            ground_truth_angle,
            ground_truth_twist,
        )

        # Create detections
        detections = [make_detection(x, y) for x, y in projected]

        tracks, estimated_angle, estimated_twist = tracker.update(detections)

        # Check recovery accuracy
        assert len(tracks) == len(sail_3d_config.sail.telltales)
        assert abs(estimated_angle - ground_truth_angle) < 5.0  # Within 5 degrees
        assert abs(estimated_twist - ground_truth_twist) < 5.0  # Within 5 degrees

    def test_2dof_recovery_with_twist(self, sail_3d_config: Sail3DConfig):
        """Tracker should recover both angle and twist accurately."""
        tracker = Sail3DTrackerV2(sail_3d_config)

        ground_truth_angle = 15.0
        ground_truth_twist = 12.0

        # Project telltales at ground truth
        projected = project_telltales(
            sail_3d_config.sail,
            sail_3d_config.camera,
            ground_truth_angle,
            ground_truth_twist,
        )

        # Create detections
        detections = [make_detection(x, y) for x, y in projected]

        tracks, estimated_angle, estimated_twist = tracker.update(detections)

        # Check recovery accuracy
        # Note: 2-DOF is harder than 1-DOF, allow 7 degree tolerance
        assert len(tracks) == len(sail_3d_config.sail.telltales)
        assert abs(estimated_angle - ground_truth_angle) < 7.0
        assert abs(estimated_twist - ground_truth_twist) < 7.0

    def test_2dof_recovery_with_noise(self, sail_3d_config: Sail3DConfig):
        """Tracker should handle noisy detections reasonably."""
        tracker = Sail3DTrackerV2(sail_3d_config)

        ground_truth_angle = 25.0
        ground_truth_twist = 15.0
        noise_std = 10.0  # 10 pixels noise

        # Project telltales at ground truth
        projected = project_telltales(
            sail_3d_config.sail,
            sail_3d_config.camera,
            ground_truth_angle,
            ground_truth_twist,
        )

        # Create noisy detections
        np.random.seed(42)
        detections = [
            make_detection(
                x + np.random.normal(0, noise_std), y + np.random.normal(0, noise_std)
            )
            for x, y in projected
        ]

        _tracks, estimated_angle, estimated_twist = tracker.update(detections)

        # Should still recover reasonably (within 8 degrees for noisy data)
        assert abs(estimated_angle - ground_truth_angle) < 8.0
        assert abs(estimated_twist - ground_truth_twist) < 8.0

    def test_get_projected_positions(self, sail_3d_config: Sail3DConfig):
        """get_projected_positions should return correct format."""
        tracker = Sail3DTrackerV2(sail_3d_config)

        positions = tracker.get_projected_positions(15.0, 10.0)

        assert len(positions) == len(sail_3d_config.sail.telltales)
        for telltale_id, x, y in positions:
            assert isinstance(telltale_id, str)
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_last_estimated_params(self, sail_3d_config: Sail3DConfig):
        """Last estimated params should be updated after update()."""
        tracker = Sail3DTrackerV2(sail_3d_config)

        assert tracker.last_estimated_angle is None
        assert tracker.last_estimated_twist is None

        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 20.0, 10.0
        )
        detections = [make_detection(x, y) for x, y in projected]

        tracker.update(detections)

        assert tracker.last_estimated_angle is not None
        assert tracker.last_estimated_twist is not None


# ============================================================================
# Benchmark Tests
# ============================================================================


class TestSail3DTrackerV2Benchmark:
    """Benchmark tests for latency."""

    def test_latency_target(self, sail_3d_config: Sail3DConfig):
        """Tracker latency should be under 30ms target for real-time (30+ FPS)."""
        tracker = Sail3DTrackerV2(sail_3d_config)

        # Create detections at some parameters
        projected = project_telltales(
            sail_3d_config.sail, sail_3d_config.camera, 20.0, 15.0
        )
        detections = [make_detection(x, y) for x, y in projected]

        # Warm up
        tracker.update(detections)

        # Benchmark
        NUM_RUNS = 50
        latencies = []

        for _ in range(NUM_RUNS):
            tracker.frame_id = 0
            tracker._last_angle = None
            tracker._last_twist = None

            start = time.perf_counter()
            tracker.update(detections)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        avg_latency = np.mean(latencies)

        # Should be under 30ms for 30+ FPS real-time tracking
        assert avg_latency < 30.0, (
            f"Average latency {avg_latency:.1f}ms exceeds 30ms target"
        )

        # Print for visibility in test output
        print(f"\n2-DOF Tracker average latency: {avg_latency:.2f}ms")
