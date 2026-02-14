"""
2-DOF 3D Sail Tracker with rotation and twist.

This tracker jointly estimates:
1. Base angle (rotation): sail rotation at the foot
2. Twist: additional rotation at the head relative to the foot

The sail becomes a ruled surface where:
    angle(v) = base_angle + v * twist

Uses a hybrid optimization approach:
1. Coarse 2D grid search over (angle, twist)
2. Continuous 2D refinement with scipy.optimize.minimize
3. Hungarian algorithm for optimal 1-to-1 assignment
"""

import logging
import math

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from models import Detection, Track
from models.sail_3d import Sail3DConfig
from projection import project_telltales

logger = logging.getLogger(__name__)


class Sail3DTrackerV2:
    """
    2-DOF Tracker that assigns detections to telltale positions using 3D projection.

    The tracker models the sail as a twisted surface in 3D space.
    Given 2D detections, it estimates:
    - The base sail angle at the foot (continuous)
    - The twist angle added at the head (continuous)
    - The assignment of detections to telltales (discrete)

    Cost function:
        cost = alpha * normalized_distance + beta * (1 - confidence)

    Uses Hungarian algorithm for optimal 1-to-1 assignment at each (angle, twist),
    then optimizes both parameters to minimize total assignment cost.
    """

    def __init__(
        self,
        config: Sail3DConfig,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_distance: float = 0.3,
        confidence_thresh: float = 0.0,
    ):
        """
        Initialize Sail3DTrackerV2.

        Args:
            config: Sail3DConfig with sail geometry, camera, and optimization params
            alpha: Weight for distance component (default 0.7)
            beta: Weight for confidence component (default 0.3)
            max_distance: Maximum normalized distance for valid match (default 0.3)
            confidence_thresh: Minimum confidence threshold for detections (default 0.0)
        """
        self.config = config
        self.alpha = alpha
        self.beta = beta
        self.max_distance = max_distance
        self.confidence_thresh = confidence_thresh
        self.frame_id = 0

        # Store last estimated parameters for temporal smoothing (optional future enhancement)
        self._last_angle: float | None = None
        self._last_twist: float | None = None

        # Precompute image diagonal for normalization
        width, height = config.camera.image_size
        self.diagonal = math.sqrt(width**2 + height**2)

        logger.info(
            f"[Sail3DTrackerV2] Initialized with {len(config.sail.telltales)} telltales, "
            f"angle range [{config.angle_min}, {config.angle_max}]°, "
            f"twist range [{config.twist_min}, {config.twist_max}]°, "
            f"alpha={alpha}, beta={beta}, max_distance={max_distance}"
        )

    def update(self, detections: list[Detection]) -> tuple[list[Track], float, float]:
        """
        Assign detections to telltale positions and estimate sail angle and twist.

        Args:
            detections: List of Detection objects

        Returns:
            Tuple of (tracks, base_angle_deg, twist_deg):
            - tracks: List of Track objects with telltale IDs
            - base_angle_deg: Estimated base sail angle in degrees
            - twist_deg: Estimated twist angle in degrees
        """
        self.frame_id += 1

        # Filter by confidence threshold
        valid_detections = [
            d for d in detections if d.confidence >= self.confidence_thresh
        ]

        logger.debug(
            f"[Sail3DTrackerV2] Frame {self.frame_id}: {len(valid_detections)}/{len(detections)} "
            f"detections above confidence threshold {self.confidence_thresh}"
        )

        if not valid_detections or not self.config.sail.telltales:
            default_angle = self._last_angle if self._last_angle is not None else self.config.angle_min
            default_twist = self._last_twist if self._last_twist is not None else self.config.twist_min
            return [], default_angle, default_twist

        # 1. Coarse 2D grid search over (angle, twist)
        best_angle, best_twist, best_cost = self._coarse_search_2d(valid_detections)

        logger.debug(
            f"[Sail3DTrackerV2] Frame {self.frame_id}: Coarse search found "
            f"angle={best_angle:.1f}°, twist={best_twist:.1f}° with cost={best_cost:.4f}"
        )

        # 2. Continuous 2D refinement around best parameters
        refined_angle, refined_twist = self._refine_2d(valid_detections, best_angle, best_twist)

        logger.debug(
            f"[Sail3DTrackerV2] Frame {self.frame_id}: Refined to "
            f"angle={refined_angle:.2f}°, twist={refined_twist:.2f}°"
        )

        # 3. Final assignment at refined parameters
        tracks = self._assign_at_params(valid_detections, refined_angle, refined_twist)

        # Update last parameters for potential temporal smoothing
        self._last_angle = refined_angle
        self._last_twist = refined_twist

        logger.debug(
            f"[Sail3DTrackerV2] Frame {self.frame_id}: Assigned {len(tracks)}/{len(valid_detections)} "
            f"detections at angle={refined_angle:.2f}°, twist={refined_twist:.2f}°"
        )

        return tracks, refined_angle, refined_twist

    def _coarse_search_2d(self, detections: list[Detection]) -> tuple[float, float, float]:
        """
        Evaluate cost on a 2D grid of (angle, twist) and return the best combination.

        Args:
            detections: List of Detection objects

        Returns:
            Tuple of (best_angle, best_twist, best_cost)
        """
        angle_min = self.config.angle_min
        angle_max = self.config.angle_max
        angle_steps = self.config.coarse_steps

        twist_min = self.config.twist_min
        twist_max = self.config.twist_max
        twist_steps = self.config.coarse_steps_twist

        # Generate angle and twist samples
        if angle_steps == 1:
            angles = [(angle_min + angle_max) / 2]
        else:
            angles = np.linspace(angle_min, angle_max, angle_steps + 1)

        if twist_steps == 1:
            twists = [(twist_min + twist_max) / 2]
        else:
            twists = np.linspace(twist_min, twist_max, twist_steps + 1)

        best_angle = angles[0]
        best_twist = twists[0]
        best_cost = float('inf')

        for angle in angles:
            for twist in twists:
                cost, _ = self._compute_cost_at_params(detections, angle, twist)
                if cost < best_cost:
                    best_cost = cost
                    best_angle = angle
                    best_twist = twist

        return float(best_angle), float(best_twist), best_cost

    def _compute_cost_at_params(
        self,
        detections: list[Detection],
        base_angle_deg: float,
        twist_deg: float,
    ) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
        """
        Compute optimal assignment cost at given sail parameters.

        Args:
            detections: List of Detection objects
            base_angle_deg: Base sail angle in degrees
            twist_deg: Twist angle in degrees

        Returns:
            Tuple of (total_cost, (row_indices, col_indices))
        """
        # Project telltales at these parameters
        projected = project_telltales(
            self.config.sail, self.config.camera, base_angle_deg, twist_deg
        )

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(detections, projected)

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute total cost of assignment
        total_cost = cost_matrix[row_ind, col_ind].sum()

        return total_cost, (row_ind, col_ind)

    def _build_cost_matrix(
        self,
        detections: list[Detection],
        projected: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build cost matrix between detections and projected telltale positions.

        Shape: (num_detections, num_telltales)

        Cost = alpha * normalized_distance + beta * (1 - confidence)
        """
        num_dets = len(detections)
        num_telltales = len(projected)

        cost_matrix = np.zeros((num_dets, num_telltales))

        for i, detection in enumerate(detections):
            # Confidence cost (same for all telltales)
            confidence_cost = self.beta * (1 - detection.confidence)

            # Detection center
            xyxy = detection.bbox.xyxy
            det_cx = (xyxy.x1 + xyxy.x2) / 2
            det_cy = (xyxy.y1 + xyxy.y2) / 2

            for j, (proj_x, proj_y) in enumerate(projected):
                # Skip invalid projections (behind camera)
                if math.isnan(proj_x) or math.isnan(proj_y):
                    cost_matrix[i, j] = float('inf')
                    continue

                # Euclidean distance normalized by diagonal
                dx = det_cx - proj_x
                dy = det_cy - proj_y
                distance = math.sqrt(dx**2 + dy**2) / self.diagonal
                distance_cost = self.alpha * distance

                cost_matrix[i, j] = distance_cost + confidence_cost

        return cost_matrix

    def _refine_2d(
        self,
        detections: list[Detection],
        initial_angle: float,
        initial_twist: float,
    ) -> tuple[float, float]:
        """
        Continuous 2D optimization around initial parameter estimates.

        Args:
            detections: List of Detection objects
            initial_angle: Starting angle from coarse search
            initial_twist: Starting twist from coarse search

        Returns:
            Tuple of (refined_angle, refined_twist)
        """
        # Define search bounds around initial estimates
        angle_margin = 5.0  # degrees
        twist_margin = 5.0  # degrees

        angle_lower = max(self.config.angle_min, initial_angle - angle_margin)
        angle_upper = min(self.config.angle_max, initial_angle + angle_margin)
        twist_lower = max(self.config.twist_min, initial_twist - twist_margin)
        twist_upper = min(self.config.twist_max, initial_twist + twist_margin)

        # Objective function for 2D optimization
        def objective(params: np.ndarray) -> float:
            angle, twist = params
            cost, _ = self._compute_cost_at_params(detections, angle, twist)
            return cost

        # Initial guess
        x0 = np.array([initial_angle, initial_twist])

        # Bounds
        bounds = [(angle_lower, angle_upper), (twist_lower, twist_upper)]

        # Optimize using L-BFGS-B (fast, handles bounds)
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20, 'ftol': 1e-4},
        )

        return float(result.x[0]), float(result.x[1])

    def _assign_at_params(
        self,
        detections: list[Detection],
        base_angle_deg: float,
        twist_deg: float,
    ) -> list[Track]:
        """
        Perform final assignment at the given parameters and create Track objects.

        Args:
            detections: List of Detection objects
            base_angle_deg: Base sail angle in degrees
            twist_deg: Twist angle in degrees

        Returns:
            List of Track objects
        """
        # Project telltales
        projected = project_telltales(
            self.config.sail, self.config.camera, base_angle_deg, twist_deg
        )

        # Build cost matrix and solve assignment
        cost_matrix = self._build_cost_matrix(detections, projected)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build tracks, filtering by max_distance
        tracks = []
        telltales = self.config.sail.telltales

        for det_idx, tell_idx in zip(row_ind, col_ind, strict=False):
            detection = detections[det_idx]
            telltale = telltales[tell_idx]
            proj_x, proj_y = projected[tell_idx]

            # Compute distance for this match
            xyxy = detection.bbox.xyxy
            det_cx = (xyxy.x1 + xyxy.x2) / 2
            det_cy = (xyxy.y1 + xyxy.y2) / 2

            dx = det_cx - proj_x
            dy = det_cy - proj_y
            distance = math.sqrt(dx**2 + dy**2) / self.diagonal

            # Reject if distance exceeds threshold
            if distance > self.max_distance:
                logger.debug(
                    f"[Sail3DTrackerV2] Frame {self.frame_id}: Rejecting match "
                    f"detection[{det_idx}] -> {telltale.id} (distance={distance:.3f} > {self.max_distance})"
                )
                continue

            track = Track(
                detection=detection,
                track_id=telltale.id,
                frame_id=self.frame_id,
            )
            tracks.append(track)

            logger.debug(
                f"[Sail3DTrackerV2] Frame {self.frame_id}: Matched detection[{det_idx}] "
                f"(conf={detection.confidence:.3f}) -> {telltale.id} (distance={distance:.3f})"
            )

        return tracks

    def get_projected_positions(
        self,
        base_angle_deg: float,
        twist_deg: float,
    ) -> list[tuple[str, float, float]]:
        """
        Get projected 2D positions of all telltales at given parameters.

        Useful for visualization and debugging.

        Args:
            base_angle_deg: Base sail angle in degrees
            twist_deg: Twist angle in degrees

        Returns:
            List of (telltale_id, x, y) tuples
        """
        projected = project_telltales(
            self.config.sail, self.config.camera, base_angle_deg, twist_deg
        )
        return [
            (t.id, p[0], p[1])
            for t, p in zip(self.config.sail.telltales, projected, strict=False)
        ]

    @property
    def last_estimated_angle(self) -> float | None:
        """Return the last estimated base sail angle, or None if no frames processed."""
        return self._last_angle

    @property
    def last_estimated_twist(self) -> float | None:
        """Return the last estimated twist angle, or None if no frames processed."""
        return self._last_twist
