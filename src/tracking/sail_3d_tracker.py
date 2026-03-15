"""
3D Sail Tracker with perspective-aware telltale tracking.

This tracker jointly estimates:
1. Point assignments: which detection corresponds to which telltale (discrete)
2. Sail angle: the rotation of the sail from angle_min to angle_max (continuous)

Uses a hybrid optimization approach:
1. Coarse search: evaluate discrete angle steps
2. Continuous refinement: scipy.optimize for fine-tuning
3. Hungarian algorithm for optimal 1-to-1 assignment
"""

import logging
import math

import numpy as np
from projection import project_telltales
from scipy.optimize import linear_sum_assignment, minimize_scalar

from models import Detection, Track
from models.sail_3d import Sail3DConfig

logger = logging.getLogger(__name__)


class Sail3DTracker:
    """
    Tracker that assigns detections to telltale positions using 3D projection.

    The tracker models the sail as a rectangle in 3D space that can rotate
    around the mast. Given 2D detections, it estimates both:
    - The sail angle (continuous)
    - The assignment of detections to telltales (discrete)

    Cost function:
        cost = alpha * normalized_distance + beta * (1 - confidence)

    Uses Hungarian algorithm for optimal 1-to-1 assignment at each angle,
    then optimizes the angle to minimize total assignment cost.
    """

    def __init__(
        self,
        config: Sail3DConfig,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_distance: float = 0.2,
        confidence_thresh: float = 0.0,
    ):
        """
        Initialize Sail3DTracker.

        Args:
            config: Sail3DConfig with sail geometry, camera, and optimization params
            alpha: Weight for distance component (default 0.7)
            beta: Weight for confidence component (default 0.3)
            max_distance: Maximum normalized distance for valid match (default 0.2 = 20% of diagonal)
            confidence_thresh: Minimum confidence threshold for detections (default 0.0)
        """
        self.config = config
        self.alpha = alpha
        self.beta = beta
        self.max_distance = max_distance
        self.confidence_thresh = confidence_thresh
        self.frame_id = 0

        # Store last estimated angle for temporal smoothing (optional future enhancement)
        self._last_angle: float | None = None

        # Precompute image diagonal for normalization
        width, height = config.camera.image_size
        self.diagonal = math.sqrt(width**2 + height**2)

        logger.info(
            f"[Sail3DTracker] Initialized with {len(config.sail.telltales)} telltales, "
            f"angle range [{config.angle_min}, {config.angle_max}] degrees, "
            f"alpha={alpha}, beta={beta}, max_distance={max_distance}"
        )

    def update(self, detections: list[Detection]) -> tuple[list[Track], float]:
        """
        Assign detections to telltale positions and estimate sail angle.

        Args:
            detections: List of Detection objects

        Returns:
            Tuple of (tracks, estimated_angle_deg):
            - tracks: List of Track objects with telltale IDs
            - estimated_angle_deg: Estimated sail angle in degrees
        """
        self.frame_id += 1

        # Filter by confidence threshold
        valid_detections = [
            d for d in detections if d.confidence >= self.confidence_thresh
        ]

        logger.debug(
            f"[Sail3DTracker] Frame {self.frame_id}: {len(valid_detections)}/{len(detections)} "
            f"detections above confidence threshold {self.confidence_thresh}"
        )

        if not valid_detections or not self.config.sail.telltales:
            return (
                [],
                self._last_angle
                if self._last_angle is not None
                else self.config.angle_min,
            )

        # 1. Coarse search: evaluate discrete angles
        best_angle, best_cost = self._coarse_search(valid_detections)

        logger.debug(
            f"[Sail3DTracker] Frame {self.frame_id}: Coarse search found angle={best_angle:.1f}° "
            f"with cost={best_cost:.4f}"
        )

        # 2. Continuous refinement around best angle
        refined_angle = self._refine_angle(valid_detections, best_angle)

        logger.debug(
            f"[Sail3DTracker] Frame {self.frame_id}: Refined angle={refined_angle:.2f}°"
        )

        # 3. Final assignment at refined angle
        tracks = self._assign_at_angle(valid_detections, refined_angle)

        # Update last angle for potential temporal smoothing
        self._last_angle = refined_angle

        logger.debug(
            f"[Sail3DTracker] Frame {self.frame_id}: Assigned {len(tracks)}/{len(valid_detections)} "
            f"detections at angle={refined_angle:.2f}°"
        )

        return tracks, refined_angle

    def _coarse_search(self, detections: list[Detection]) -> tuple[float, float]:
        """
        Evaluate cost at discrete angle steps and return the best angle.

        Args:
            detections: List of Detection objects

        Returns:
            Tuple of (best_angle, best_cost)
        """
        angle_min = self.config.angle_min
        angle_max = self.config.angle_max
        steps = self.config.coarse_steps

        # Generate angle samples
        if steps == 1:
            angles = [(angle_min + angle_max) / 2]
        else:
            angles = np.linspace(angle_min, angle_max, steps + 1)

        best_angle = angles[0]
        best_cost = float("inf")

        for angle in angles:
            cost, _ = self._compute_cost_at_angle(detections, angle)
            if cost < best_cost:
                best_cost = cost
                best_angle = angle

        return float(best_angle), best_cost

    def _compute_cost_at_angle(
        self,
        detections: list[Detection],
        angle_deg: float,
    ) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
        """
        Compute optimal assignment cost at a given sail angle.

        Args:
            detections: List of Detection objects
            angle_deg: Sail angle in degrees

        Returns:
            Tuple of (total_cost, (row_indices, col_indices))
        """
        # Project telltales at this angle
        projected = project_telltales(self.config.sail, self.config.camera, angle_deg)

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
                    cost_matrix[i, j] = float("inf")
                    continue

                # Euclidean distance normalized by diagonal
                dx = det_cx - proj_x
                dy = det_cy - proj_y
                distance = math.sqrt(dx**2 + dy**2) / self.diagonal
                distance_cost = self.alpha * distance

                cost_matrix[i, j] = distance_cost + confidence_cost

        return cost_matrix

    def _refine_angle(
        self,
        detections: list[Detection],
        initial_angle: float,
    ) -> float:
        """
        Continuous optimization around initial angle estimate.

        Args:
            detections: List of Detection objects
            initial_angle: Starting angle from coarse search

        Returns:
            Refined angle in degrees
        """
        # Define search bounds around initial estimate
        search_margin = 5.0  # degrees
        lower_bound = max(self.config.angle_min, initial_angle - search_margin)
        upper_bound = min(self.config.angle_max, initial_angle + search_margin)

        # If bounds are the same, return initial angle
        if lower_bound >= upper_bound:
            return initial_angle

        # Objective function
        def objective(angle: float) -> float:
            cost, _ = self._compute_cost_at_angle(detections, angle)
            return cost

        # Optimize
        result = minimize_scalar(
            objective,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": 0.1},  # 0.1 degree tolerance
        )

        return float(result.x)

    def _assign_at_angle(
        self,
        detections: list[Detection],
        angle_deg: float,
    ) -> list[Track]:
        """
        Perform final assignment at the given angle and create Track objects.

        Args:
            detections: List of Detection objects
            angle_deg: Sail angle in degrees

        Returns:
            List of Track objects
        """
        # Project telltales
        projected = project_telltales(self.config.sail, self.config.camera, angle_deg)

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
                    f"[Sail3DTracker] Frame {self.frame_id}: Rejecting match "
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
                f"[Sail3DTracker] Frame {self.frame_id}: Matched detection[{det_idx}] "
                f"(conf={detection.confidence:.3f}) -> {telltale.id} (distance={distance:.3f})"
            )

        return tracks

    def get_projected_positions(
        self, angle_deg: float
    ) -> list[tuple[str, float, float]]:
        """
        Get projected 2D positions of all telltales at a given angle.

        Useful for visualization and debugging.

        Args:
            angle_deg: Sail angle in degrees

        Returns:
            List of (telltale_id, x, y) tuples
        """
        projected = project_telltales(self.config.sail, self.config.camera, angle_deg)
        return [
            (t.id, p[0], p[1])
            for t, p in zip(self.config.sail.telltales, projected, strict=False)
        ]

    @property
    def last_estimated_angle(self) -> float | None:
        """Return the last estimated sail angle, or None if no frames processed."""
        return self._last_angle
