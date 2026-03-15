"""
Layout-based tracker that assigns detections to predefined layout positions.

Uses weighted distance + confidence scoring with Hungarian (1-to-1) assignment.
Track IDs are the layout position IDs (e.g., "TL", "TR").
"""

import logging
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from models import Detection, Track
from models.layout import Layout, LayoutPosition

logger = logging.getLogger(__name__)


class LayoutTracker:
    """
    Tracker that assigns detections to predefined layout positions.

    Instead of matching detections across frames (temporal tracking),
    this tracker matches each detection to the nearest layout position
    using a weighted cost function.

    Cost function:
        cost = alpha * normalized_distance + beta * (1 - confidence)

    Uses Hungarian algorithm for optimal 1-to-1 assignment.
    """

    def __init__(
        self,
        layout: Layout,
        width: int,
        height: int,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_distance: float = 0.2,
        confidence_thresh: float = 0.0,
    ):
        """
        Initialize LayoutTracker.

        Args:
            layout: Layout object with predefined positions
            width: Image width in pixels
            height: Image height in pixels
            alpha: Weight for distance component (default 0.7)
            beta: Weight for confidence component (default 0.3)
            max_distance: Maximum normalized distance for valid match (default 0.2 = 20% of diagonal)
            confidence_thresh: Minimum confidence threshold for detections (default 0.0)
        """
        self.layout = layout
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
        self.max_distance = max_distance
        self.confidence_thresh = confidence_thresh
        self.frame_id = 0

        # Precompute diagonal for normalization
        self.diagonal = math.sqrt(width**2 + height**2)

        # Precompute layout positions in pixels
        self._layout_pixels = [pos.to_pixel(width, height) for pos in layout.positions]

        logger.info(
            f"[LayoutTracker] Initialized with {len(layout.positions)} layout positions, "
            f"alpha={alpha}, beta={beta}, max_distance={max_distance}"
        )

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Assign detections to layout positions.

        Args:
            detections: List of Detection objects

        Returns:
            List of Track objects with layout IDs as track_id
        """
        self.frame_id += 1

        # Filter by confidence threshold
        valid_detections = [
            d for d in detections if d.confidence >= self.confidence_thresh
        ]

        logger.debug(
            f"[LayoutTracker] Frame {self.frame_id}: {len(valid_detections)}/{len(detections)} "
            f"detections above confidence threshold {self.confidence_thresh}"
        )

        if not valid_detections or not self.layout.positions:
            return []

        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(valid_detections)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build tracks from matches
        tracks = []
        for det_idx, layout_idx in zip(row_ind, col_ind, strict=False):
            detection = valid_detections[det_idx]
            layout_pos = self.layout.positions[layout_idx]

            # Check if match is valid (within max_distance)
            distance = self._normalized_distance(detection, layout_idx)
            if distance > self.max_distance:
                logger.debug(
                    f"[LayoutTracker] Frame {self.frame_id}: Rejecting match "
                    f"detection[{det_idx}] -> {layout_pos.id} (distance={distance:.3f} > {self.max_distance})"
                )
                continue

            track = Track(
                detection=detection,
                track_id=layout_pos.id,
                frame_id=self.frame_id,
            )
            tracks.append(track)

            logger.debug(
                f"[LayoutTracker] Frame {self.frame_id}: Matched detection[{det_idx}] "
                f"(conf={detection.confidence:.3f}) -> {layout_pos.id} (distance={distance:.3f})"
            )

        logger.debug(
            f"[LayoutTracker] Frame {self.frame_id}: Assigned {len(tracks)}/{len(valid_detections)} "
            f"detections to layout positions"
        )

        return tracks

    def _compute_cost_matrix(self, detections: list[Detection]) -> np.ndarray:
        """
        Compute cost matrix between detections and layout positions.

        Shape: (num_detections, num_layout_positions)

        Cost = alpha * normalized_distance + beta * (1 - confidence)
        """
        num_dets = len(detections)
        num_layouts = len(self.layout.positions)

        cost_matrix = np.zeros((num_dets, num_layouts))

        for i, detection in enumerate(detections):
            confidence_cost = self.beta * (1 - detection.confidence)

            for j in range(num_layouts):
                distance = self._normalized_distance(detection, j)
                distance_cost = self.alpha * distance

                cost_matrix[i, j] = distance_cost + confidence_cost

        return cost_matrix

    def _normalized_distance(self, detection: Detection, layout_idx: int) -> float:
        """
        Compute normalized Euclidean distance from detection center to layout position.

        Returns distance normalized by image diagonal (0-1 range typically).
        """
        # Detection center
        xyxy = detection.bbox.xyxy
        det_cx = (xyxy.x1 + xyxy.x2) / 2
        det_cy = (xyxy.y1 + xyxy.y2) / 2

        # Layout position in pixels
        layout_x, layout_y = self._layout_pixels[layout_idx]

        # Euclidean distance normalized by diagonal
        dx = det_cx - layout_x
        dy = det_cy - layout_y
        distance = math.sqrt(dx**2 + dy**2) / self.diagonal

        return distance

    def get_layout_position(self, layout_id: str) -> LayoutPosition | None:
        """Get layout position by ID."""
        for pos in self.layout.positions:
            if pos.id == layout_id:
                return pos
        return None
