# Pure Python ByteTrack implementation (adapted from Yolov5_DeepSort_Pytorch)
# Uses scipy.optimize.linear_sum_assignment for assignment (no lap required)
# MIT License

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Detection, Track


class STrack:
    """
    Single track object that stores a Detection

    Input:
        detection: Detection object
    """

    def __init__(self, detection: Detection):
        self.detection = detection
        self.track_id = None
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0

    def activate(self, frame_id: int, track_id: int):
        """Activate track with frame_id and track_id"""
        self.is_activated = True
        self.track_id = track_id
        self.start_frame = frame_id
        self.frame_id = frame_id

    def update(self, detection: Detection, frame_id: int):
        """Update track with new detection"""
        self.detection = detection
        self.frame_id = frame_id

    def to_xyxy(self):
        """Get XYXY coordinates from detection"""
        return self.detection.bbox.xyxy


class ByteTracker:
    """
    ByteTracker implementation using Detection objects and XYXY format
    """

    def __init__(
        self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        # Use lower threshold for matching lost tracks (more lenient for reconnection)
        self.match_thresh_lost = max(
            0.3, match_thresh - 0.3
        )  # Lower threshold for lost tracks
        self.frame_rate = frame_rate
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Update tracker with new detections

        Input:
            detections: List of Detection objects (Pydantic validated)

        Output:
            List[Track]: List of Track objects (Pydantic validated)
        """
        self.frame_id += 1
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Processing {len(detections)} detections"
        )
        activated_stracks = []
        removed_stracks = []

        # Convert detections to STrack objects
        det_stracks = []
        filtered_count = 0
        for detection in detections:
            if detection.confidence >= self.track_thresh:
                s = STrack(detection)
                det_stracks.append(s)
            else:
                filtered_count += 1
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Filtered {filtered_count} detections below threshold ({self.track_thresh}), {len(det_stracks)} detections above threshold"
        )

        # Remove lost tracks that have been lost for too long
        new_lost_stracks = []
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.track_buffer:
                logger.debug(
                    f"[ByteTracker] Frame {self.frame_id}: Removing lost track id={t.track_id} (lost for {self.frame_id - t.frame_id} frames, buffer={self.track_buffer})"
                )
                removed_stracks.append(t)
            else:
                new_lost_stracks.append(t)
        self.lost_stracks = new_lost_stracks
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: {len(self.tracked_stracks)} tracked tracks, {len(self.lost_stracks)} lost tracks (removed {len(removed_stracks)} old lost tracks)"
        )

        # Association with currently tracked tracks
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Associating {len(self.tracked_stracks)} tracked tracks with {len(det_stracks)} detections"
        )
        matches, u_track, u_detection = self._associate(
            self.tracked_stracks, det_stracks
        )
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Found {len(matches)} matches, {len(u_track)} unmatched tracks, {len(u_detection)} unmatched detections"
        )

        # Update matched tracks
        for t_idx, d_idx in matches:
            track = self.tracked_stracks[t_idx]
            det = det_stracks[d_idx]
            logger.debug(
                f"[ByteTracker] Frame {self.frame_id}: Matching track id={track.track_id} with detection (conf={det.detection.confidence:.3f}, class={det.detection.class_id})"
            )
            track.update(det.detection, self.frame_id)
            activated_stracks.append(track)

        # Unmatched tracks become lost
        for idx in u_track:
            track = self.tracked_stracks[idx]
            track.end_frame = self.frame_id
            logger.debug(
                f"[ByteTracker] Frame {self.frame_id}: Track id={track.track_id} unmatched, marking as lost"
            )
            self.lost_stracks.append(track)

        # Try to match unmatched detections to lost tracks (within buffer window)
        lost_candidates = [
            t
            for t in self.lost_stracks
            if self.frame_id - t.frame_id <= self.track_buffer
        ]
        lost_det_sublist = [det_stracks[i] for i in u_detection]
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Trying to match {len(lost_candidates)} lost tracks with {len(lost_det_sublist)} unmatched detections (using lower threshold {self.match_thresh_lost:.3f})"
        )
        matches_lost, _u_lost, _u_detection_new = self._associate(
            lost_candidates, lost_det_sublist, use_lost_thresh=True
        )
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Reactivated {len(matches_lost)} lost tracks"
        )

        # Collect lost tracks to reactivate and indices to remove after processing
        reactivated_lost_tracks = []
        lost_tracks_to_remove = []
        for l_idx, d_idx in matches_lost:
            if d_idx < len(u_detection):
                lost_track = lost_candidates[l_idx]
                det = det_stracks[u_detection[d_idx]]
                logger.debug(
                    f"[ByteTracker] Frame {self.frame_id}: Reactivating lost track id={lost_track.track_id} (was lost for {self.frame_id - lost_track.frame_id} frames)"
                )
                lost_track.update(det.detection, self.frame_id)
                lost_track.is_activated = True
                reactivated_lost_tracks.append(lost_track)
                lost_tracks_to_remove.append(lost_track)
            else:
                logger.error(
                    f"[ByteTracker] Frame {self.frame_id}: d_idx {d_idx} out of range for u_detection (len={len(u_detection)}). matches_lost={matches_lost}, u_detection={u_detection}, lost_candidates={len(lost_candidates)}, det_stracks={len(det_stracks)}"
                )

        # After processing all matches, update lists
        for lost_track in lost_tracks_to_remove:
            if lost_track in self.lost_stracks:
                self.lost_stracks.remove(lost_track)
        activated_stracks.extend(reactivated_lost_tracks)

        # Update unmatched detections after lost matching
        matched_lost_det_indices = [d for _, d in matches_lost]
        unmatched_detection_final = [
            idx
            for j, idx in enumerate(u_detection)
            if j not in matched_lost_det_indices
        ]

        # Activate new tracks for remaining unmatched detections
        for idx in unmatched_detection_final:
            det = det_stracks[idx]
            logger.debug(
                f"[ByteTracker] Frame {self.frame_id}: Activating new track id={self.next_id} (conf={det.detection.confidence:.3f}, class={det.detection.class_id})"
            )
            det.activate(self.frame_id, self.next_id)
            self.next_id += 1
            activated_stracks.append(det)

        self.tracked_stracks = activated_stracks

        # Remove lost tracks that have exceeded the buffer
        removed_lost_tracks = []
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.track_buffer:
                logger.debug(
                    f"[BYTETracker] Removing lost track id={t.track_id} (lost for {self.frame_id - t.frame_id} frames, buffer={self.track_buffer})"
                )
                removed_lost_tracks.append(t)
        for t in removed_lost_tracks:
            self.lost_stracks.remove(t)
            removed_stracks.append(t)

        # Prepare output as Track Pydantic models
        output_tracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                logger.debug(
                    f"[ByteTracker] Frame {self.frame_id}: Skipping unactivated track id={track.track_id}"
                )
                continue
            # Create Track from STrack's Detection and track_id
            track_obj = Track(
                detection=track.detection,
                track_id=track.track_id,
                frame_id=self.frame_id,
            )
            output_tracks.append(track_obj)
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Returning {len(output_tracks)} active tracks (total tracked: {len(self.tracked_stracks)}, lost: {len(self.lost_stracks)})"
        )
        return output_tracks

    def _associate(self, tracks, detections, use_lost_thresh=False):
        """Associate tracks with detections using IoU matching"""
        if len(tracks) == 0 or len(detections) == 0:
            logger.debug(
                f"[ByteTracker] Frame {self.frame_id}: Association skipped (tracks={len(tracks)}, detections={len(detections)})"
            )
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Use lower threshold for lost track matching
        threshold = self.match_thresh_lost if use_lost_thresh else self.match_thresh
        threshold_label = "lost" if use_lost_thresh else "active"

        cost_matrix = self.cost_matrix(tracks, detections)
        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Cost matrix shape: {cost_matrix.shape}, min={cost_matrix.min():.3f}, max={cost_matrix.max():.3f}, mean={cost_matrix.mean():.3f} (threshold={threshold:.3f} for {threshold_label} tracks)"
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind, strict=False):
            iou_val = self.iou(tracks[r], detections[c])
            cost_val = cost_matrix[r, c]
            if iou_val >= threshold:
                track_id = (
                    tracks[r].track_id if tracks[r].track_id is not None else "new"
                )
                logger.debug(
                    f"[ByteTracker] Frame {self.frame_id}: Match found - track[{r}] id={track_id} <-> det[{c}] (IoU={iou_val:.3f}, cost={cost_val:.3f}, threshold={threshold:.3f})"
                )
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)
            else:
                track_id = (
                    tracks[r].track_id if tracks[r].track_id is not None else "new"
                )
                logger.debug(
                    f"[ByteTracker] Frame {self.frame_id}: Match rejected - track[{r}] id={track_id} <-> det[{c}] (IoU={iou_val:.3f} < {threshold:.3f}, cost={cost_val:.3f})"
                )

        logger.debug(
            f"[ByteTracker] Frame {self.frame_id}: Association result - {len(matches)} matches, {len(unmatched_tracks)} unmatched tracks, {len(unmatched_detections)} unmatched detections"
        )
        return matches, unmatched_tracks, unmatched_detections

    @classmethod
    def iou(cls, track1: STrack, track2: STrack):
        """Calculate IoU between two STrack objects using their Detection bboxes"""
        xyxy1 = track1.detection.bbox.xyxy
        xyxy2 = track2.detection.bbox.xyxy
        x1, y1, x2, y2 = xyxy1.x1, xyxy1.y1, xyxy1.x2, xyxy1.y2
        x1g, y1g, x2g, y2g = xyxy2.x1, xyxy2.y1, xyxy2.x2, xyxy2.y2
        xa = max(x1, x1g)
        ya = max(y1, y1g)
        xb = min(x2, x2g)
        yb = min(y2, y2g)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - inter
        iou_val = inter / union if union > 0 else 0.0
        return iou_val

    @classmethod
    def cost_matrix(cls, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate cost matrix between tracks and detections of shape (len(tracks), len(detections))"""
        return np.array(
            [
                [1 - cls.iou(track, detection) for detection in detections]
                for track in tracks
            ]
        )
