from models import Detection, Track, TrackerConfig
from tracker_utils.byte_tracker import ByteTracker


class Tracker:
    """
    Tracker service class for object tracking

    Input:
        config: TrackerConfig object containing tracker configuration

    The tracker's update method:
        Input: List[Detection] objects
        Output: List[Track] objects
    """

    def __init__(self, config: TrackerConfig):
        """
        Initialize Tracker with tracker configuration

        Input:
            config: TrackerConfig object (Pydantic validated)
        """
        self.config = config
        self._byte_tracker: ByteTracker | None = None

    @property
    def byte_tracker(self) -> ByteTracker:
        """Get the ByteTracker instance (lazy initialization)"""
        if self._byte_tracker is None:
            self._byte_tracker = ByteTracker(
                track_thresh=self.config.track_thresh,
                track_buffer=self.config.track_buffer,
                match_thresh=self.config.match_thresh,
            )
        return self._byte_tracker

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Update tracker with new detections

        Input:
            detections: List of Detection objects (Pydantic validated)

        Output:
            List[Track]: List of Track objects (Pydantic validated)
        """
        return self.byte_tracker.update(detections)
