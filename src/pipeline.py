"""Crop module tracking pipeline as a computational block with memory."""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from crop_module import CropModulePCA, MaskDetectorSAM
from crop_module.background_detector_vpi import BackgroundDetectorVPI
from detector import Detector
from layout_tracker import LayoutTracker
from models import Image, Layout, ModelSpecs, PipelineConfig
from tracker_utils.render_tracks import draw_tracks


def make_json_serializable(obj):
    """Convert numpy arrays and Pydantic models to JSON-serializable format."""
    # Handle numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)

    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # Handle collections
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]

    return obj


class Pipeline:
    """
    Computational block with memory for tracking telltales with layout tracker and crop module PCA analysis.

    Processes frames one at a time and maintains state (EMA tracking, etc.).
    Designed to work with a Streamer that feeds frames incrementally.
    """

    def __init__(
        self, config: PipelineConfig, layout: Layout, project_root: Path | None = None
    ):
        """
        Initialize pipeline with configuration and layout.

        Args:
            config: Pipeline configuration
            layout: Layout object with predefined positions
            project_root: Project root directory for resolving relative paths
        """
        self.config = config
        self.layout = layout
        self.project_root = project_root or Path.cwd()

        # Resolve model path relative to project root
        model_path = config.detector.model_path
        if not model_path.is_absolute():
            model_path = self.project_root / model_path

        # Initialize detector
        logger.info(
            f"Initializing {config.detector.architecture} detector with {model_path}"
        )
        specs = ModelSpecs(
            model_path=model_path, architecture=config.detector.architecture
        )
        self.detector = Detector(specs)
        logger.info("✓ Detector initialized")

        # Device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")

        # Initialize mask detector (SAM2) - optional
        self.mask_detector = None
        logger.info(f"Initializing MaskDetectorSAM (SAM2) on {self.device}")
        try:
            self.mask_detector = MaskDetectorSAM(device=self.device)
            logger.info("✓ MaskDetectorSAM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MaskDetectorSAM: {e}")
            logger.warning(
                "Continuing without mask detector - PCA will run on full crops"
            )

        # Background detector will be initialized when video dimensions are known
        self.background_detector = None
        self.width = None
        self.height = None
        self.fps = None
        self.alpha_frame = None

        # Layout tracker will be initialized when video dimensions are known
        self.layout_tracker = None
        self.crop_module = None

        # Per-track EMA state for arrow sense temporal smoothing (memory)
        self.fusion_ema_by_track: dict[str, np.ndarray] = {}
        self.track_last_seen_frame: dict[str, int] = {}

        # Set direction prior if not present
        if self.layout.direction is None:
            self.layout.direction = (1.0, 0.0)
            logger.info("Set layout direction to (1.0, 0.0)")

        logger.info(
            f"Loaded layout with {len(layout.positions)} positions, direction={layout.direction}"
        )

    def initialize_for_video(self, width: int, height: int, fps: float):
        """
        Initialize components that require video dimensions.

        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Video frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps

        # Initialize background detector (required)
        self._initialize_background_detector(width, height)

        # Verify background detector was initialized successfully
        if self.background_detector is None:
            raise RuntimeError(
                "Background detector initialization failed. "
                "The pipeline requires BackgroundDetectorVPI to work correctly."
            )

        # Initialize layout tracker
        self._initialize_layout_tracker(width, height)

        # Initialize crop module
        self._initialize_crop_module()

        # Compute EMA alpha from tau
        dt_seconds = (1.0 / fps) if fps and fps > 0 else 0.0
        self.alpha_frame = (
            math.exp(-dt_seconds / self.config.arrow_sense.tau_seconds)
            if (dt_seconds > 0.0 and self.config.arrow_sense.tau_seconds > 0.0)
            else 0.0
        )

    def _initialize_background_detector(self, width: int, height: int):
        """
        Initialize background detector with video dimensions.

        Raises:
            RuntimeError: If background detector initialization fails
        """
        if self.background_detector is not None:
            return

        logger.info("Initializing BackgroundDetectorVPI")
        backend = self.config.background_detector.backend
        if backend == "cuda" and self.device != "cuda":
            logger.warning("CUDA requested but not available, using CPU")
            backend = "cpu"

        self.background_detector = BackgroundDetectorVPI(
            image_size=(width, height),
            backend=backend,
            learn_rate=self.config.background_detector.learn_rate,
        )
        logger.info("✓ BackgroundDetectorVPI initialized")

        # Ensure it was actually initialized
        if self.background_detector is None:
            raise RuntimeError(
                "BackgroundDetectorVPI initialization failed - detector is None"
            )

    def _initialize_layout_tracker(self, width: int, height: int):
        """Initialize layout tracker with video dimensions."""
        if self.layout_tracker is not None:
            return

        logger.info("Initializing LayoutTracker")
        lt_config = self.config.layout_tracker
        self.layout_tracker = LayoutTracker(
            layout=self.layout,
            width=width,
            height=height,
            alpha=lt_config.alpha,
            beta=lt_config.beta,
            max_distance=lt_config.max_distance,
            confidence_thresh=lt_config.confidence_thresh,
        )
        logger.info("✓ LayoutTracker initialized")

    def _initialize_crop_module(self):
        """Initialize crop module with layout direction."""
        if self.crop_module is not None:
            return

        logger.info("Initializing CropModulePCA")
        cm_config = self.config.crop_module
        self.crop_module = CropModulePCA(
            n_components=cm_config.n_components,
            use_grayscale=cm_config.use_grayscale,
            mask_detector=self.mask_detector,
            background_detector=None,  # Keep PCA axis computation unchanged
            layout_direction=self.layout.direction,
            mask_fusion_alpha=cm_config.mask_fusion_alpha,
            sam_fail_min_coverage=cm_config.sam_fail_min_coverage,
            sam_fail_max_coverage=cm_config.sam_fail_max_coverage,
            mask_fusion_eps=cm_config.mask_fusion_eps,
            use_motion_in_pca_mask=cm_config.use_motion_in_pca_mask,
            use_motion_for_direction=cm_config.use_motion_for_direction,
        )
        logger.info("✓ CropModulePCA initialized")

    def _mask_to_float01(self, mask: np.ndarray) -> np.ndarray:
        """Convert uint/bool/float mask to float32 in [0, 1]."""
        m = mask.astype(np.float32, copy=False)
        max_val = float(np.max(m)) if m.size else 0.0
        if max_val > 1.0:
            m = m / 255.0
        return np.clip(m, 0.0, 1.0)

    def _unit_normalize_vec2(self, v: np.ndarray) -> np.ndarray:
        """Return 2D unit vector; [0,0] if degenerate."""
        v2 = np.asarray(v, dtype=np.float32)[:2]
        n = float(np.linalg.norm(v2))
        if n <= self.config.arrow_sense.eps:
            return np.zeros(2, dtype=np.float32)
        return v2 / n

    def _resize_nearest(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize single-channel mask with nearest-neighbor."""
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def _compute_signed_unit_axis_for_track(
        self,
        *,
        axis_2d: np.ndarray,
        track_id: str,
        bbox,
        sam_full_mask_for_bbox: np.ndarray | None,
        motion_full_mask: np.ndarray | None,
    ) -> np.ndarray:
        """
        Determine a stable arrow sense (sign) for a PCA axis.

        Uses per-frame fusion weights:
          W_t = SAM_mask * normalize(Motion)
        and temporal smoothing:
          F_t = alpha*F_{t-1} + (1-alpha)*W_t

        Sense decision uses:
          score = sum(F_t * projection)
        where projection is the pixel coordinate projected onto the *unsigned* PCA axis.
        """
        axis_u = self._unit_normalize_vec2(axis_2d)
        if float(np.linalg.norm(axis_u)) <= self.config.arrow_sense.eps:
            return axis_u

        # Crop coordinates (match crop_module's bbox clipping)
        if self.width is None or self.height is None:
            raise RuntimeError(
                "Pipeline not initialized for video. Call initialize_for_video() first."
            )
        x1 = max(0, int(bbox.xyxy.x1))
        y1 = max(0, int(bbox.xyxy.y1))
        x2 = min(int(self.width), int(bbox.xyxy.x2))
        y2 = min(int(self.height), int(bbox.xyxy.y2))
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 1 or crop_h <= 1:
            return axis_u

        # Build per-frame fusion weight mask W_t in crop coords if possible.
        W_t = None
        if sam_full_mask_for_bbox is not None and motion_full_mask is not None:
            sam_crop = sam_full_mask_for_bbox[y1:y2, x1:x2]
            motion_crop = motion_full_mask[y1:y2, x1:x2]
            if sam_crop.size and motion_crop.size:
                sam_crop_f = self._mask_to_float01(sam_crop)
                motion_crop_f = self._mask_to_float01(motion_crop)
                motion_norm = motion_crop_f / (
                    float(np.max(motion_crop_f)) + self.config.arrow_sense.eps
                )
                W_t = (sam_crop_f * motion_norm).astype(np.float32)

        # Update / reuse EMA state.
        F_prev = self.fusion_ema_by_track.get(track_id)
        if F_prev is not None and F_prev.shape != (crop_h, crop_w):
            F_prev = self._resize_nearest(F_prev, crop_w, crop_h)

        if W_t is not None:
            if F_prev is None:
                F_t = W_t
            else:
                if self.alpha_frame is None:
                    raise RuntimeError(
                        "Pipeline not initialized for video. Call initialize_for_video() first."
                    )
                F_t = (
                    self.alpha_frame * F_prev + (1.0 - self.alpha_frame) * W_t
                ).astype(np.float32)
            self.fusion_ema_by_track[track_id] = F_t
        else:
            F_t = F_prev

        # No fusion weights available: cannot infer sense robustly.
        if F_t is None or float(np.sum(F_t)) <= self.config.arrow_sense.eps:
            return axis_u

        # Compute score = sum(weights * projection) in crop coordinates.
        ys, xs = np.mgrid[0:crop_h, 0:crop_w]
        cx = crop_w // 2
        cy = crop_h // 2
        proj = (xs - cx) * axis_u[0] + (ys - cy) * axis_u[1]

        fusion_threshold = self.config.arrow_sense.fusion_threshold
        if fusion_threshold > 0.0:
            sel = F_t > fusion_threshold
            if not np.any(sel):
                return axis_u
            score = float(np.sum(F_t[sel] * proj[sel]))
        else:
            score = float(np.sum(F_t * proj))

        return axis_u if score >= 0.0 else (-axis_u)

    def _purge_ema_state(self, frame_number: int):
        """Purge per-track EMA state for tracks not seen recently."""
        purge_state_after_frames = (
            round(self.config.arrow_sense.purge_state_after_seconds * self.fps)
            if self.fps and self.fps > 0
            else 0
        )

        if purge_state_after_frames > 0:
            for tid in list(self.track_last_seen_frame.keys()):
                if (
                    frame_number - self.track_last_seen_frame[tid]
                    > purge_state_after_frames
                ):
                    self.track_last_seen_frame.pop(tid, None)
                    self.fusion_ema_by_track.pop(tid, None)

    def process_frame(self, frame: np.ndarray, frame_number: int) -> dict:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Frame as numpy array (BGR format)
            frame_number: Frame number (0-indexed)

        Returns:
            Dictionary with:
            - frame_number: Frame number
            - tracks: List of Track objects
            - pca_vectors: Dict mapping track_id to PCA vector [dx, dy]
            - rendered_frame: (optional) Rendered frame if rendering enabled
            - movement_mask: (optional) Movement mask if background detector available
        """
        if self.width is None or self.height is None:
            raise RuntimeError(
                "Pipeline not initialized for video. Call initialize_for_video() first."
            )

        # Convert frame to Image model
        image = Image(image=frame, rgb_bgr="BGR")

        # Step 1: Detect objects
        detections = self.detector.detect(image)

        # Step 2: Track with layout
        tracks = self.layout_tracker.update(detections)

        # Step 3: Generate movement mask and masks once, then compute PCA
        pca_vectors: dict[int, list[float]] = {}
        masks = None
        movement_mask = None

        # Generate movement mask for full frame (required)
        if self.background_detector is None:
            raise RuntimeError(
                f"Background detector not initialized. Cannot process frame {frame_number}. "
                "Ensure initialize_for_video() was called successfully."
            )

        movement_mask = self.background_detector.generate_foreground_mask(image.image)
        logger.debug(f"Frame {frame_number}: Generated movement mask")

        if tracks:
            # Get bboxes from tracks
            bboxes = [track.detection.bbox for track in tracks]
            bbox_list = [bbox.to_numpy() for bbox in bboxes]

            # Generate masks on full image (only once)
            if self.mask_detector is not None:
                try:
                    masks = self.mask_detector.generate_masks(image.image, bbox_list)
                    logger.debug(f"Frame {frame_number}: Generated {len(masks)} masks")
                except Exception as e:
                    logger.warning(f"Frame {frame_number}: Mask generation failed: {e}")
                    masks = None

            # Run PCA analysis
            try:
                pca_results = self.crop_module.analyze_crop(
                    image,
                    bboxes,
                    precomputed_masks=masks,
                    precomputed_movement_mask=None,  # Sense is handled below
                )
                logger.debug(
                    f"Frame {frame_number}: Computed {len(pca_results)} PCA vectors"
                )

                # Associate PCA vectors to track IDs
                for bbox_idx, (track, pca_vector) in enumerate(
                    zip(tracks, pca_results, strict=False)
                ):
                    track_id_key = str(track.track_id)
                    sam_full_mask_for_bbox = (
                        masks[bbox_idx]
                        if (masks is not None and bbox_idx < len(masks))
                        else None
                    )

                    signed_axis = self._compute_signed_unit_axis_for_track(
                        axis_2d=pca_vector,
                        track_id=track_id_key,
                        bbox=track.detection.bbox,
                        sam_full_mask_for_bbox=sam_full_mask_for_bbox,
                        motion_full_mask=movement_mask,
                    )

                    # JSON output expects signed, 2D unit-normalized vector.
                    pca_vectors[track.track_id] = signed_axis.tolist()
                    self.track_last_seen_frame[track_id_key] = frame_number
                    logger.debug(
                        f"Frame {frame_number}: Track {track.track_id} -> PCA unsigned={pca_vector}, signed_unit={signed_axis}"
                    )
            except Exception as e:
                logger.warning(f"Frame {frame_number}: PCA computation failed: {e}")

        # Purge per-track EMA state for tracks not seen recently.
        self._purge_ema_state(frame_number)

        # Prepare result
        result = {
            "frame_number": frame_number,
            "tracks": tracks,
            "pca_vectors": pca_vectors,
        }

        # Optional rendering (memory intensive)
        if self.config.output.render_masks or self.config.output.render_arrows:
            rendered_frame = draw_tracks(
                image.to_bgr(),
                tracks,
                {0: {"name": "telltale", "color": (0, 255, 0)}},  # green
                show_confidence=True,
                show_class_name=False,
            )

            # Overlay masks if enabled
            if (
                self.config.output.render_masks
                and masks is not None
                and self.mask_detector is not None
            ):
                rendered_frame = self.mask_detector.render_masks(
                    rendered_frame, masks, alpha=0.3
                )

            # Draw arrows if enabled
            if self.config.output.render_arrows:
                vis_config = self.config.visualization
                for track in tracks:
                    track_id = track.track_id
                    if track_id in pca_vectors:
                        pca_vector = pca_vectors[track_id]
                        if isinstance(pca_vector, list) and len(pca_vector) >= 2:
                            dx, dy = pca_vector[0], pca_vector[1]
                            xyxy = track.detection.bbox.xyxy
                            center_x = int((xyxy.x1 + xyxy.x2) / 2)
                            center_y = int((xyxy.y1 + xyxy.y2) / 2)
                            end_x = int(center_x + dx * vis_config.arrow_scale)
                            end_y = int(center_y + dy * vis_config.arrow_scale)

                            cv2.arrowedLine(
                                rendered_frame,
                                (center_x, center_y),
                                (end_x, end_y),
                                vis_config.arrow_color,
                                vis_config.arrow_thickness,
                                tipLength=0.3,
                            )

            result["rendered_frame"] = rendered_frame

        # Include movement mask if available and needed
        if movement_mask is not None:
            result["movement_mask"] = movement_mask

        return result
