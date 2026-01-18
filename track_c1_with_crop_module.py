"""Main script to track C1.mp4 with layout tracker and crop module pipeline."""

import math
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from crop_module import CropModulePCA, MaskDetectorSAM
from crop_module.background_detector_vpi import BackgroundDetectorVPI
from detector import Detector
from layout_tracker import LayoutTracker
from models import Image, Layout, ModelSpecs
from tracker_utils.render_tracks import draw_tracks
from video import FFmpegVideoWriter, VideoReader


def make_json_serializable(obj):
    """Convert numpy arrays and Pydantic models to JSON-serializable format."""
    import numpy as np

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


def visualize_pca_vectors(
    video_path: Path,
    json_path: Path,
    output_path: Path,
    arrow_scale: float = 50.0,
    arrow_thickness: int = 2,
    arrow_color: tuple[int, int, int] = (0, 255, 255),  # Yellow in BGR
):
    """
    Visualize PCA vectors on video frames.

    Args:
        video_path: Path to raw video file
        json_path: Path to JSON file with tracking results and PCA vectors
        output_path: Path to save output video with vector visualization
        arrow_scale: Scale factor for arrow length (larger = longer arrows)
        arrow_thickness: Thickness of arrow lines
        arrow_color: Color of arrows in BGR format
    """
    logger.info("=" * 60)
    logger.info("Visualizing PCA vectors on video")
    logger.info("=" * 60)

    # Load JSON results
    logger.info(f"Loading results from {json_path}")
    with open(json_path) as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} frames of results")

    # Open video
    reader = VideoReader.open_video_file(str(video_path))
    writer = FFmpegVideoWriter(
        str(output_path), reader.specs.fps, reader.specs.resolution
    )

    logger.info(f"Processing video: {video_path.name}")
    logger.info(f"Output: {output_path.name}")

    frame_number = 0
    total_arrows_drawn = 0
    
    while True:
        ret, frame = reader.read()
        if not ret:
            break

        # Get results for current frame
        if frame_number < len(results):
            frame_result = results[frame_number]
            tracks = frame_result.get("tracks", [])
            pca_vectors = frame_result.get("pca_vectors", {})

            # Draw each track's bbox and PCA vector
            for track in tracks:
                # Extract track info
                if isinstance(track, dict):
                    track_id = track.get("track_id")
                    detection = track.get("detection", {})
                    bbox = detection.get("bbox", {})
                    xyxy = bbox.get("xyxy", {})
                else:
                    # Already converted objects
                    track_id = getattr(track, "track_id", None)
                    detection = getattr(track, "detection", None)
                    if detection:
                        bbox_obj = getattr(detection, "bbox", None)
                        if bbox_obj:
                            xyxy = getattr(bbox_obj, "xyxy", None)
                        else:
                            xyxy = None
                    else:
                        xyxy = None

                if xyxy is None:
                    continue

                # Get bbox coordinates
                if isinstance(xyxy, dict):
                    x1, y1 = int(xyxy.get("x1", 0)), int(xyxy.get("y1", 0))
                    x2, y2 = int(xyxy.get("x2", 0)), int(xyxy.get("y2", 0))
                else:
                    x1, y1 = int(xyxy.x1), int(xyxy.y1)
                    x2, y2 = int(xyxy.x2), int(xyxy.y2)

                # Calculate bbox center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw track ID
                label = f"ID:{track_id}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Draw PCA vector if available
                if track_id is not None and str(track_id) in pca_vectors:
                    pca_vector = pca_vectors[str(track_id)]

                    # PCA vector should be 2D: [dx, dy] representing the principal component
                    if isinstance(pca_vector, list) and len(pca_vector) >= 2:
                        dx, dy = pca_vector[0], pca_vector[1]
                        
                        # Scale the vector for visualization
                        end_x = int(center_x + dx * arrow_scale)
                        end_y = int(center_y + dy * arrow_scale)

                        # Draw arrow from center to scaled endpoint
                        cv2.arrowedLine(
                            frame,
                            (center_x, center_y),
                            (end_x, end_y),
                            arrow_color,
                            arrow_thickness,
                            tipLength=0.3,
                        )
                        
                        # Vector is expected to be unit-normalized in JSON.
                        pca_label = f"u=({dx:.2f},{dy:.2f})"
                        cv2.putText(
                            frame,
                            pca_label,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            arrow_color,
                            2,
                        )
                        
                        total_arrows_drawn += 1

        # Write frame
        writer.write(frame)

        frame_number += 1
        if frame_number % 100 == 0:
            logger.info(f"Processed {frame_number} frames, {total_arrows_drawn} arrows drawn so far")

    # Release resources
    writer.release()
    reader.release()

    logger.info("✓ PCA vector visualization completed")
    logger.info(f"  Output saved to: {output_path}")
    logger.info(f"  Total frames: {frame_number}")
    logger.info(f"  Total arrows drawn: {total_arrows_drawn}")


def main():
    """Main function to process C1.mp4 with layout tracker and crop module."""
    # Paths
    project_root = Path(__file__).parent
    video_path = project_root / "assets" / "C1.mp4"
    layout_path = project_root / "assets" / "layouts" / "C1_layout.json"
    # Use best.pt (assuming this is the best_1_class model)
    # If you have a specific best_1_class.pt file, update this path
    model_path = project_root / "checkpoints" / "best_1_class.pt"
    output_folder = project_root / "assets" / "processed"
    output_video_path = output_folder / "C1_crop_module_tracked.mp4"
    output_json_path = output_folder / "C1_crop_module_tracked.json"

    # Validate paths
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout not found: {layout_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Initializing pipeline components")
    logger.info("=" * 60)

    # Load layout
    logger.info(f"Loading layout from {layout_path}")
    with open(layout_path) as f:
        layout_data = json.load(f)
    layout = Layout.from_json_dict(layout_data)
    
    # Set direction prior for C1 (pointing right)
    if layout.direction is None:
        layout.direction = (1.0, 0.0)
        logger.info("Set layout direction to (1.0, 0.0) for C1")
    
    logger.info(f"Loaded layout with {len(layout.positions)} positions, direction={layout.direction}")

    # Initialize detector (RT-DETR)
    logger.info(f"Initializing RT-DETR detector with {model_path}")
    specs = ModelSpecs(model_path=model_path, architecture="rt-detr")
    detector = Detector(specs)
    logger.info("✓ Detector initialized")

    # Open video to get resolution
    reader = VideoReader.open_video_file(str(video_path))
    width, height = reader.specs.resolution
    fps = reader.specs.fps
    total_frames = reader.specs.frame_count
    reader.release()

    logger.info(f"Video specs: {width}x{height} @ {fps} fps, processing first {total_frames} frames")

    # ---------------------------------------------------------------------
    # Arrow sense temporal smoothing (EMA) parameters
    #
    # We keep PCA axis computation unchanged (unsigned). We only determine a
    # stable *sense* (sign) using:
    #   W_t = SAM_mask * normalize(Motion)
    #   F_t = alpha * F_{t-1} + (1 - alpha) * W_t
    #
    # alpha is derived from the EMA time constant tau in seconds:
    #   dt = 1/fps
    #   alpha = exp(-dt/tau)
    # ---------------------------------------------------------------------
    arrow_sense_tau_seconds = 0.2
    arrow_sense_eps = 1e-6
    fusion_threshold = 0.0  # 0 = use all pixels; >0 = ignore low-confidence weights
    purge_state_after_seconds = 0.5  # drop per-track EMA state after inactivity

    dt_seconds = (1.0 / fps) if fps and fps > 0 else 0.0
    alpha_frame = (
        math.exp(-dt_seconds / arrow_sense_tau_seconds)
        if (dt_seconds > 0.0 and arrow_sense_tau_seconds > 0.0)
        else 0.0
    )
    purge_state_after_frames = (
        int(round(purge_state_after_seconds * fps)) if fps and fps > 0 else 0
    )

    # Per-track EMA state in crop coordinates (resized as needed).
    # Track IDs in this pipeline are strings (e.g. "MT-C"), so keep keys as str.
    fusion_ema_by_track: dict[str, np.ndarray] = {}
    track_last_seen_frame: dict[str, int] = {}

    def _mask_to_float01(mask: np.ndarray) -> np.ndarray:
        """Convert uint/bool/float mask to float32 in [0, 1]."""
        m = mask.astype(np.float32, copy=False)
        max_val = float(np.max(m)) if m.size else 0.0
        if max_val > 1.0:
            m = m / 255.0
        return np.clip(m, 0.0, 1.0)

    def _unit_normalize_vec2(v: np.ndarray) -> np.ndarray:
        """Return 2D unit vector; [0,0] if degenerate."""
        v2 = np.asarray(v, dtype=np.float32)[:2]
        n = float(np.linalg.norm(v2))
        if n <= arrow_sense_eps:
            return np.zeros(2, dtype=np.float32)
        return v2 / n

    def _resize_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize single-channel mask with nearest-neighbor."""
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def _compute_signed_unit_axis_for_track(
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
        axis_u = _unit_normalize_vec2(axis_2d)
        if float(np.linalg.norm(axis_u)) <= arrow_sense_eps:
            return axis_u

        # Crop coordinates (match crop_module's bbox clipping)
        x1 = max(0, int(bbox.xyxy.x1))
        y1 = max(0, int(bbox.xyxy.y1))
        x2 = min(int(width), int(bbox.xyxy.x2))
        y2 = min(int(height), int(bbox.xyxy.y2))
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
                sam_crop_f = _mask_to_float01(sam_crop)
                motion_crop_f = _mask_to_float01(motion_crop)
                motion_norm = motion_crop_f / (
                    float(np.max(motion_crop_f)) + arrow_sense_eps
                )
                W_t = (sam_crop_f * motion_norm).astype(np.float32)

        # Update / reuse EMA state.
        F_prev = fusion_ema_by_track.get(track_id)
        if F_prev is not None and F_prev.shape != (crop_h, crop_w):
            F_prev = _resize_nearest(F_prev, crop_w, crop_h)

        if W_t is not None:
            if F_prev is None:
                F_t = W_t
            else:
                F_t = (alpha_frame * F_prev + (1.0 - alpha_frame) * W_t).astype(
                    np.float32
                )
            fusion_ema_by_track[track_id] = F_t
        else:
            F_t = F_prev

        # No fusion weights available: cannot infer sense robustly.
        if F_t is None or float(np.sum(F_t)) <= arrow_sense_eps:
            return axis_u

        # Compute score = sum(weights * projection) in crop coordinates.
        ys, xs = np.mgrid[0:crop_h, 0:crop_w]
        cx = crop_w // 2
        cy = crop_h // 2
        proj = (xs - cx) * axis_u[0] + (ys - cy) * axis_u[1]

        if fusion_threshold > 0.0:
            sel = F_t > fusion_threshold
            if not np.any(sel):
                return axis_u
            score = float(np.sum(F_t[sel] * proj[sel]))
        else:
            score = float(np.sum(F_t * proj))

        return axis_u if score >= 0.0 else (-axis_u)

    # Initialize layout tracker
    logger.info("Initializing LayoutTracker")
    layout_tracker = LayoutTracker(
        layout=layout,
        width=width,
        height=height,
        alpha=0.7,
        beta=0.3,
        max_distance=0.2,
        confidence_thresh=0.0,
    )
    logger.info("✓ LayoutTracker initialized")

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Initialize mask detector (SAM2)
    logger.info(f"Initializing MaskDetectorSAM (SAM2) on {device}")
    try:
        mask_detector = MaskDetectorSAM(device=device)
        logger.info("✓ MaskDetectorSAM initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MaskDetectorSAM: {e}")
        logger.warning("Continuing without mask detector - PCA will run on full crops")
        mask_detector = None

    # Initialize background detector for movement analysis
    logger.info("Initializing BackgroundDetectorVPI")
    try:
        background_detector = BackgroundDetectorVPI(
            image_size=(width, height),
            backend="cuda" if device == "cuda" else "cpu",
            learn_rate=0.01,
        )
        logger.info("✓ BackgroundDetectorVPI initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize BackgroundDetectorVPI: {e}")
        logger.warning("Continuing without background detector - arrow sense will use PCA axis only")
        background_detector = None

    # Initialize crop module with PCA (2 components for 2D vectors)
    logger.info("Initializing CropModulePCA with 2 components")
    crop_module = CropModulePCA(
        n_components=2,
        use_grayscale=True,
        mask_detector=mask_detector,
        # Keep PCA axis computation unchanged (no internal movement-based flipping).
        background_detector=None,
        layout_direction=layout.direction,
    )
    logger.info("✓ CropModulePCA initialized")

    # Reopen video for processing
    reader = VideoReader.open_video_file(str(video_path))
    writer = FFmpegVideoWriter(
        str(output_video_path), reader.specs.fps, reader.specs.resolution
    )
    
    # Create foreground mask video writer (raw mask, like outVideoFGMask in background.py)
    fgmask_output_path = output_folder / f"{video_path.stem}_fgmask.mp4"
    fgmask_writer = None
    
    if background_detector is not None:
        fgmask_writer = FFmpegVideoWriter(
            str(fgmask_output_path), reader.specs.fps, reader.specs.resolution
        )

    logger.info("=" * 60)
    logger.info("Processing video frames (first 30 frames)")
    logger.info("=" * 60)

    # Class info for rendering
    class_info = {
        0: {"name": "telltale", "color": (0, 255, 0)},  # green
    }

    # Process frames and collect results
    results_timeline = []
    frame_number = 0
    while frame_number < total_frames:
        ret, frame = reader.read()
        if not ret:
            break

        # Convert frame to Image model
        image = Image(image=frame, rgb_bgr="BGR")

        # Step 1: Detect objects
        detections = detector.detect(image)

        # Step 2: Track with layout
        tracks = layout_tracker.update(detections)

        # Step 3: Generate movement mask and masks once, then compute PCA
        pca_vectors: dict[int, list[float]] = {}
        masks = None
        movement_mask = None

        # Generate movement mask for full frame (if background detector available)
        if background_detector is not None:
            try:
                movement_mask = background_detector.generate_foreground_mask(image.image)
                logger.debug(f"Frame {frame_number}: Generated movement mask")
            except Exception as e:
                logger.warning(f"Frame {frame_number}: Movement mask generation failed: {e}")
        
        if tracks:
            # Get bboxes from tracks
            bboxes = [track.detection.bbox for track in tracks]
            bbox_list = [bbox.to_numpy() for bbox in bboxes]

            # Generate masks on full image (only once)
            masks = None
            if mask_detector is not None:
                try:
                    masks = mask_detector.generate_masks(image.image, bbox_list)
                    logger.debug(
                        f"Frame {frame_number}: Generated {len(masks)} masks"
                    )
                except Exception as e:
                    logger.warning(f"Frame {frame_number}: Mask generation failed: {e}")
                    masks = None

            # Run PCA analysis (pass masks and movement mask to avoid double inference)
            try:
                pca_results = crop_module.analyze_crop(
                    image,
                    bboxes,
                    precomputed_masks=masks,
                    # IMPORTANT: do not pass movement mask here; sense is handled below.
                    precomputed_movement_mask=None,
                )
                logger.debug(f"Frame {frame_number}: Computed {len(pca_results)} PCA vectors")

                # Associate PCA vectors to track IDs
                for bbox_idx, (track, pca_vector) in enumerate(zip(tracks, pca_results)):
                    track_id_key = str(track.track_id)
                    sam_full_mask_for_bbox = (
                        masks[bbox_idx]
                        if (masks is not None and bbox_idx < len(masks))
                        else None
                    )

                    signed_axis = _compute_signed_unit_axis_for_track(
                        axis_2d=pca_vector,
                        track_id=track_id_key,
                        bbox=track.detection.bbox,
                        sam_full_mask_for_bbox=sam_full_mask_for_bbox,
                        motion_full_mask=movement_mask,
                    )

                    # JSON output expects signed, 2D unit-normalized vector.
                    pca_vectors[track.track_id] = signed_axis.tolist()
                    track_last_seen_frame[track_id_key] = frame_number
                    logger.debug(
                        f"Frame {frame_number}: Track {track.track_id} -> PCA unsigned={pca_vector}, signed_unit={signed_axis}"
                    )
            except Exception as e:
                logger.warning(f"Frame {frame_number}: PCA computation failed: {e}")

        # Purge per-track EMA state for tracks not seen recently.
        if purge_state_after_frames > 0:
            for tid in list(track_last_seen_frame.keys()):
                if frame_number - track_last_seen_frame[tid] > purge_state_after_frames:
                    track_last_seen_frame.pop(tid, None)
                    fusion_ema_by_track.pop(tid, None)

        # Step 4: Render frame with tracks
        rendered_frame = draw_tracks(
            image.to_bgr(),
            tracks,
            class_info,
            show_confidence=True,
            show_class_name=False,
        )

        # Overlay masks if available
        if masks is not None and mask_detector is not None:
            rendered_frame = mask_detector.render_masks(rendered_frame, masks, alpha=0.3)

        # Write rendered frame
        writer.write(rendered_frame)

        # Write raw foreground mask to fgmask video (like outVideoFGMask in background.py)
        if movement_mask is not None and fgmask_writer is not None:
            # Convert binary mask to BGR format (3-channel grayscale)
            fgmask_bgr = cv2.cvtColor(movement_mask * 255, cv2.COLOR_GRAY2BGR)
            fgmask_writer.write(fgmask_bgr)
        
        # Store results
        frame_result = {
            "frame_number": frame_number,
            "tracks": tracks,
            "pca_vectors": pca_vectors,
        }
        results_timeline.append(frame_result)

        frame_number += 1
        if frame_number % 10 == 0 or frame_number == total_frames:
            logger.info(
                f"Processed frame {frame_number}/{total_frames} - "
                f"{len(tracks)} tracks, {len(pca_vectors)} PCA vectors"
            )

    # Release resources
    writer.release()
    if fgmask_writer is not None:
        fgmask_writer.release()
    reader.release()

    # Serialize and save results
    logger.info("Saving results to JSON")
    serializable_results = make_json_serializable(results_timeline)
    with open(output_json_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("✓ Processing completed successfully!")
    logger.info("=" * 60)
    logger.info(f"  - Output video: {output_video_path}")
    logger.info(f"  - Output JSON: {output_json_path}")
    if background_detector is not None:
        logger.info(f"  - Foreground mask video (raw): {fgmask_output_path}")
    logger.info(f"  - Total frames processed: {frame_number}")
    logger.info(
        f"  - Total track entries: {sum(len(frame['tracks']) for frame in results_timeline)}"
    )
    logger.info(
        f"  - Frames with PCA vectors: {sum(1 for f in results_timeline if f['pca_vectors'])}"
    )

    # Generate visualization video with PCA vectors
    logger.info("")
    logger.info("=" * 60)
    logger.info("Generate PCA vector visualization from existing JSON")
    logger.info("=" * 60)

    vector_output_path = output_folder / f"{video_path.stem}_pca_vectors.mp4"
    visualize_pca_vectors(
        video_path=video_path,
        json_path=output_json_path,
        output_path=vector_output_path,
        arrow_scale=50.0,  # Adjust to make arrows longer/shorter
        arrow_thickness=3,
        arrow_color=(0, 255, 255),  # Yellow arrows
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Visualization completed successfully")
    logger.info("=" * 60)
    logger.info(f"Output:")
    logger.info(f"  - PCA vectors video: {vector_output_path}")
    logger.info(f"Using data from:")
    logger.info(f"  - Raw video: {video_path}")
    logger.info(f"  - JSON results: {output_json_path}")


if __name__ == "__main__":
    main()
