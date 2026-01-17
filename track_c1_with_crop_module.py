"""Main script to track C1.mp4 with layout tracker and crop module pipeline."""

import json
import sys
from pathlib import Path

from loguru import logger

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from crop_module import CropModulePCA, MaskDetectorSAM
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


def main():
    """Main function to process C1.mp4 with layout tracker and crop module."""
    # Paths
    project_root = Path(__file__).parent
    video_path = project_root / "assets" / "C1.mp4"
    layout_path = project_root / "assets" / "layouts" / "C1_layout.json"
    # Use best.pt (assuming this is the best_1_class model)
    # If you have a specific best_1_class.pt file, update this path
    model_path = project_root / "checkpoints" / "best.pt"
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
    logger.info(f"Loaded layout with {len(layout.positions)} positions")

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

    logger.info(f"Video specs: {width}x{height} @ {fps} fps, {total_frames} frames")

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

    # Initialize mask detector (SAM2)
    logger.info("Initializing MaskDetectorSAM (SAM2)")
    try:
        mask_detector = MaskDetectorSAM(device="cpu")
        logger.info("✓ MaskDetectorSAM initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MaskDetectorSAM: {e}")
        logger.warning("Continuing without mask detector - PCA will run on full crops")
        mask_detector = None

    # Initialize crop module with PCA
    logger.info("Initializing CropModulePCA")
    crop_module = CropModulePCA(
        n_components=1, use_grayscale=True, mask_detector=mask_detector
    )
    logger.info("✓ CropModulePCA initialized")

    # Reopen video for processing
    reader = VideoReader.open_video_file(str(video_path))
    writer = FFmpegVideoWriter(
        str(output_video_path), reader.specs.fps, reader.specs.resolution
    )

    logger.info("=" * 60)
    logger.info("Processing video frames")
    logger.info("=" * 60)

    # Class info for rendering
    class_info = {
        0: {"name": "telltale", "color": (0, 255, 0)},  # green
    }

    # Process frames and collect results
    results_timeline = []
    frame_number = 0

    while True:
        ret, frame = reader.read()
        if not ret:
            break

        # Convert frame to Image model
        image = Image(image=frame, rgb_bgr="BGR")

        # Step 1: Detect objects
        detections = detector.detect(image)

        # Step 2: Track with layout
        tracks = layout_tracker.update(detections)

        # Step 3: Generate masks and compute PCA for tracked objects
        pca_vectors = {}
        if tracks:
            # Get bboxes from tracks
            bboxes = [track.detection.bbox for track in tracks]
            bbox_list = [bbox.to_numpy() for bbox in bboxes]

            # Generate masks on full image
            if mask_detector is not None:
                try:
                    masks = mask_detector.generate_masks(image.image, bbox_list)
                    logger.debug(
                        f"Frame {frame_number}: Generated {len(masks)} masks"
                    )
                except Exception as e:
                    logger.warning(f"Frame {frame_number}: Mask generation failed: {e}")
                    masks = None
            else:
                masks = None

            # Run PCA analysis
            try:
                pca_results = crop_module.analyze_crop(image, bboxes)
                logger.debug(f"Frame {frame_number}: Computed {len(pca_results)} PCA vectors")

                # Associate PCA vectors to track IDs
                for track, pca_vector in zip(tracks, pca_results):
                    pca_vectors[track.track_id] = pca_vector.tolist()
                    logger.debug(
                        f"Frame {frame_number}: Track {track.track_id} -> PCA: {pca_vector}"
                    )
            except Exception as e:
                logger.warning(f"Frame {frame_number}: PCA computation failed: {e}")

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
    logger.info(f"  - Total frames processed: {frame_number}")
    logger.info(
        f"  - Total track entries: {sum(len(frame['tracks']) for frame in results_timeline)}"
    )
    logger.info(
        f"  - Frames with PCA vectors: {sum(1 for f in results_timeline if f['pca_vectors'])}"
    )


if __name__ == "__main__":
    main()
