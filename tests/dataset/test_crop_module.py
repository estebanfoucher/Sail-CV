"""Test suite for crop module with YOLO dataset."""

from pathlib import Path
import sys

import cv2
import numpy as np

# Add src to path at the beginning to ensure correct imports
# Must be before any imports to avoid conflicts with tests/dataset package
project_root = Path(__file__).parent.parent.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import after path is set
from models import Image, ModelSpecs
from dataset import YOLODatasetLoader  # noqa: E402
from detector import Detector  # noqa: E402
from crop_module import CropModulePCA, MaskDetectorSAM  # noqa: E402
from crop_module.utils import extract_crop_from_bbox  # noqa: E402


def test_loader():
    """Step 1: Test dataset loading and verify with detector rendering."""
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "home_telltale"
    output_dir = project_root / "output_tests" / "dataset" / "step1_loader"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_random_pairs(10)

    print(f"Loaded {len(pairs)} image-detection pairs")

    # Initialize true detector (YOLO)
    model_path = project_root / "checkpoints" / "yolo-s.pt"
    specs = ModelSpecs(model_path=model_path, architecture="yolo")
    detector = Detector(specs)

    # Process each pair
    for idx, (image_name, image, detections) in enumerate(pairs):
        print(f"Processing {idx+1}/10: {image_name} ({len(detections)} detections)")

        # Render detections
        rendered_image = detector.render_result(image, detections)

        # Save rendered image
        output_path = output_dir / f"{idx:02d}_{image_name}"
        cv2.imwrite(str(output_path), rendered_image.to_bgr())

    print(f"✓ Step 1 complete: Saved {len(pairs)} rendered images to {output_dir}")


def test_crop_extraction():
    """Step 2: Test crop extraction pipeline."""
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "home_telltale"
    output_dir = project_root / "output_tests" / "dataset" / "step2_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_random_pairs(10)

    print(f"Extracting crops from {len(pairs)} image-detection pairs")

    # Process each pair
    for idx, (image_name, image, detections) in enumerate(pairs):
        print(f"Processing {idx+1}/10: {image_name} ({len(detections)} detections)")

        # Extract crops for each detection
        for det_idx, detection in enumerate(detections):
            crop = extract_crop_from_bbox(image, detection.bbox)

            # Save crop
            crop_filename = f"{idx:02d}_{image_name}_crop_{det_idx:02d}.jpg"
            output_path = output_dir / crop_filename
            cv2.imwrite(str(output_path), crop)

    print(f"✓ Step 2 complete: Saved crops to {output_dir}")


def test_mask_generation():
    """Step 2.5: Test mask generation with SAM."""
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "home_telltale"
    output_dir = project_root / "output_tests" / "dataset" / "step2_mask"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_random_pairs(10)

    print(f"Generating masks for {len(pairs)} image-detection pairs")

    # Initialize mask detector (SAM)
    try:
        mask_detector = MaskDetectorSAM(device="cpu")
        print("✓ MaskDetectorSAM initialized successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize MaskDetectorSAM: {e}")
        print("  Skipping mask generation test. Install mobile-sam or segment-anything to enable.")
        return

    # Process each pair
    for idx, (image_name, image, detections) in enumerate(pairs):
        print(f"Processing {idx+1}/10: {image_name} ({len(detections)} detections)")

        if not detections:
            continue

        try:
            # Generate all masks at once on full image
            bboxes = [det.bbox.to_numpy() for det in detections]
            masks = mask_detector.generate_masks(image.image, bboxes)
            print(f"  Generated {len(masks)} masks")

            # Render all masks on full image
            mask_rendered = mask_detector.render_masks(image.image, masks)

            # Save full image with mask visualization
            output_filename = f"{idx:02d}_{image_name}_masks.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), mask_rendered)

            # Also save individual crop masks for inspection
            for det_idx, (detection, mask) in enumerate(zip(detections, masks)):
                crop = extract_crop_from_bbox(image, detection.bbox)
                if crop.size == 0:
                    continue

                # Extract mask region for this crop
                x1, y1, x2, y2 = (
                    detection.bbox.xyxy.x1,
                    detection.bbox.xyxy.y1,
                    detection.bbox.xyxy.x2,
                    detection.bbox.xyxy.y2,
                )
                img_h, img_w = image.image.shape[:2]
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(x1 + 1, min(x2, img_w))
                y2 = max(y1 + 1, min(y2, img_h))
                crop_mask = mask[y1:y2, x1:x2]

                # Render mask on crop
                crop_mask_rendered = mask_detector.render_mask(crop, crop_mask)

                # Save individual crop mask
                crop_filename = f"{idx:02d}_{image_name}_crop_{det_idx:02d}_mask.jpg"
                crop_output_path = output_dir / crop_filename
                cv2.imwrite(str(crop_output_path), crop_mask_rendered)

        except Exception as e:
            print(f"  ⚠ Error generating masks: {e}")
            continue

    print(f"✓ Step 2.5 complete: Saved mask visualizations to {output_dir}")


def test_pca_results():
    """Step 3: Test PCA analysis and visualize results."""
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "home_telltale"
    output_dir = project_root / "output_tests" / "dataset" / "step3_pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_random_pairs(10)

    print(f"Running PCA analysis on {len(pairs)} image-detection pairs")

    # Initialize mask detector (SAM) - optional
    mask_detector = None
    try:
        mask_detector = MaskDetectorSAM(device="cpu")
        print("✓ Using MaskDetectorSAM for masked PCA")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize MaskDetectorSAM: {e}")
        print("  Running PCA without masks. Install mobile-sam or segment-anything to enable masks.")

    # Initialize PCA module with 2 components to get full 2D principal axis direction
    # Pass mask_detector if available
    pca_module = CropModulePCA(
        n_components=2, use_grayscale=True, mask_detector=mask_detector
    )

    # Process each pair
    for idx, (image_name, image, detections) in enumerate(pairs):
        print(f"Processing {idx+1}/10: {image_name} ({len(detections)} detections)")

        if not detections:
            continue

        # Get bounding boxes
        bboxes = [det.bbox for det in detections]

        # Run PCA analysis
        pca_results = pca_module.analyze_crop(image, bboxes)

        # Visualize PCA results on crops
        for det_idx, (detection, pca_result) in enumerate(zip(detections, pca_results)):
            # Extract crop
            crop = extract_crop_from_bbox(image, detection.bbox)

            if crop.size == 0:
                continue

            # Create visualization
            crop_viz = crop.copy()

            h, w = crop.shape[:2]
            center_x, center_y = w // 2, h // 2

            # Get principal axis direction (2D vector)
            if len(pca_result) >= 2:
                axis_x, axis_y = pca_result[0], pca_result[1]
            else:
                # Fallback if only 1 component
                axis_x, axis_y = pca_result[0] if len(pca_result) > 0 else 1.0, 0.0

            # Normalize the direction vector
            axis_length = np.sqrt(axis_x**2 + axis_y**2)
            if axis_length > 1e-6:
                axis_x_norm = axis_x / axis_length
                axis_y_norm = axis_y / axis_length
            else:
                axis_x_norm, axis_y_norm = 1.0, 0.0

            # Scale the arrow to be visible but not too long
            arrow_length = min(w, h) * 0.4
            end_x = int(center_x + axis_x_norm * arrow_length)
            end_y = int(center_y + axis_y_norm * arrow_length)

            # Draw the principal axis as a thick arrow
            # Draw main line
            cv2.arrowedLine(
                crop_viz,
                (center_x, center_y),
                (end_x, end_y),
                (0, 255, 0),  # Green color
                thickness=3,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )

            # Draw center point (start of arrow)
            cv2.circle(crop_viz, (center_x, center_y), 5, (0, 0, 255), -1)  # Red center point

            # Draw a perpendicular line to show it's an axis (optional, makes it more obvious)
            perp_length = min(w, h) * 0.1
            perp_x = int(center_x - axis_y_norm * perp_length)
            perp_y = int(center_y + axis_x_norm * perp_length)
            perp_x2 = int(center_x + axis_y_norm * perp_length)
            perp_y2 = int(center_y - axis_x_norm * perp_length)
            cv2.line(crop_viz, (perp_x, perp_y), (perp_x2, perp_y2), (255, 255, 0), 2)  # Yellow perpendicular

            # Add text with PCA values
            cv2.putText(
                crop_viz,
                f"PCA: ({axis_x:.2f}, {axis_y:.2f})",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Save crop with PCA visualization
            crop_filename = f"{idx:02d}_{image_name}_crop_{det_idx:02d}_pca.jpg"
            output_path = output_dir / crop_filename
            cv2.imwrite(str(output_path), crop_viz)

    print(f"✓ Step 3 complete: Saved PCA visualizations to {output_dir}")


def test_full_crop_module_pipeline():
    """Full pipeline test: Load -> Mask -> PCA on same examples."""
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "home_telltale"
    output_dir = project_root / "output_tests" / "dataset" / "step4_full_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset - use same pairs for consistency
    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_random_pairs(10)

    print(f"Running full pipeline on {len(pairs)} image-detection pairs")

    # Initialize mask detector (SAM)
    mask_detector = None
    try:
        mask_detector = MaskDetectorSAM(device="cpu")
        print("✓ MaskDetectorSAM initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize MaskDetectorSAM: {e}")
        print("  Running pipeline without masks.")

    # Initialize PCA module with mask detector
    pca_module = CropModulePCA(
        n_components=2, use_grayscale=True, mask_detector=mask_detector
    )

    # Process each pair through full pipeline
    for idx, (image_name, image, detections) in enumerate(pairs):
        print(f"Processing {idx+1}/10: {image_name} ({len(detections)} detections)")

        if not detections:
            continue

        # Step 1: Generate masks on full image
        bboxes = [det.bbox for det in detections]
        bbox_list = [bbox.to_numpy() for bbox in bboxes]

        all_masks = None
        if mask_detector is not None:
            try:
                all_masks = mask_detector.generate_masks(image.image, bbox_list)
                print(f"  Generated {len(all_masks)} masks")
            except Exception as e:
                print(f"  ⚠ Error generating masks: {e}")
                all_masks = None

        # Step 2: Run PCA analysis (which will use masks if available)
        pca_results = pca_module.analyze_crop(image, bboxes)

        # Step 3: Create comprehensive visualization
        # Full image with masks and bboxes
        full_viz = image.image.copy()

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = (
                int(det.bbox.xyxy.x1),
                int(det.bbox.xyxy.y1),
                int(det.bbox.xyxy.x2),
                int(det.bbox.xyxy.y2),
            )
            cv2.rectangle(full_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Overlay masks if available
        if all_masks is not None:
            full_viz = mask_detector.render_masks(full_viz, all_masks, alpha=0.3)

        # Save full image visualization
        full_output_path = output_dir / f"{idx:02d}_{image_name}_full.jpg"
        cv2.imwrite(str(full_output_path), full_viz)

        # Step 4: Create individual crop visualizations with mask and PCA
        for det_idx, (detection, pca_result) in enumerate(zip(detections, pca_results)):
            # Extract crop
            crop = extract_crop_from_bbox(image, detection.bbox)

            if crop.size == 0:
                continue

            # Extract mask for this crop
            crop_mask = None
            if all_masks is not None and det_idx < len(all_masks):
                full_mask = all_masks[det_idx]
                x1 = max(0, detection.bbox.xyxy.x1)
                y1 = max(0, detection.bbox.xyxy.y1)
                x2 = min(image.image.shape[1], detection.bbox.xyxy.x2)
                y2 = min(image.image.shape[0], detection.bbox.xyxy.y2)
                crop_mask = full_mask[y1:y2, x1:x2]

                # Ensure mask matches crop size
                crop_h, crop_w = crop.shape[:2]
                if crop_mask.shape != (crop_h, crop_w):
                    crop_mask = cv2.resize(
                        crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )

            # Create visualization: crop with mask overlay and PCA axis
            crop_viz = crop.copy()

            # Overlay mask if available
            if crop_mask is not None:
                # Apply mask overlay
                mask_normalized = (
                    crop_mask.astype(np.float32) / 255.0
                    if crop_mask.max() > 1
                    else crop_mask.astype(np.float32)
                )
                overlay = crop_viz.copy()
                overlay[mask_normalized > 0.5] = [0, 255, 0]  # Green
                crop_viz = cv2.addWeighted(crop_viz, 0.7, overlay, 0.3, 0)

            # Draw PCA axis
            h, w = crop.shape[:2]
            center_x, center_y = w // 2, h // 2

            if len(pca_result) >= 2:
                axis_x, axis_y = pca_result[0], pca_result[1]
            else:
                axis_x, axis_y = (
                    pca_result[0] if len(pca_result) > 0 else 1.0,
                    0.0,
                )

            # Normalize the direction vector
            axis_length = np.sqrt(axis_x**2 + axis_y**2)
            if axis_length > 1e-6:
                axis_x_norm = axis_x / axis_length
                axis_y_norm = axis_y / axis_length
            else:
                axis_x_norm, axis_y_norm = 1.0, 0.0

            # Draw principal axis arrow
            arrow_length = min(w, h) * 0.4
            end_x = int(center_x + axis_x_norm * arrow_length)
            end_y = int(center_y + axis_y_norm * arrow_length)

            cv2.arrowedLine(
                crop_viz,
                (center_x, center_y),
                (end_x, end_y),
                (0, 0, 255),  # Red arrow
                thickness=3,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )

            # Draw center point
            cv2.circle(crop_viz, (center_x, center_y), 5, (255, 0, 0), -1)

            # Add text
            mask_info = (
                f"mask: {crop_mask.mean() * 100:.1f}%"
                if crop_mask is not None
                else "no mask"
            )
            cv2.putText(
                crop_viz,
                f"PCA: ({axis_x:.2f}, {axis_y:.2f}) | {mask_info}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Save crop visualization
            crop_filename = (
                f"{idx:02d}_{image_name}_crop_{det_idx:02d}_full_pipeline.jpg"
            )
            crop_output_path = output_dir / crop_filename
            cv2.imwrite(str(crop_output_path), crop_viz)

    print(f"✓ Full pipeline complete: Saved visualizations to {output_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Testing dataset loader and rendering")
    print("=" * 60)
    test_loader()

    print("\n" + "=" * 60)
    print("Step 2: Testing crop extraction")
    print("=" * 60)
    test_crop_extraction()

    print("\n" + "=" * 60)
    print("Step 2.5: Testing mask generation")
    print("=" * 60)
    test_mask_generation()

    print("\n" + "=" * 60)
    print("Step 3: Testing PCA results")
    print("=" * 60)
    test_pca_results()

    print("\n" + "=" * 60)
    print("Step 4: Testing full pipeline")
    print("=" * 60)
    test_full_crop_module_pipeline()

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
    """Full pipeline test: Load -> Mask -> PCA on same examples."""
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "home_telltale"
    output_dir = project_root / "output_tests" / "dataset" / "step4_full_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset - use same pairs for consistency
    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_random_pairs(10)

    print(f"Running full pipeline on {len(pairs)} image-detection pairs")

    # Initialize mask detector (SAM)
    mask_detector = None
    try:
        mask_detector = MaskDetectorSAM(device="cpu")
        print("✓ MaskDetectorSAM initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize MaskDetectorSAM: {e}")
        print("  Running pipeline without masks.")

    # Initialize PCA module with mask detector
    pca_module = CropModulePCA(
        n_components=2, use_grayscale=True, mask_detector=mask_detector
    )

    # Process each pair through full pipeline
    for idx, (image_name, image, detections) in enumerate(pairs):
        print(f"Processing {idx+1}/10: {image_name} ({len(detections)} detections)")

        if not detections:
            continue

        # Step 1: Generate masks on full image
        bboxes = [det.bbox for det in detections]
        bbox_list = [bbox.to_numpy() for bbox in bboxes]

        all_masks = None
        if mask_detector is not None:
            try:
                all_masks = mask_detector.generate_masks(image.image, bbox_list)
                print(f"  Generated {len(all_masks)} masks")
            except Exception as e:
                print(f"  ⚠ Error generating masks: {e}")
                all_masks = None

        # Step 2: Run PCA analysis (which will use masks if available)
        pca_results = pca_module.analyze_crop(image, bboxes)

        # Step 3: Create comprehensive visualization
        # Full image with masks and bboxes
        full_viz = image.image.copy()

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = (
                int(det.bbox.xyxy.x1),
                int(det.bbox.xyxy.y1),
                int(det.bbox.xyxy.x2),
                int(det.bbox.xyxy.y2),
            )
            cv2.rectangle(full_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Overlay masks if available
        if all_masks is not None:
            full_viz = mask_detector.render_masks(full_viz, all_masks, alpha=0.3)

        # Save full image visualization
        full_output_path = output_dir / f"{idx:02d}_{image_name}_full.jpg"
        cv2.imwrite(str(full_output_path), full_viz)

        # Step 4: Create individual crop visualizations with mask and PCA
        for det_idx, (detection, pca_result) in enumerate(zip(detections, pca_results)):
            # Extract crop
            crop = extract_crop_from_bbox(image, detection.bbox)

            if crop.size == 0:
                continue

            # Extract mask for this crop
            crop_mask = None
            if all_masks is not None and det_idx < len(all_masks):
                full_mask = all_masks[det_idx]
                x1 = max(0, detection.bbox.xyxy.x1)
                y1 = max(0, detection.bbox.xyxy.y1)
                x2 = min(image.image.shape[1], detection.bbox.xyxy.x2)
                y2 = min(image.image.shape[0], detection.bbox.xyxy.y2)
                crop_mask = full_mask[y1:y2, x1:x2]

                # Ensure mask matches crop size
                crop_h, crop_w = crop.shape[:2]
                if crop_mask.shape != (crop_h, crop_w):
                    crop_mask = cv2.resize(
                        crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                    )

            # Create visualization: crop with mask overlay and PCA axis
            crop_viz = crop.copy()

            # Overlay mask if available
            if crop_mask is not None:
                # Apply mask overlay
                mask_normalized = (
                    crop_mask.astype(np.float32) / 255.0
                    if crop_mask.max() > 1
                    else crop_mask.astype(np.float32)
                )
                overlay = crop_viz.copy()
                overlay[mask_normalized > 0.5] = [0, 255, 0]  # Green
                crop_viz = cv2.addWeighted(crop_viz, 0.7, overlay, 0.3, 0)

            # Draw PCA axis
            h, w = crop.shape[:2]
            center_x, center_y = w // 2, h // 2

            if len(pca_result) >= 2:
                axis_x, axis_y = pca_result[0], pca_result[1]
            else:
                axis_x, axis_y = (
                    pca_result[0] if len(pca_result) > 0 else 1.0,
                    0.0,
                )

            # Normalize the direction vector
            axis_length = np.sqrt(axis_x**2 + axis_y**2)
            if axis_length > 1e-6:
                axis_x_norm = axis_x / axis_length
                axis_y_norm = axis_y / axis_length
            else:
                axis_x_norm, axis_y_norm = 1.0, 0.0

            # Draw principal axis arrow
            arrow_length = min(w, h) * 0.4
            end_x = int(center_x + axis_x_norm * arrow_length)
            end_y = int(center_y + axis_y_norm * arrow_length)

            cv2.arrowedLine(
                crop_viz,
                (center_x, center_y),
                (end_x, end_y),
                (0, 0, 255),  # Red arrow
                thickness=3,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )

            # Draw center point
            cv2.circle(crop_viz, (center_x, center_y), 5, (255, 0, 0), -1)            # Add text
            mask_info = (
                f"mask: {crop_mask.mean() * 100:.1f}%"
                if crop_mask is not None
                else "no mask"
            )
            cv2.putText(
                crop_viz,
                f"PCA: ({axis_x:.2f}, {axis_y:.2f}) | {mask_info}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Save crop visualization
            crop_filename = (
                f"{idx:02d}_{image_name}_crop_{det_idx:02d}_full_pipeline.jpg"
            )
            crop_output_path = output_dir / crop_filename
            cv2.imwrite(str(crop_output_path), crop_viz)

    print(f"✓ Full pipeline complete: Saved visualizations to {output_dir}")
