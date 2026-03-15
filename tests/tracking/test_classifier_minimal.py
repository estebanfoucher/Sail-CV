"""Minimal test for classifier to verify it's working correctly."""

import json
from pathlib import Path

import cv2

from classifyer import Classifier
from models.bounding_box import XYXY, BoundingBox
from models.classifier import ClassifierConfig

project_root = Path(__file__).resolve().parents[2]


def test_classifier_minimal():
    """Test classifier on crops extracted from first frame of fixture."""
    # Paths
    fixture_video = project_root / "fixtures" / "C1_fixture.mp4"
    fixture_results_json = (
        project_root / "output_tests" / "pipeline" / "C1_fixture_tracked.json"
    )
    output_dir = project_root / "output_tests" / "classifier_test"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load first frame from video
    cap = cv2.VideoCapture(str(fixture_video))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read first frame from {fixture_video}")

    print(f"✓ Loaded first frame: shape={frame.shape}")

    # Load results JSON to get bboxes from first frame
    with fixture_results_json.open() as f:
        results = json.load(f)

    first_frame = results[0]
    tracks = first_frame["tracks"]

    print(f"✓ Found {len(tracks)} tracks in first frame")

    # Initialize classifier
    classifier_config = ClassifierConfig(
        model_path=project_root / "checkpoints" / "classifyer_224.pt",
        padding_factor=0.25,
        confidence_threshold=0.0,
    )
    classifier = Classifier(classifier_config)
    print("✓ Classifier initialized")

    # Extract crops and classify
    classifications_results = []

    for idx, track in enumerate(tracks):
        # Get bbox from track
        bbox_dict = track["detection"]["bbox"]["xyxy"]
        bbox = BoundingBox(
            xyxy=XYXY(
                x1=int(bbox_dict["x1"]),
                y1=int(bbox_dict["y1"]),
                x2=int(bbox_dict["x2"]),
                y2=int(bbox_dict["y2"]),
            )
        )

        track_id = track["track_id"]

        # Extract padded crop
        img_height, img_width = frame.shape[:2]
        x1 = float(bbox.xyxy.x1)
        y1 = float(bbox.xyxy.y1)
        x2 = float(bbox.xyxy.x2)
        y2 = float(bbox.xyxy.y2)

        # Calculate padding
        padding_factor = 0.25
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        pad_x = bbox_width * padding_factor
        pad_y = bbox_height * padding_factor

        # Extend bbox with padding
        x1_padded = max(0, int(x1 - pad_x))
        y1_padded = max(0, int(y1 - pad_y))
        x2_padded = min(img_width, int(x2 + pad_x))
        y2_padded = min(img_height, int(y2 + pad_y))

        # Extract crop
        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]

        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            print(f"  Crop {idx} (track {track_id}): Invalid crop")
            continue

        # Classify crop
        class_id, confidence = classifier.classify_crop(crop)

        # Store result
        result = {
            "crop_number": idx,
            "track_id": track_id,
            "original_bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            },
            "padded_bbox": {
                "x1": x1_padded,
                "y1": y1_padded,
                "x2": x2_padded,
                "y2": y2_padded,
            },
            "crop_shape": list(crop.shape),
            "class_id": class_id,
            "confidence": confidence,
        }
        classifications_results.append(result)

        print(
            f"  Crop {idx} (track {track_id}): "
            f"shape={crop.shape}, "
            f"class_id={class_id}, "
            f"confidence={confidence:.3f}"
        )

        # Save crop image for inspection
        crop_filename = output_dir / f"crop_{idx}_track_{track_id}_class_{class_id}.jpg"
        cv2.imwrite(str(crop_filename), crop)

        # Also save with bbox drawn on original frame
        frame_with_bbox = frame.copy()
        cv2.rectangle(
            frame_with_bbox,
            (x1_padded, y1_padded),
            (x2_padded, y2_padded),
            (0, 255, 0) if class_id is not None else (0, 0, 255),
            2,
        )
        cv2.rectangle(
            frame_with_bbox,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            1,
        )
        cv2.putText(
            frame_with_bbox,
            f"{idx}: class={class_id} conf={confidence:.2f}"
            if class_id is not None
            else f"{idx}: no detection",
            (x1_padded, y1_padded - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        bbox_filename = output_dir / f"frame_with_bbox_{idx}.jpg"
        cv2.imwrite(str(bbox_filename), frame_with_bbox)

    # Save results JSON
    results_json_path = output_dir / "classifications.json"
    with results_json_path.open("w") as f:
        json.dump(classifications_results, f, indent=2)

    print(f"\n✓ Results saved to {results_json_path}")
    print(f"✓ Crop images saved to {output_dir}")

    # Summary
    total_crops = len(classifications_results)
    crops_with_class = sum(
        1 for r in classifications_results if r["class_id"] is not None
    )
    unique_classes = {
        r["class_id"] for r in classifications_results if r["class_id"] is not None
    }

    print("\nSummary:")
    print(f"  Total crops processed: {total_crops}")
    print(f"  Crops with classification: {crops_with_class}")
    print(f"  Unique class IDs: {sorted(unique_classes)}")

    if crops_with_class == 0:
        print(
            "\n⚠ WARNING: No crops were classified! The classifier may not be working correctly."
        )
    elif len(unique_classes) == 1 and 0 in unique_classes:
        print(
            "\n⚠ WARNING: All classifications returned class_id=0. This might indicate an issue."
        )
    else:
        print("\n✓ Classifier appears to be working correctly with multiple class IDs.")

    return classifications_results


if __name__ == "__main__":
    test_classifier_minimal()
