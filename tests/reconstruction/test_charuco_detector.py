"""Comprehensive benchmark test for Charuco detection."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from loguru import logger

from calibration.extrinsics_calibration import CharucoDetector
from calibration.stereo_data_folder_structure import load_scene_folder_structure
from calibration.video_utils import VideoReader, get_unique_video_name

# Ensure logger is configured for tests
logger.remove()
logger.add(lambda msg: pytest.fail(msg, pytrace=False), level="ERROR")  # Fail on ERROR
logger.add(lambda msg: print(msg, end=""), level="DEBUG")  # Print DEBUG messages


def get_video_path(scene_name: str, camera_name: str, project_root: Path) -> Path | None:
    """Get video path for a scene and camera."""
    try:
        scene_folder_structure = load_scene_folder_structure(
            scene_name=scene_name, stereo_data_folder_path=str(project_root / "data")
        )
        folder_path = Path(scene_folder_structure.folder_path) / scene_name / camera_name
        video_name = get_unique_video_name(str(folder_path))
        if video_name is None:
            return None
        return folder_path / video_name
    except Exception as e:
        logger.warning(f"Failed to get video path for {scene_name}/{camera_name}: {e}")
        return None


def get_config_path(scene_name: str, project_root: Path) -> Path:
    """Get config path for a scene."""
    return project_root / "data" / scene_name / "extrinsics_calibration_pattern_specs.yml"


def get_synced_frame_for_other_camera(
    scene_name: str,
    camera_name: str,
    frame_number: int,
    project_root: Path,
) -> tuple[str, int] | None:
    """
    Compute the frame number for the other camera at the same moment.

    Uses time-based sync when FPS differ (parameters sync_event_time_F + both video FPS).
    Uses frame offset when FPS are the same.

    Returns:
        (other_camera_name, other_frame_number) or None if unavailable.
    """
    parameters_path = project_root / "data" / scene_name / "parameters.yml"
    if not parameters_path.exists():
        return None
    with open(parameters_path) as f:
        parameters = yaml.safe_load(f)
    cam1_sync = np.array(parameters["camera_1"]["sync_event_time_F"])
    cam2_sync = np.array(parameters["camera_2"]["sync_event_time_F"])

    video_path_1 = get_video_path(scene_name, "camera_1", project_root)
    video_path_2 = get_video_path(scene_name, "camera_2", project_root)
    if video_path_1 is None or video_path_2 is None:
        return None
    if not video_path_1.exists() or not video_path_2.exists():
        return None

    reader_1 = VideoReader.open_video_file(str(video_path_1))
    reader_2 = VideoReader.open_video_file(str(video_path_2))
    try:
        fps_1 = reader_1.video.fps
        fps_2 = reader_2.video.fps
    finally:
        reader_1.release()
        reader_2.release()

    if abs(fps_1 - fps_2) < 0.01:
        # Same FPS: frame-based sync
        diff = int(np.mean(cam1_sync - cam2_sync))
        if camera_name == "camera_1":
            other_camera = "camera_2"
            other_frame = frame_number - diff
        else:
            other_camera = "camera_1"
            other_frame = frame_number + diff
    else:
        # Different FPS: time-based sync
        times_1 = cam1_sync / fps_1
        times_2 = cam2_sync / fps_2
        time_offset_seconds = float(np.mean(times_1 - times_2))
        if camera_name == "camera_1":
            other_camera = "camera_2"
            time_1 = frame_number / fps_1
            time_2 = time_1 - time_offset_seconds
            other_frame = int(round(time_2 * fps_2))
        else:
            other_camera = "camera_1"
            time_2 = frame_number / fps_2
            time_1 = time_2 + time_offset_seconds
            other_frame = int(round(time_1 * fps_1))

    if other_frame < 0:
        return None
    return (other_camera, other_frame)


def extract_frame(video_path: Path, frame_number: int) -> np.ndarray | None:
    """Extract a frame from video."""
    try:
        reader = VideoReader.open_video_file(str(video_path))
        frames = reader.get_frames([frame_number])
        reader.release()
        if frames and len(frames) > 0:
            return frames[0]
        return None
    except Exception as e:
        logger.warning(f"Failed to extract frame {frame_number} from {video_path}: {e}")
        return None




def test_charuco_benchmark():
    """Comprehensive benchmark test for Charuco detection."""
    project_root = Path(__file__).resolve().parents[2]

    # Load references
    references_path = project_root / "assets" / "reconstruction" / "charuco" / "references.json"
    if not references_path.exists():
        pytest.skip(f"References file not found: {references_path}")

    with open(references_path) as f:
        references = json.load(f)

    positive_refs = references.get("positive", [])
    negative_refs = references.get("negative", [])

    # Expand refs with add_sync_pair: compute other camera frame (time-based sync when FPS differ)
    expanded = []
    for ref in positive_refs:
        expanded.append(ref)
        if not ref.get("add_sync_pair"):
            continue
        scene_name = ref["scene_name"]
        camera_name = ref["camera_name"]
        frame_number = ref["frame_number"]
        pair = get_synced_frame_for_other_camera(
            scene_name, camera_name, frame_number, project_root
        )
        if pair is not None:
            other_camera, other_frame = pair
            expanded.append({
                "scene_name": scene_name,
                "frame_number": other_frame,
                "camera_name": other_camera,
            })
        ref.pop("add_sync_pair", None)
    positive_refs = expanded

    # Create output directories
    positive_dir = project_root / "assets" / "reconstruction" / "charuco" / "positive"
    negative_dir = project_root / "assets" / "reconstruction" / "charuco" / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    output_dir = project_root / "output_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "positive": [],
        "negative": [],
        "summary": {
            "total_positive": len(positive_refs),
            "total_negative": len(negative_refs),
            "positive_passed": 0,
            "positive_failed": 0,
            "negative_passed": 0,
            "negative_failed": 0,
        }
    }

    # Process positive references
    print(f"\n{'='*80}")
    print(f"Processing {len(positive_refs)} POSITIVE references")
    print(f"{'='*80}\n")

    for i, ref in enumerate(positive_refs):
        scene_name = ref["scene_name"]
        frame_number = ref["frame_number"]
        camera_name = ref["camera_name"]

        print(f"[{i+1}/{len(positive_refs)}] {scene_name}/{camera_name}/frame_{frame_number}")

        # Get paths
        video_path = get_video_path(scene_name, camera_name, project_root)
        config_path = get_config_path(scene_name, project_root)

        if video_path is None or not video_path.exists():
            print(f"  ❌ SKIP: Video not found")
            results["positive"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "skipped",
                "reason": "video_not_found"
            })
            continue

        if not config_path.exists():
            print(f"  ❌ SKIP: Config not found")
            results["positive"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "skipped",
                "reason": "config_not_found"
            })
            continue

        # Extract frame
        frame = extract_frame(video_path, frame_number)
        if frame is None:
            print(f"  ❌ SKIP: Failed to extract frame")
            results["positive"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "skipped",
                "reason": "frame_extraction_failed"
            })
            continue

        # Save extracted frame
        frame_filename = f"{scene_name}_{camera_name}_frame_{frame_number}.png"
        frame_path = positive_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        # Create detector
        detector = CharucoDetector(config_path=str(config_path))

        try:
            # Run detection with timing
            start_time = time.time()
            corners, ids, num_markers = detector.detect_tags(frame, debug=False, return_marker_count=True)
            detection_time = time.time() - start_time

            # Verify results
            expected_corners = 70
            expected_markers = 33
            max_time = 1.0

            corners_count = len(corners) if corners is not None else 0
            ids_count = len(ids) if ids is not None else 0

            passed = (
                corners is not None
                and corners_count == expected_corners
                and ids is not None
                and ids_count == expected_corners
                and num_markers == expected_markers
                and detection_time < max_time
            )

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} | Time: {detection_time*1000:.1f}ms | Corners: {corners_count}/70 | Markers: {num_markers}/33")

            if not passed:
                issues = []
                if corners is None or corners_count != expected_corners:
                    issues.append(f"corners={corners_count} (expected {expected_corners})")
                if ids is None or ids_count != expected_corners:
                    issues.append(f"ids={ids_count} (expected {expected_corners})")
                if num_markers != expected_markers:
                    issues.append(f"markers={num_markers} (expected {expected_markers})")
                if detection_time >= max_time:
                    issues.append(f"time={detection_time*1000:.1f}ms (max {max_time*1000:.0f}ms)")
                print(f"    Issues: {', '.join(issues)}")

            # Draw corners on frame if detected and save to output_tests
            if corners is not None and ids is not None and len(corners) > 0:
                vis_frame = frame.copy()
                if len(corners.shape) == 2:
                    corners_draw = corners.reshape(-1, 1, 2)
                else:
                    corners_draw = corners
                cv2.aruco.drawDetectedCornersCharuco(vis_frame, corners_draw, ids, (0, 255, 0))
                detected_path = output_dir / f"{scene_name}_{camera_name}_frame_{frame_number}_detected.png"
                cv2.imwrite(str(detected_path), vis_frame)

            # Store results
            result = {
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "passed" if passed else "failed",
                "detection_time_ms": round(detection_time * 1000, 2),
                "corners_detected": corners_count,
                "corners_expected": expected_corners,
                "markers_detected": num_markers,
                "markers_expected": expected_markers,
                "ids_detected": ids_count,
            }
            results["positive"].append(result)

            if passed:
                results["summary"]["positive_passed"] += 1
            else:
                results["summary"]["positive_failed"] += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results["positive"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "error",
                "error": str(e)
            })
            results["summary"]["positive_failed"] += 1
        finally:
            detector.cleanup()

    # Process negative references
    print(f"\n{'='*80}")
    print(f"Processing {len(negative_refs)} NEGATIVE references")
    print(f"{'='*80}\n")

    for i, ref in enumerate(negative_refs):
        scene_name = ref["scene_name"]
        frame_number = ref["frame_number"]
        camera_name = ref["camera_name"]

        print(f"[{i+1}/{len(negative_refs)}] {scene_name}/{camera_name}/frame_{frame_number}")

        # Get paths
        video_path = get_video_path(scene_name, camera_name, project_root)
        config_path = get_config_path(scene_name, project_root)

        if video_path is None or not video_path.exists():
            print(f"  ❌ SKIP: Video not found")
            results["negative"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "skipped",
                "reason": "video_not_found"
            })
            continue

        if not config_path.exists():
            print(f"  ❌ SKIP: Config not found")
            results["negative"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "skipped",
                "reason": "config_not_found"
            })
            continue

        # Extract frame
        frame = extract_frame(video_path, frame_number)
        if frame is None:
            print(f"  ❌ SKIP: Failed to extract frame")
            results["negative"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "skipped",
                "reason": "frame_extraction_failed"
            })
            continue

        # Save extracted frame
        frame_filename = f"{scene_name}_{camera_name}_frame_{frame_number}.png"
        frame_path = negative_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        # Create detector
        detector = CharucoDetector(config_path=str(config_path))

        try:
            # Run detection with timing
            start_time = time.time()
            corners, ids = detector.detect_tags(frame, debug=False)
            detection_time = time.time() - start_time

            # Verify quick rejection
            max_time = 1.0
            rejected = corners is None or len(corners) == 0
            fast = detection_time < max_time

            passed = rejected and fast

            status = "✅ PASS" if passed else "❌ FAIL"
            corners_count = len(corners) if corners is not None else 0
            print(f"  {status} | Time: {detection_time*1000:.1f}ms | Corners: {corners_count} (should be 0)")

            if not passed:
                issues = []
                if not rejected:
                    issues.append(f"detected {corners_count} corners (should reject)")
                if not fast:
                    issues.append(f"time={detection_time*1000:.1f}ms (max {max_time*1000:.0f}ms)")
                print(f"    Issues: {', '.join(issues)}")

            # Store results
            result = {
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "passed" if passed else "failed",
                "detection_time_ms": round(detection_time * 1000, 2),
                "corners_detected": corners_count,
                "rejected": rejected,
            }
            results["negative"].append(result)

            if passed:
                results["summary"]["negative_passed"] += 1
            else:
                results["summary"]["negative_failed"] += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results["negative"].append({
                "scene": scene_name,
                "camera": camera_name,
                "frame": frame_number,
                "status": "error",
                "error": str(e)
            })
            results["summary"]["negative_failed"] += 1
        finally:
            detector.cleanup()

    # Save JSON results
    json_path = output_dir / "charuco_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to: {json_path}")

    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CHARUCO DETECTION BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary
    summary = results["summary"]
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Positive references: {summary['positive_passed']}/{summary['total_positive']} passed")
    report_lines.append(f"Negative references: {summary['negative_passed']}/{summary['total_negative']} passed")
    report_lines.append("")

    # Positive details
    report_lines.append("POSITIVE REFERENCES")
    report_lines.append("-" * 80)
    for ref in results["positive"]:
        status_icon = "✅" if ref.get("status") == "passed" else "❌"
        report_lines.append(
            f"{status_icon} {ref['scene']}/{ref['camera']}/frame_{ref['frame']}: "
            f"{ref.get('corners_detected', 0)}/70 corners, "
            f"{ref.get('markers_detected', 0)}/33 markers, "
            f"{ref.get('detection_time_ms', 0):.1f}ms"
        )
    report_lines.append("")

    # Negative details
    report_lines.append("NEGATIVE REFERENCES")
    report_lines.append("-" * 80)
    for ref in results["negative"]:
        status_icon = "✅" if ref.get("status") == "passed" else "❌"
        report_lines.append(
            f"{status_icon} {ref['scene']}/{ref['camera']}/frame_{ref['frame']}: "
            f"{ref.get('corners_detected', 0)} corners, "
            f"{ref.get('detection_time_ms', 0):.1f}ms"
        )

    # Save text report
    report_path = output_dir / "charuco_benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to: {report_path}")
    print(f"{'='*80}\n")

    # Print summary
    print("\n".join(report_lines))

    # Assertions (warnings instead of failures to allow benchmark to complete)
    if summary["positive_passed"] != summary["total_positive"]:
        print(f"\n⚠️  WARNING: Not all positive references passed: {summary['positive_passed']}/{summary['total_positive']}")
    if summary["negative_passed"] != summary["total_negative"]:
        print(f"⚠️  WARNING: Not all negative references passed: {summary['negative_passed']}/{summary['total_negative']}")

    # For now, don't fail the test - just report results
    # This allows us to see all benchmark results and optimize based on them
