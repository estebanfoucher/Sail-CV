from pathlib import Path
from calibration import Scene
import pytest


def _is_raw_data_available():
    try:
        # Get project root (go up from src/ to project root)
        project_root = Path(__file__).parent.parent

        scene_name = "scene_3"
        stereo_data_folder_path = str(project_root / "data")

        # Create scene and calibrate intrinsics for both cameras
        scene = Scene(scene_name, stereo_data_folder_path)
        return True
    except Exception as e:
        return False


@pytest.mark.skipif(not _is_raw_data_available(), reason="Raw data not available")
def test_calibrate_intrinsics_scene():
    project_root = Path(__file__).parent.parent

    scene_name = "scene_3"
    stereo_data_folder_path = str(project_root / "data")

    # Create scene and calibrate intrinsics for both cameras
    scene = Scene(scene_name, stereo_data_folder_path)
    scene.create_calibration()

    # Create output directories
    output_dir = project_root / "output_tests" / "compute_intrinsics"
    (output_dir / "camera_1").mkdir(parents=True, exist_ok=True)
    (output_dir / "camera_2").mkdir(parents=True, exist_ok=True)

    scene.calibration._compute_intrinsics(
        "camera_1",
        save_path=str(output_dir / "camera_1" / "intrinsics.json"),
        temporal_calib_step_sec=10,
    )
    scene.calibration._compute_intrinsics(
        "camera_2",
        save_path=str(output_dir / "camera_2" / "intrinsics.json"),
        temporal_calib_step_sec=10,
    )

    assert (output_dir / "camera_1" / "intrinsics.json").exists(), (
        "Intrinsics file not found for camera 1"
    )
    assert (output_dir / "camera_2" / "intrinsics.json").exists(), (
        "Intrinsics file not found for camera 2"
    )

    # Cleanup resources
    scene.cleanup()

    print("Intrinsic calibration test passed")
