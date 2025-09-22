from pathlib import Path


def test_calibrate_intrinsics_scene():
    from mv_utils import Scene

    # Get project root (go up from src/ to project root)
    project_root = Path(__file__).parent.parent

    scene_name = "scene_3"
    stereo_data_folder_path = str(project_root / "data")

    # Create scene and calibrate intrinsics for both cameras
    scene = Scene(scene_name, stereo_data_folder_path)

    # Calibrate intrinsics for both cameras
    scene.calibrate_all_intrinsics()

    # Verify that intrinsics files were created
    for camera_name in ["camera_1", "camera_2"]:
        intrinsics_path = (
            Path(scene.scene_folder_structure.folder_path)
            / scene.scene_folder_structure.get_calibration_intrinsics_folder_name()
            / camera_name
            / "intrinsics.json"
        )
        assert intrinsics_path.exists(), f"Intrinsics file not found for {camera_name}"
        print(f"Intrinsic calibration saved to {intrinsics_path}")

    print("Intrinsic calibration test passed")
