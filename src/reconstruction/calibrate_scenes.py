from pathlib import Path

from calibration import Scene


def calibrate_scene(scene_name: str, stereo_data_folder_path: str | None = None):
    scene = Scene(scene_name, stereo_data_folder_path)
    scene.create_calibration()
    # Force recompute intrinsics to ensure orientation-corrected image size is used
    scene.calibration.compute_extrinsics_calibration(recompute_intrinsics=False)
    scene.calibration.save_calibration_summary()
    scene.cleanup()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    stereo_data_folder_path = str(project_root / "data")
    scene_name = "scene_15"
    calibrate_scene(scene_name, stereo_data_folder_path)
