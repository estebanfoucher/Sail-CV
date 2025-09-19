from mv_utils import Scene


def calibrate_scene(scene_name: str, stereo_data_folder_path: str = None):
    scene = Scene(scene_name, stereo_data_folder_path)
    calibration = scene.create_calibration()
    calibration.compute_extrinsics_calibration(recompute_intrinsics=False)
    calibration.save_calibration_summary()
    
    # Add explicit cleanup
    scene.cleanup()
    
    print(f"Scene {scene_name} passed")

if __name__ == "__main__":
    stereo_data_folder_path = "../data/"
    scene_name = "scene_3"
    calibrate_scene(scene_name, stereo_data_folder_path)