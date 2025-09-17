from mv_utils import Scene, ExtrinsicCalibration, load_stereo_data_folder_structure

stereo_data_folder_structure = load_stereo_data_folder_structure()

scene_names = stereo_data_folder_structure.get_scene_folders()


for scene_name in scene_names:
    
    scene = Scene(scene_name)
    print(f"Calibrating {scene_name}")
    extrinsic_calibration = scene.create_extrinsic_calibration()
    try:
        extrinsics = extrinsic_calibration.load_extrinsics()
        print(f"Calibration loaded for {scene_name}")
    except:
        print(f"Calibration not found for {scene_name}, computing...")
        extrinsics = extrinsic_calibration.calibrate_extrinsics()
        
    extrinsic_calibration.save_extrinsics_summary()
    print(f"Extrinsics saved for {scene_name}")