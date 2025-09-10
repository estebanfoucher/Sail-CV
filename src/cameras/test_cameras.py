import json
from cameras import create_cameras_from_stereo_calibration, export_cameras_to_cloudcompare

def test_normal():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/normal"
    
    # Load calibration data from JSON file
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    # Create cameras from calibration data
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")
    
def test_1m_translation_z():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/1m_translation_z"
    
    # Load calibration data from JSON file
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [0, 0, 1]
    calibration_data["rotation_matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Create cameras from calibration data
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

def test_1m_translation_y():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/1m_translation_y"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [0, 1, 0]
    calibration_data["rotation_matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")
    
def test_1m_translation_x():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/1m_translation_x"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [1, 0, 0]
    calibration_data["rotation_matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

def test_90_rotation_y():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/90_rotation_y"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    # translation to 0, 0, 0
    calibration_data["translation_vector"] = [0, 0, 0]
    calibration_data["rotation_matrix"] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

def test_90_rotation_x():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/90_rotation_x"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [0, 0, 0]
    calibration_data["rotation_matrix"] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

def test_90_rotation_z():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/90_rotation_z"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [0, 0, 0]
    calibration_data["rotation_matrix"] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

def test_180_rotation_y():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/180_rotation_y"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [0, 0, 0]
    calibration_data["rotation_matrix"] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

def test_90_rotation_x_and_1m_translation_z():
    extrinsics_calibration_path = "test_assets/extrinsics_calibration.json"
    image_1_path = "test_assets/images/frame_1.png"
    image_2_path = "test_assets/images/frame_2.png"
    output_dir = "test_assets/camera_pyramids/90_rotation_x_and_1m_translation_z"
    
    with open(extrinsics_calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    calibration_data["translation_vector"] = [0, 0, 1]
    calibration_data["rotation_matrix"] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    camera1, camera2 = create_cameras_from_stereo_calibration(calibration_data, image_1_path, image_2_path)
    export_cameras_to_cloudcompare([camera1, camera2], output_dir, "ply")

if __name__ == "__main__":
   test_normal()