import sys
from pathlib import Path

import pytest
import json
from stereo.image import convert_image

def test_process_pair():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from process_pairs import process_pair, instantiate_mast3r_engine
    
    from video import VideoReader
    
    video_1_path = str(Path(__file__).parent.parent / "assets" / "scene_3" / "camera_1" / "camera_1.mp4")
    video_2_path = str(Path(__file__).parent.parent / "assets" / "scene_3" / "camera_2" / "camera_2.mp4")
    video_1_reader = VideoReader.open_video_file(video_1_path)
    video_2_reader = VideoReader.open_video_file(video_2_path)
    
    image_1 = video_1_reader.read()[1]
    image_2 = video_2_reader.read()[1]
    
    mast3r_engine = instantiate_mast3r_engine()
    
    with open(Path(__file__).parent.parent / "assets" / "scene_3" / "calibration.json", "r") as f:
        calibration_params = json.load(f)
    
    output_folder = Path(__file__).parent.parent / "output_tests" / "test_process_pair"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    process_pair(convert_image(image_1), convert_image(image_2), mast3r_engine, None, calibration_params, "test_process_pair", output_folder=output_folder, subsample=16, render_cameras=True)
    
    
def test_process_pair_with_sam():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from process_pairs import process_pair, instantiate_mast3r_engine, instantiate_sam
    
    from video import VideoReader
    
    video_1_path = str(Path(__file__).parent.parent / "assets" / "scene_3" / "camera_1" / "camera_1.mp4")
    video_2_path = str(Path(__file__).parent.parent / "assets" / "scene_3" / "camera_2" / "camera_2.mp4")
    video_1_reader = VideoReader.open_video_file(video_1_path)
    video_2_reader = VideoReader.open_video_file(video_2_path)
    
    image_1 = video_1_reader.read()[1]
    image_2 = video_2_reader.read()[1]
    
    mast3r_engine = instantiate_mast3r_engine()
    sam = instantiate_sam()
    
    with open(Path(__file__).parent.parent / "assets" / "scene_3" / "calibration.json", "r") as f:
        calibration_params = json.load(f)
    
    output_folder = Path(__file__).parent.parent / "output_tests" / "test_process_pair_with_sam"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    process_pair(convert_image(image_1), convert_image(image_2), mast3r_engine, sam, calibration_params, "test_process_pair_with_sam", point_prompt_1=(256, 144), point_prompt_2=(256, 144), output_folder=output_folder, subsample=16)
    