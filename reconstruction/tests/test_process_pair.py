import sys
from pathlib import Path

import pytest
import json
import PIL.Image
from PIL.ImageOps import exif_transpose

def test_process_pair():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from process_pairs import process_pair, instantiate_mast3r_engine
    from stereo.convert_calibration import convert_calibration_parameters

    image_1_path = Path(__file__).parent.parent / "assets" / "scene_10" / "camera_1.png"
    image_2_path = Path(__file__).parent.parent / "assets" / "scene_10" / "camera_2.png"

    image_1 = exif_transpose(PIL.Image.open(image_1_path)).convert("RGB")
    image_2 = exif_transpose(PIL.Image.open(image_2_path)).convert("RGB")

    mast3r_engine = instantiate_mast3r_engine()

    with open(Path(__file__).parent.parent / "assets" / "scene_10" / "calibration.json", "r") as f:
        calibration_data = json.load(f)

    # Convert calibration parameters to match web app behavior
    calibration_params = convert_calibration_parameters(calibration_data, target_size=512, patch_size=16)

    output_folder = Path(__file__).parent.parent / "output_tests" / "test_process_pair"
    output_folder.mkdir(parents=True, exist_ok=True)

    process_pair(image_1, image_2, mast3r_engine, None, calibration_params, "test_process_pair", output_folder=output_folder, subsample=4, render_cameras=True, save_match_render=True)


def test_process_pair_with_sam():
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from process_pairs import process_pair, instantiate_mast3r_engine, instantiate_sam
    from stereo.convert_calibration import convert_calibration_parameters

    image_1_path = Path(__file__).parent.parent / "assets" / "scene_10" / "camera_1.png"
    image_2_path = Path(__file__).parent.parent / "assets" / "scene_10" / "camera_2.png"

    image_1 = exif_transpose(PIL.Image.open(image_1_path)).convert("RGB")
    image_2 = exif_transpose(PIL.Image.open(image_2_path)).convert("RGB")

    mast3r_engine = instantiate_mast3r_engine()
    sam = instantiate_sam()

    with open(Path(__file__).parent.parent / "assets" / "scene_10" / "calibration.json", "r") as f:
        calibration_data = json.load(f)

    # Convert calibration parameters to match web app behavior
    calibration_params = convert_calibration_parameters(calibration_data, target_size=512, patch_size=16)

    output_folder = Path(__file__).parent.parent / "output_tests" / "test_process_pair_with_sam"
    output_folder.mkdir(parents=True, exist_ok=True)

    process_pair(image_1, image_2, mast3r_engine, sam, calibration_params, "test_process_pair_with_sam", point_prompt_1=(256, 144), point_prompt_2=(256, 144), output_folder=output_folder, subsample=16, save_match_render=True)
