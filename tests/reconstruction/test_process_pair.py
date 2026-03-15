import json
from pathlib import Path

import PIL.Image
from PIL.ImageOps import exif_transpose

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_process_pair():
    from process_pairs import instantiate_mast3r_engine, process_pair

    from stereo.convert_calibration import convert_calibration_parameters

    image_1_path = (
        PROJECT_ROOT / "assets" / "reconstruction" / "scene_10" / "camera_1.png"
    )
    image_2_path = (
        PROJECT_ROOT / "assets" / "reconstruction" / "scene_10" / "camera_2.png"
    )

    image_1 = exif_transpose(PIL.Image.open(image_1_path)).convert("RGB")
    image_2 = exif_transpose(PIL.Image.open(image_2_path)).convert("RGB")

    mast3r_engine = instantiate_mast3r_engine()

    with open(
        PROJECT_ROOT / "assets" / "reconstruction" / "scene_10" / "calibration.json"
    ) as f:
        calibration_data = json.load(f)

    # Convert calibration parameters to match web app behavior
    calibration_params = convert_calibration_parameters(
        calibration_data, target_size=512, patch_size=16
    )

    output_folder = PROJECT_ROOT / "output_tests" / "test_process_pair"
    output_folder.mkdir(parents=True, exist_ok=True)

    process_pair(
        image_1,
        image_2,
        mast3r_engine,
        None,
        calibration_params,
        "test_process_pair",
        output_folder=output_folder,
        subsample=4,
        render_cameras=True,
        save_match_render=True,
    )


def test_process_pair_with_sam():
    from process_pairs import instantiate_mast3r_engine, instantiate_sam, process_pair

    from stereo.convert_calibration import convert_calibration_parameters

    image_1_path = (
        PROJECT_ROOT / "assets" / "reconstruction" / "scene_10" / "camera_1.png"
    )
    image_2_path = (
        PROJECT_ROOT / "assets" / "reconstruction" / "scene_10" / "camera_2.png"
    )

    image_1 = exif_transpose(PIL.Image.open(image_1_path)).convert("RGB")
    image_2 = exif_transpose(PIL.Image.open(image_2_path)).convert("RGB")

    mast3r_engine = instantiate_mast3r_engine()
    sam = instantiate_sam()

    with open(
        PROJECT_ROOT / "assets" / "reconstruction" / "scene_10" / "calibration.json"
    ) as f:
        calibration_data = json.load(f)

    # Convert calibration parameters to match web app behavior
    calibration_params = convert_calibration_parameters(
        calibration_data, target_size=512, patch_size=16
    )

    output_folder = PROJECT_ROOT / "output_tests" / "test_process_pair_with_sam"
    output_folder.mkdir(parents=True, exist_ok=True)

    process_pair(
        image_1,
        image_2,
        mast3r_engine,
        sam,
        calibration_params,
        "test_process_pair_with_sam",
        point_prompt_1=(256, 144),
        point_prompt_2=(256, 144),
        output_folder=output_folder,
        subsample=16,
        save_match_render=True,
    )
