import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from cameras import (
    create_cameras_from_stereo_calibration,
    export_cameras_to_cloudcompare,
)
from stereo.convert_calibration import convert_calibration_parameters
from stereo.image import resize_image
from stereo.mast3r import MASt3RInferenceEngine
from stereo.saver import save_point_cloud_ply
from stereo.triangulation import extract_colors_from_image, triangulate_points
from unitaries.sam import SAM

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
INPUT_PAIR_FOLDER = PROJECT_ROOT / "output" / "extracted_pairs" / "scene_8"
INPUT_CALIBRATION_FOLDER = PROJECT_ROOT / "output" / "extracted_pairs" / "scene_8"
OUTPUT_FOLDER = PROJECT_ROOT / "output" / "scene_8"
RENDER_CAMERAS = True
SUBSAMPLE = 8


def process_pair(
    mast3r_engine,
    sam,
    calibration_params,
    input_pair_folder_path,
    pair_name,
    render_cameras=False,
):
    logger.info(f"Processing pair: {pair_name}")

    # load images
    images = mast3r_engine.load_images(
        f"{input_pair_folder_path}/frame_1.png",
        f"{input_pair_folder_path}/frame_2.png",
        size=512,
    )

    # run inference
    output = mast3r_engine.run_inference(images)
    # extract raw data
    raw_data = mast3r_engine.extract_raw_data(output, subsample=SUBSAMPLE)

    # load image1
    img1_pil = resize_image(f"{input_pair_folder_path}/frame_1.png", size=512)
    img1_pil.save(f"{OUTPUT_FOLDER}/{pair_name}_img1_resized.png")

    # save img2
    img2_pil = resize_image(f"{input_pair_folder_path}/frame_2.png", size=512)
    img2_pil.save(f"{OUTPUT_FOLDER}/{pair_name}_img2_resized.png")

    # convert PIL Image to numpy array for SAM
    img1_array = np.array(img1_pil)

    # infer with SAM and get mask
    mask_result = sam.predict(img1_array, point=(256, 144))

    # save rendered image
    rendered_image = sam.render_result(
        img1_array, mask_result["mask"], [mask_result["point"]]
    )

    rendered_image_brg = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{OUTPUT_FOLDER}/{pair_name}_rendered_image.png", rendered_image_brg)

    if render_cameras:
        img2_array = np.array(img2_pil)
        camera1, camera2 = create_cameras_from_stereo_calibration(
            calibration_params, img1_array, img2_array
        )
        export_cameras_to_cloudcompare(
            [camera1, camera2], f"{OUTPUT_FOLDER}/camera_pyramids_{pair_name}", "ply"
        )

    # filter pairs with mask
    from stereo.triangulation import filter_pairs_with_mask

    matches_im0_filtered, matches_im1_filtered = filter_pairs_with_mask(
        raw_data["matches_im0"], raw_data["matches_im1"], mask_result["mask"]
    )

    # triangulate filtered pairs using resized image colors
    point_cloud = triangulate_points(
        matches_im0_filtered, matches_im1_filtered, calibration_params
    )

    # save point cloud in output folder
    colors_for_points = extract_colors_from_image(matches_im0_filtered, img1_array)
    save_point_cloud_ply(
        point_cloud, colors_for_points, f"{OUTPUT_FOLDER}/point_cloud_{pair_name}.ply"
    )


def main():
    # create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # instanciate sam
    sam = SAM(model_path="FastSAM-x.pt")

    # instanciate mast3r
    mast3r_engine = MASt3RInferenceEngine(model_path=MODEL_PATH, device="cuda")
    mast3r_engine.load_model()

    # convert calibration parameters
    import json

    with open(f"{INPUT_CALIBRATION_FOLDER}/calibration.json") as f:
        calib_data = json.load(f)

    calibration_params = convert_calibration_parameters(
        calib_data, original_size=(1920, 1080)
    )

    pair_folders = [
        f
        for f in os.listdir(INPUT_PAIR_FOLDER)
        if os.path.isdir(os.path.join(INPUT_PAIR_FOLDER, f))
    ]
    for pair_folder in pair_folders:
        pair_folder_path = os.path.join(INPUT_PAIR_FOLDER, pair_folder)
        pair_name = pair_folder
        process_pair(
            mast3r_engine,
            sam,
            calibration_params,
            pair_folder_path,
            pair_name,
            render_cameras=True,
        )


if __name__ == "__main__":
    main()
