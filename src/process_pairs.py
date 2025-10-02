import os
from pathlib import Path
import json
import time

import cv2
import numpy as np
from loguru import logger

import torch

from cameras import (
    create_cameras_from_stereo_calibration,
    export_cameras_to_cloudcompare,
)
from stereo.convert_calibration import convert_calibration_parameters
from stereo.image import resize_image, load_image, preprocess_image
from stereo.mast3r import MASt3RInferenceEngine
from stereo.saver import save_point_cloud_ply, save_point_cloud_obj
from stereo.triangulation import (
    extract_colors_from_image,
    filter_pairs_with_mask,
    triangulate_points,
)
from unitaries.sam import SAM

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
SAM_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "FastSAM-x.pt"
RENDER_CAMERAS = True
SUBSAMPLE = 8


def process_pair(
    image_1,
    image_2,
    mast3r_engine,
    sam,
    calibration_params,
    pair_name,
    render_cameras=False,
    output_folder=None,
    point_prompt_1=None,
    point_prompt_2=None,
):
    logger.info(f"Processing pair: {pair_name}")
    
    
    images = [preprocess_image(image_1, size=512, idx=0), preprocess_image(image_2, size=512, idx=1)]
    
    
    
    # run inference
    output = mast3r_engine.run_inference(images)
    # extract raw data
    raw_data = mast3r_engine.extract_raw_data(output, subsample=SUBSAMPLE)

    # load image1
    img1_pil = resize_image(image_1, size=512)
    img1_pil.save(f"{output_folder}/{pair_name}_img1_resized.png")

    # save img2
    img2_pil = resize_image(image_2, size=512)
    img2_pil.save(f"{output_folder}/{pair_name}_img2_resized.png")

    matches_im0 = raw_data["matches_im0"]
    matches_im1 = raw_data["matches_im1"]
    img1_array = np.array(img1_pil)
    img2_array = np.array(img2_pil)
    if point_prompt_1 is not None:
        logger.info(f"Applying point prompt 1: {point_prompt_1} for filtering")
        # convert PIL Images to numpy arrays for SAM
        
        # infer with SAM and get masks for both images
        mask_result_img1 = sam.predict(img1_array, point=point_prompt_1)

        # save rendered images
        rendered_image1 = sam.render_result(
            img1_array, mask_result_img1["mask"], [mask_result_img1["point"]]
        )
        rendered_image1_brg = cv2.cvtColor(rendered_image1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"{output_folder}/{pair_name}_rendered_image1.png", rendered_image1_brg
        )
        # First filter: using mask from image 1 (camera 1)
        matches_im0, matches_im1 = filter_pairs_with_mask(
            matches_im0,
            matches_im1,
            mask_result_img1["mask"],
            camera=1,
        )

    if point_prompt_2 is not None:
        logger.info(f"Applying point prompt 2: {point_prompt_2} for filtering")
        mask_result_img2 = sam.predict(img2_array, point=point_prompt_2)
        rendered_image2 = sam.render_result(
            img2_array, mask_result_img2["mask"], [mask_result_img2["point"]]
        )
        rendered_image2_brg = cv2.cvtColor(rendered_image2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"{output_folder}/{pair_name}_rendered_image2.png", rendered_image2_brg
        )
        # Second filter: using mask from image 2 (camera 2) on the already filtered results
        matches_im0, matches_im1 = filter_pairs_with_mask(
            matches_im0,
            matches_im1,
            mask_result_img2["mask"],
            camera=2,
        )

    logger.info(f"Original matches: {len(raw_data['matches_im0'])}")
    logger.info(f"After camera 1 filtering: {len(matches_im0)}")
    logger.info(f"Final filtered matches (in both masks): {len(matches_im0)}")

    # triangulate filtered pairs using resized image colors
    point_cloud = triangulate_points(matches_im0, matches_im1, calibration_params)

    # save point cloud in output folder
    colors_for_points = extract_colors_from_image(matches_im0, img1_array)
    save_point_cloud_ply(
        point_cloud, colors_for_points, f"{output_folder}/point_cloud_{pair_name}.ply"
    )
    # Also save as OBJ format for better Gradio Model3D compatibility
    save_point_cloud_obj(
        point_cloud, colors_for_points, f"{output_folder}/point_cloud_{pair_name}.obj"
    )

    if render_cameras:
        camera1, camera2 = create_cameras_from_stereo_calibration(
            calibration_params, img1_array, img2_array
        )
        export_cameras_to_cloudcompare(
            [camera1, camera2], f"{output_folder}/camera_pyramids_{pair_name}", "ply"
        )


def instantiate_sam() -> SAM:
    return SAM(SAM_MODEL_PATH)


def instantiate_mast3r_engine() -> MASt3RInferenceEngine:
    mast3r_engine = MASt3RInferenceEngine(model_path=MODEL_PATH, device="cuda")
    mast3r_engine.load_model()
    return mast3r_engine


def process_scene(
    scene_name: str,
    sam: SAM,
    mast3r_engine: MASt3RInferenceEngine,
    point_prompt_1: tuple,
    point_prompt_2: tuple,
):
    INPUT_PAIR_FOLDER = PROJECT_ROOT / "output" / "extracted_pairs" / scene_name
    INPUT_CALIBRATION_FOLDER = INPUT_PAIR_FOLDER
    OUTPUT_FOLDER = PROJECT_ROOT / "output" / scene_name

    # create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    with open(f"{INPUT_CALIBRATION_FOLDER}/calibration.json") as f:
        calib_data = json.load(f)

    calibration_params = convert_calibration_parameters(
        calib_data, 
        original_size=(1920, 1080),
        target_size=512,
        patch_size=16
    )

    pair_folders = [
        f
        for f in os.listdir(INPUT_PAIR_FOLDER)
        if os.path.isdir(os.path.join(INPUT_PAIR_FOLDER, f))
    ]
    for pair_folder in pair_folders:
        pair_folder_path = os.path.join(INPUT_PAIR_FOLDER, pair_folder)
        pair_name = pair_folder
        image_1 = load_image(f"{pair_folder_path}/frame_1.png", size=512)
        image_2 = load_image(f"{pair_folder_path}/frame_2.png", size=512)
        process_pair(
            image_1,
            image_2,
            mast3r_engine,
            sam,
            calibration_params,
            pair_name,
            render_cameras=True,
            output_folder=OUTPUT_FOLDER,
            point_prompt_1=point_prompt_1,
            point_prompt_2=point_prompt_2,
        )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    #sam = instantiate_sam()
    mast3r_engine = instantiate_mast3r_engine()
    # make a 1 second sleep
    time.sleep(1)
    process_scene(
        "scene_8",
        None,
        mast3r_engine,
        point_prompt_1=None,
        point_prompt_2=None,
    )
    # process_scene("scene_3", sam, mast3r_engine, point_prompt_1=(256, 144), point_prompt_2=(150, 70))
    # process_scene("scene_7", sam, mast3r_engine, point_prompt_1=(300, 120), point_prompt_2=(256, 144))
