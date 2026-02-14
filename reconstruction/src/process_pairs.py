from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
from loguru import logger

from cameras import (
    create_cameras_from_stereo_calibration,
    export_cameras_to_cloudcompare,
)
from stereo.image import (
    convert_image,
    crop_to_match_resolution,
    preprocess_image,
    resize_image,
)
from stereo.mast3r import MASt3RInferenceEngine
from stereo.saver import (
    render_match_correspondences,
    save_point_cloud_obj,
    save_point_cloud_ply,
)
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
    save_resized_frames=False,
    save_obj_file=False,
    subsample=None,
    save_match_render=False,
    match_render_output_folder=None,
):
    logger.info(f"Processing pair: {pair_name}")

    # Extract calibration resolution and normalize images to match it
    # IMPORTANT: calibration_params["image_size"] is the EXIF-corrected resolution
    # (the processed frame size used during calibration, after EXIF transpose and rotation)
    # Format: [width, height] -> target_W, target_H
    # We MUST match this EXIF-corrected resolution, not the original video resolution
    if "image_size" in calibration_params:
        calib_image_size = calibration_params["image_size"]
        target_W = calib_image_size[0]  # width (EXIF-corrected)
        target_H = calib_image_size[1]  # height (EXIF-corrected)
        logger.debug(
            f"Calibration image_size (EXIF-corrected): {calib_image_size} (W x H), "
            f"target: {target_W}x{target_H}"
        )

        # Normalize image_1 to calibration resolution if needed
        # Note: If image_1 comes from VideoReader, it's already EXIF-corrected
        if isinstance(image_1, np.ndarray):
            H1, W1 = image_1.shape[
                :2
            ]  # (height, width) - should be EXIF-corrected if from VideoReader
            if target_H != H1 or target_W != W1:
                logger.debug(
                    f"Cropping image_1 from {W1}x{H1} to match calibration {target_W}x{target_H}"
                )
                image_1 = crop_to_match_resolution(image_1, target_H, target_W)
                # Convert to PIL for preprocess_image
                image_1 = convert_image(image_1)
        else:
            # PIL Image - convert to numpy, crop, then convert back
            W1, H1 = image_1.size
            if target_H != H1 or target_W != W1:
                logger.debug(
                    f"Cropping image_1 from {W1}x{H1} to match calibration {target_W}x{target_H}"
                )
                image_1_array = np.array(image_1)
                image_1_array = crop_to_match_resolution(
                    image_1_array, target_H, target_W
                )
                image_1 = PIL.Image.fromarray(image_1_array)

        # Normalize image_2 to calibration resolution if needed
        # Note: If image_2 comes from VideoReader, it's already EXIF-corrected
        if isinstance(image_2, np.ndarray):
            H2, W2 = image_2.shape[
                :2
            ]  # (height, width) - should be EXIF-corrected if from VideoReader
            if target_H != H2 or target_W != W2:
                logger.debug(
                    f"Cropping image_2 from {W2}x{H2} to match calibration {target_W}x{target_H}"
                )
                image_2 = crop_to_match_resolution(image_2, target_H, target_W)
                # Convert to PIL for preprocess_image
                image_2 = convert_image(image_2)
        else:
            # PIL Image - convert to numpy, crop, then convert back
            W2, H2 = image_2.size
            if target_H != H2 or target_W != W2:
                logger.debug(
                    f"Cropping image_2 from {W2}x{H2} to match calibration {target_W}x{target_H}"
                )
                image_2_array = np.array(image_2)
                image_2_array = crop_to_match_resolution(
                    image_2_array, target_H, target_W
                )
                image_2 = PIL.Image.fromarray(image_2_array)
    else:
        logger.warning(
            "No image_size in calibration_params, skipping resolution normalization"
        )

    images = [
        preprocess_image(image_1, size=512, idx=0),
        preprocess_image(image_2, size=512, idx=1),
    ]

    # run inference
    output = mast3r_engine.run_inference(images)
    # extract raw data - use provided subsample or default
    subsample_value = subsample if subsample is not None else SUBSAMPLE
    raw_data = mast3r_engine.extract_raw_data(output, subsample=subsample_value)

    # load image1
    img1_pil = resize_image(image_1, size=512)
    if save_resized_frames:
        img1_pil.save(f"{output_folder}/{pair_name}_img1_resized.png")

    # save img2
    img2_pil = resize_image(image_2, size=512)
    if save_resized_frames:
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

    # Render and save match correspondences if requested
    if save_match_render:
        # Use specified match render output folder, or default to output_folder/match_renders
        if match_render_output_folder is not None:
            match_render_dir = Path(match_render_output_folder)
        elif output_folder is not None:
            match_render_dir = Path(output_folder) / "match_renders"
        else:
            match_render_dir = None

        if match_render_dir is not None:
            match_render_path = match_render_dir / f"{pair_name}_matches.png"
            render_match_correspondences(
                img1_pil,
                img2_pil,
                matches_im0,
                matches_im1,
                str(match_render_path),
                density_factor=32,
            )
            logger.info(f"Match correspondences rendered to {match_render_path}")

    # triangulate filtered pairs using resized image colors
    point_cloud = triangulate_points(matches_im0, matches_im1, calibration_params)

    # save point cloud in output folder
    colors_for_points = extract_colors_from_image(matches_im0, img1_array)
    save_point_cloud_ply(
        point_cloud, colors_for_points, f"{output_folder}/point_cloud_{pair_name}.ply"
    )
    # Also save as OBJ format for better Gradio Model3D compatibility
    if save_obj_file:
        save_point_cloud_obj(
            point_cloud,
            colors_for_points,
            f"{output_folder}/point_cloud_{pair_name}.obj",
        )

    if render_cameras:
        logger.info(f"Creating camera pyramids for pair: {pair_name}")
        try:
            camera1, camera2 = create_cameras_from_stereo_calibration(
                calibration_params, img1_array, img2_array
            )
            logger.info(f"Cameras created successfully: {camera1.name}, {camera2.name}")

            # Export camera pyramids in both PLY and OBJ formats
            logger.info("Exporting camera pyramids to PLY format...")
            ply_success = export_cameras_to_cloudcompare(
                [camera1, camera2],
                f"{output_folder}/camera_pyramids_{pair_name}",
                "ply",
            )
            logger.info(f"PLY export success: {ply_success}")

            if save_obj_file:
                logger.info("Exporting camera pyramids to OBJ format...")
                obj_success = export_cameras_to_cloudcompare(
                    [camera1, camera2],
                    f"{output_folder}/camera_pyramids_{pair_name}",
                    "obj",
                )
                logger.info(f"OBJ export success: {obj_success}")

        except Exception as e:
            logger.error(
                f"Error creating/exporting camera pyramids: {e!s}", exc_info=True
            )


def instantiate_sam() -> SAM:
    return SAM(SAM_MODEL_PATH)


def instantiate_mast3r_engine() -> MASt3RInferenceEngine:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mast3r_engine = MASt3RInferenceEngine(model_path=MODEL_PATH, device=DEVICE)
    mast3r_engine.load_model()
    return mast3r_engine
