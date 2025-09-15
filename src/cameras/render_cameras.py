#!/usr/bin/env python3
"""
Render Camera Pyramids from MASt3R output

Processes a MASt3R output folder (e.g., /mast3r/tmp/mast3r_output/) and renders
camera pyramids as PLY into `ply_rendered_cameras/` using ONLY the stereo calibration
file at `<output_root>/extrinsics_calibration_512x288.json` for geometry. The per-pair
images (resized) are used only for coloring; camera geometry stays the same across pairs.

For pair-structured outputs (output/pairs/<pair_name>/...), this script will:
- Load geometry from `<output_root>/extrinsics_calibration_512x288.json`
- Use resized images from `<pair_dir>/resized_frame/` (fallback to `inference_images/`)
- Export:
  - `<output_root>/ply_rendered_cameras/<pair_name>_frame_1_camera.ply`
  - `<output_root>/ply_rendered_cameras/<pair_name>_frame_2_camera.ply`

If no pairs are present, it renders a single set using the root's resized images
and names them with the folder name prefix.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple

from loguru import logger

# Ensure local imports work when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from cameras import Camera, create_cameras_from_stereo_calibration  # noqa: E402


def _map_container_to_host(path: str) -> str:
    # Map /mast3r/... to <project_root>/mast3r/... if needed
    if path.startswith('/mast3r'):
        return path.replace('/mast3r', os.path.join(PROJECT_ROOT, 'mast3r'), 1)
    return path


def _load_calibration(output_root: str) -> dict:
    calib_path = os.path.join(output_root, 'extrinsics_calibration_512x288.json')
    if not os.path.exists(calib_path):
        host_path = _map_container_to_host(calib_path)
        if os.path.exists(host_path):
            logger.info(f"Using host calibration path: {host_path}")
            calib_path = host_path
        else:
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with open(calib_path, 'r') as f:
        return json.load(f)


def _resolve_images_dir(pair_dir: str) -> str:
    # Prefer resized_frame, fallback to inference_images
    candidates = [
        os.path.join(pair_dir, 'resized_frame'),
        os.path.join(pair_dir, 'inference_images'),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"No resized or inference images found under {pair_dir}")


def _build_cameras_from_calibration(calibration: dict, images_dir: str) -> Tuple[Camera, Camera]:
    image_1_path = os.path.join(images_dir, 'frame_1.png')
    image_2_path = os.path.join(images_dir, 'frame_2.png')
    return create_cameras_from_stereo_calibration(calibration, image_1_path, image_2_path)


def _render_pair(pair_name: str, pair_dir: str, out_root: str, calibration: dict, pixel_sampling: int = 4, image_sampling: int = 4) -> None:
    start = time.time()
    images_dir = _resolve_images_dir(pair_dir)

    cam1, cam2 = _build_cameras_from_calibration(calibration, images_dir)

    render_dir = os.path.join(out_root, 'ply_rendered_cameras')
    os.makedirs(render_dir, exist_ok=True)

    out_cam1 = os.path.join(render_dir, f"{pair_name}_frame_1_camera.ply")
    out_cam2 = os.path.join(render_dir, f"{pair_name}_frame_2_camera.ply")

    logger.info(f"Rendering cameras for pair {pair_name} → {render_dir}")
    ok1 = cam1.export_to_cloudcompare_ply(out_cam1, pixel_sampling=pixel_sampling, image_sampling=image_sampling)
    ok2 = cam2.export_to_cloudcompare_ply(out_cam2, pixel_sampling=pixel_sampling, image_sampling=image_sampling)
    elapsed = time.time() - start
    logger.info(f"Rendered pair {pair_name}: cam1={'OK' if ok1 else 'FAIL'}, cam2={'OK' if ok2 else 'FAIL'} in {elapsed:.2f}s")


def render_cameras_for_output(output_root: str, pixel_sampling: int = 4, image_sampling: int = 4) -> None:
    output_root = os.path.abspath(output_root)
    # Map container path to host path if directory doesn't exist
    if not os.path.isdir(output_root):
        mapped = _map_container_to_host(output_root)
        if os.path.isdir(mapped):
            logger.info(f"Using host output root path: {mapped}")
            output_root = mapped
    logger.info(f"Rendering cameras from MASt3R output: {output_root}")
    calibration = _load_calibration(output_root)

    pairs_dir = os.path.join(output_root, 'pairs')
    if os.path.isdir(pairs_dir):
        pair_names = sorted([d for d in os.listdir(pairs_dir) if os.path.isdir(os.path.join(pairs_dir, d))])
        if not pair_names:
            logger.warning(f"No pair subfolders found under {pairs_dir}")
        for pair_name in pair_names:
            pair_dir = os.path.join(pairs_dir, pair_name)
            _render_pair(pair_name, pair_dir, output_root, calibration, pixel_sampling=pixel_sampling, image_sampling=image_sampling)
        return

    # Fallback: no pairs, treat root as a single pair-like directory
    images_dir = None
    for d in ('resized_frame', 'inference_images'):
        candidate = os.path.join(output_root, d)
        if os.path.isdir(candidate):
            images_dir = candidate
            break
    if images_dir is None:
        raise FileNotFoundError(f"No resized or inference images found under {output_root}")

    folder_name = os.path.basename(output_root.rstrip(os.sep))
    cam1, cam2 = _build_cameras_from_calibration(calibration, images_dir)

    render_dir = os.path.join(output_root, 'ply_rendered_cameras')
    os.makedirs(render_dir, exist_ok=True)

    out_cam1 = os.path.join(render_dir, f"{folder_name}_frame_1_camera.ply")
    out_cam2 = os.path.join(render_dir, f"{folder_name}_frame_2_camera.ply")

    logger.info(f"Rendering cameras for folder {folder_name} → {render_dir}")
    cam1.export_to_cloudcompare_ply(out_cam1, pixel_sampling=pixel_sampling, image_sampling=image_sampling)
    cam2.export_to_cloudcompare_ply(out_cam2, pixel_sampling=pixel_sampling, image_sampling=image_sampling)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Render camera pyramids from MASt3R output')
    parser.add_argument('--output_root', type=str, required=True, help='Path to MASt3R output root (e.g., /mast3r/tmp/mast3r_output)')
    parser.add_argument('--pixel_sampling', type=int, default=4, help='Sampling for wireframe (higher = sparser)')
    parser.add_argument('--image_sampling', type=int, default=4, help='Sampling for image base (higher = sparser)')
    args = parser.parse_args()

    render_cameras_for_output(args.output_root, pixel_sampling=args.pixel_sampling, image_sampling=args.image_sampling)


if __name__ == '__main__':
    main()


