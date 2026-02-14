"""CLI script to reconstruct a 3D point cloud from a calibrated stereo image pair."""

import argparse
import json
from pathlib import Path

import PIL.Image
import torch
from loguru import logger
from PIL.ImageOps import exif_transpose
from process_pairs import process_pair

from stereo.convert_calibration import convert_calibration_parameters
from stereo.mast3r import MASt3RInferenceEngine
from unitaries.sam import SAM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAST3R_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "FastSAM-x.pt"


def main():
    """Main function to reconstruct a 3D point cloud from a stereo image pair."""
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D point cloud from a calibrated stereo image pair"
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Scene name (e.g. scene_10). Loads images and calibration from assets/reconstruction/<scene>/",
    )
    parser.add_argument(
        "--image1",
        type=Path,
        help="Path to first camera image (overrides --scene)",
    )
    parser.add_argument(
        "--image2",
        type=Path,
        help="Path to second camera image (overrides --scene)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Path to calibration JSON (overrides --scene)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--mast3r-checkpoint",
        type=Path,
        default=DEFAULT_MAST3R_CHECKPOINT,
        help="Path to MASt3R model checkpoint",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Path to FastSAM checkpoint (enables SAM masking)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=8,
        help="Subsample factor for point correspondences (default: 8)",
    )
    parser.add_argument(
        "--render-cameras",
        action="store_true",
        help="Export camera pyramids alongside point cloud",
    )
    parser.add_argument(
        "--save-matches",
        action="store_true",
        help="Save match correspondence renders",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Target long edge for MASt3R inference (default: 512)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for MASt3R alignment (default: 16)",
    )

    args = parser.parse_args()

    # --- Resolve image paths and calibration ---
    if args.scene:
        scene_dir = PROJECT_ROOT / "assets" / "reconstruction" / args.scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        image1_path = scene_dir / "camera_1.png"
        image2_path = scene_dir / "camera_2.png"
        calibration_path = scene_dir / "calibration.json"
        pair_name = args.scene
    elif args.image1 and args.image2 and args.calibration:
        image1_path = args.image1
        image2_path = args.image2
        calibration_path = args.calibration
        pair_name = image1_path.stem
    else:
        parser.error(
            "Provide either --scene or all of --image1, --image2, --calibration"
        )

    # Validate inputs
    for path, label in [
        (image1_path, "Image 1"),
        (image2_path, "Image 2"),
        (calibration_path, "Calibration"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not args.mast3r_checkpoint.exists():
        raise FileNotFoundError(
            f"MASt3R checkpoint not found: {args.mast3r_checkpoint}"
        )

    # --- Setup ---
    output_folder = args.output / pair_name
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sail-CV 3D Reconstruction")
    logger.info("=" * 60)
    logger.info(f"Image 1:      {image1_path}")
    logger.info(f"Image 2:      {image2_path}")
    logger.info(f"Calibration:  {calibration_path}")
    logger.info(f"Output:       {output_folder}")
    logger.info(f"Subsample:    {args.subsample}")
    logger.info("=" * 60)

    # --- Load images ---
    image_1 = exif_transpose(PIL.Image.open(image1_path)).convert("RGB")
    image_2 = exif_transpose(PIL.Image.open(image2_path)).convert("RGB")
    logger.info(f"Images loaded: {image_1.size}, {image_2.size}")

    # --- Load and convert calibration ---
    with open(calibration_path) as f:
        calibration_data = json.load(f)
    calibration_params = convert_calibration_parameters(
        calibration_data, target_size=args.target_size, patch_size=args.patch_size
    )
    logger.info("Calibration loaded and converted")

    # --- Load MASt3R engine ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    mast3r_engine = MASt3RInferenceEngine(
        model_path=args.mast3r_checkpoint, device=device
    )
    mast3r_engine.load_model()

    # --- Optionally load SAM ---
    sam = None
    if args.sam_checkpoint:
        if not args.sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
        sam = SAM(args.sam_checkpoint)
        logger.info("SAM model loaded")

    # --- Run reconstruction ---
    process_pair(
        image_1,
        image_2,
        mast3r_engine,
        sam,
        calibration_params,
        pair_name,
        render_cameras=args.render_cameras,
        output_folder=output_folder,
        subsample=args.subsample,
        save_match_render=args.save_matches,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Reconstruction complete")
    logger.info(f"Output: {output_folder}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
