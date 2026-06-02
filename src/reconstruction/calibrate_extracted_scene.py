"""Calibrate a scene from pre-extracted intrinsic/extrinsic still images."""

from pathlib import Path

from loguru import logger

from calibration.extracted_scene_calibration import calibrate_extracted_scene
from calibration.extrinsics_calibration import get_summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    scene_dir = project_root / "assets" / "reconstruction" / "scene_innosail_0"

    results = calibrate_extracted_scene(scene_dir)
    summary = get_summary(results)
    logger.info(summary)

    summary_path = scene_dir / "extrinsics_summary.txt"
    summary_path.write_text(summary)
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
