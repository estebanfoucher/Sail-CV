import json
from pathlib import Path

import numpy as np
import yaml


def load_parameters(specs_file: str) -> dict:
    """Load checkerboard specifications from JSON or YAML file."""
    specs_path = Path(specs_file)
    if specs_path.suffix.lower() in [".json"]:
        with open(specs_path) as f:
            return json.load(f)
    elif specs_path.suffix.lower() in [".yml", ".yaml"]:
        with open(specs_path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {specs_path.suffix}")


def validate_calibration(calib_dict: dict) -> tuple[bool, str]:
    """
    Validate calibration dictionary has required keys with correct dimensions.

    Args:
        calib_dict: Dictionary containing calibration parameters

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Required keys
        required_keys = [
            "camera_matrix1",
            "camera_matrix2",
            "dist_coeffs1",
            "dist_coeffs2",
            "rotation_matrix",
            "translation_vector",
        ]

        # Check all required keys exist
        missing_keys = [key for key in required_keys if key not in calib_dict]
        if missing_keys:
            return False, f"Missing required keys: {', '.join(missing_keys)}"

        # Define validation specs: (key, expected_shape, description)
        validation_specs = [
            ("camera_matrix1", (3, 3), "3x3 matrix"),
            ("camera_matrix2", (3, 3), "3x3 matrix"),
            ("dist_coeffs1", (1, 5), "1x5 vector"),
            ("dist_coeffs2", (1, 5), "1x5 vector"),
            ("rotation_matrix", (3, 3), "3x3 matrix"),
            ("translation_vector", (3, 1), "3x1 vector"),
        ]

        # Validate each parameter
        for key, expected_shape, description in validation_specs:
            error_msg = _validate_array_parameter(
                calib_dict, key, expected_shape, description
            )
            if error_msg:
                return False, error_msg

        return True, "Calibration file is valid"

    except Exception as e:
        return False, f"Unexpected error validating calibration: {e!s}"


def _validate_array_parameter(
    calib_dict: dict, key: str, expected_shape: tuple, description: str
) -> str | None:
    """
    Validate a single array parameter from calibration dictionary.

    Args:
        calib_dict: Calibration dictionary
        key: Key to validate
        expected_shape: Expected shape tuple
        description: Human-readable description

    Returns:
        Error message if validation fails, None if successful
    """
    try:
        arr = np.array(calib_dict[key])
        if arr.shape != expected_shape:
            return f"{key} has wrong shape {arr.shape}, expected {expected_shape}"
        return None
    except (ValueError, TypeError) as e:
        return f"{key} is not a valid {description}: {e!s}"
