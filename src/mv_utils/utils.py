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
            'camera_matrix1', 'camera_matrix2', 
            'dist_coeffs1', 'dist_coeffs2',
            'rotation_matrix', 'translation_vector'
        ]
        
        # Check all required keys exist
        missing_keys = [key for key in required_keys if key not in calib_dict]
        if missing_keys:
            return False, f"Missing required keys: {', '.join(missing_keys)}"
        
        # Convert to numpy arrays and validate shapes
        # Validate camera_matrix1: 3x3 matrix
        try:
            cam1 = np.array(calib_dict['camera_matrix1'])
            if cam1.shape != (3, 3):
                return False, f"camera_matrix1 has wrong shape {cam1.shape}, expected (3, 3)"
        except (ValueError, TypeError) as e:
            return False, f"camera_matrix1 is not a valid 3x3 matrix: {str(e)}"
        
        # Validate camera_matrix2: 3x3 matrix
        try:
            cam2 = np.array(calib_dict['camera_matrix2'])
            if cam2.shape != (3, 3):
                return False, f"camera_matrix2 has wrong shape {cam2.shape}, expected (3, 3)"
        except (ValueError, TypeError) as e:
            return False, f"camera_matrix2 is not a valid 3x3 matrix: {str(e)}"
        
        # Validate dist_coeffs1: 1x5 vector
        try:
            dist1 = np.array(calib_dict['dist_coeffs1'])
            if dist1.shape != (1, 5):
                return False, f"dist_coeffs1 has wrong shape {dist1.shape}, expected (1, 5)"
        except (ValueError, TypeError) as e:
            return False, f"dist_coeffs1 is not a valid 1x5 vector: {str(e)}"
        
        # Validate dist_coeffs2: 1x5 vector
        try:
            dist2 = np.array(calib_dict['dist_coeffs2'])
            if dist2.shape != (1, 5):
                return False, f"dist_coeffs2 has wrong shape {dist2.shape}, expected (1, 5)"
        except (ValueError, TypeError) as e:
            return False, f"dist_coeffs2 is not a valid 1x5 vector: {str(e)}"
        
        # Validate rotation_matrix: 3x3 matrix
        try:
            rot = np.array(calib_dict['rotation_matrix'])
            if rot.shape != (3, 3):
                return False, f"rotation_matrix has wrong shape {rot.shape}, expected (3, 3)"
        except (ValueError, TypeError) as e:
            return False, f"rotation_matrix is not a valid 3x3 matrix: {str(e)}"
        
        # Validate translation_vector: 3x1 vector
        try:
            trans = np.array(calib_dict['translation_vector'])
            if trans.shape != (3, 1):
                return False, f"translation_vector has wrong shape {trans.shape}, expected (3, 1)"
        except (ValueError, TypeError) as e:
            return False, f"translation_vector is not a valid 3x1 vector: {str(e)}"
        
        return True, "Calibration file is valid"
        
    except Exception as e:
        return False, f"Unexpected error validating calibration: {str(e)}"
