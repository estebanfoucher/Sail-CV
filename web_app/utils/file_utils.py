"""
File utilities for MVS web application
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional


def save_uploaded_files(video_1_path: str, video_2_path: str, calibration_path: str) -> str:
    """
    Save uploaded files to a temporary directory for processing

    Args:
        video_1_path: Path to uploaded video 1
        video_2_path: Path to uploaded video 2
        calibration_path: Path to uploaded calibration file

    Returns:
        str: Path to the temporary directory containing the files
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="mvs_upload_")

    try:
        # Copy files to temporary directory
        video_1_dest = os.path.join(temp_dir, "video_1.mp4")
        video_2_dest = os.path.join(temp_dir, "video_2.mp4")
        calibration_dest = os.path.join(temp_dir, "calibration.json")

        shutil.copy2(video_1_path, video_1_dest)
        shutil.copy2(video_2_path, video_2_dest)
        shutil.copy2(calibration_path, calibration_dest)

        return temp_dir

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


def cleanup_temp_directory(temp_dir: str) -> bool:
    """
    Clean up temporary directory

    Args:
        temp_dir: Path to temporary directory

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return True
    except Exception:
        return False


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB

    Args:
        file_path: Path to file

    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0.0


def create_output_directory(base_path: str = "output") -> str:
    """
    Create output directory for processing results

    Args:
        base_path: Base path for output directory

    Returns:
        str: Path to created output directory
    """
    output_dir = Path(base_path) / "web_app_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def get_safe_filename(filename: str) -> str:
    """
    Get a safe filename by removing/replacing invalid characters

    Args:
        filename: Original filename

    Returns:
        str: Safe filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')

    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(' .')

    return safe_name


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't

    Args:
        dir_path: Path to directory

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def list_files_in_directory(dir_path: str, extensions: Optional[list] = None) -> list:
    """
    List files in directory with optional extension filtering

    Args:
        dir_path: Path to directory
        extensions: List of file extensions to filter (e.g., ['.mp4', '.json'])

    Returns:
        list: List of file paths
    """
    try:
        if not os.path.exists(dir_path):
            return []

        files = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                if extensions is None or any(item.lower().endswith(ext) for ext in extensions):
                    files.append(item_path)

        return sorted(files)

    except Exception:
        return []
