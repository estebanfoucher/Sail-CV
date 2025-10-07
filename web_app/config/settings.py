"""
Configuration settings for MVS web application
"""

import os
from pathlib import Path
from typing import Dict, Any
import logging

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
WEB_APP_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"

# Logging configuration
LOGGING_CONFIG = {
    'level': logging.DEBUG,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': [
        'console',
        'file'
    ],
    'log_file': 'web_app.log'
}

# Application settings
APP_CONFIG = {
    'title': 'MVS Interactive Web Application',
    'theme': 'soft',
    'server_port': 7861,
    'server_name': '0.0.0.0',
    'share': False,
    'debug': True
}

# Video processing settings
VIDEO_CONFIG = {
    'supported_formats': ['.mp4'],
    'max_file_size_mb': 500,
    'default_height': 375,
    'upload_height': 160,
    'frame_slider_default_max': 100,
    'sync_offset_range': (-50, 50),
    'sync_offset_default': 0
}

# UI Layout settings
UI_CONFIG = {
    'container_max_width': '1200px',
    'upload_section_margin': '20px 0',
    'status_box_lines': 8,
    'status_box_max_lines': 12,
    'column_scales': {
        'upload': 1,
        'status': 2
    }
}

# File handling settings
FILE_CONFIG = {
    'temp_dir': 'temp_uploads',
    'allowed_video_extensions': ['.mp4'],
    'allowed_calibration_extensions': ['.json'],
    'cleanup_on_exit': True
}

# Validation settings
VALIDATION_CONFIG = {
    'calibration_required_keys': [
        'camera_matrix1', 'camera_matrix2',
        'dist_coeffs1', 'dist_coeffs2',
        'rotation_matrix', 'translation_vector'
    ],
    'calibration_matrix_shapes': {
        'camera_matrix1': (3, 3),
        'camera_matrix2': (3, 3),
        'dist_coeffs1': (1, 5),
        'dist_coeffs2': (1, 5),
        'rotation_matrix': (3, 3),
        'translation_vector': (3, 1)
    }
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        'base_dir': BASE_DIR,
        'web_app_dir': WEB_APP_DIR,
        'src_dir': SRC_DIR,
        'logging': LOGGING_CONFIG,
        'app': APP_CONFIG,
        'video': VIDEO_CONFIG,
        'ui': UI_CONFIG,
        'file': FILE_CONFIG,
        'validation': VALIDATION_CONFIG
    }

def setup_logging():
    """Setup logging configuration"""
    config = LOGGING_CONFIG
    
    # Create handlers
    handlers = []
    
    if 'console' in config['handlers']:
        handlers.append(logging.StreamHandler())
    
    if 'file' in config['handlers']:
        log_file = WEB_APP_DIR / config['log_file']
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=config['level'],
        format=config['format'],
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
