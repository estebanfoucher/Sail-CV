"""
Constants for MVS web application
"""

# Video processing constants
VIDEO_FORMATS = ['.mp4']
CALIBRATION_FORMATS = ['.json']

# UI Constants
DEFAULT_VIDEO_HEIGHT = 375
UPLOAD_WIDGET_HEIGHT = 160
DEFAULT_FRAME_SLIDER_MAX = 100
SYNC_OFFSET_RANGE = (-50, 50)
DEFAULT_SYNC_OFFSET = 0

# Status messages
STATUS_MESSAGES = {
    'ready': 'Ready to upload files...',
    'no_videos': 'No videos loaded',
    'videos_ready': 'Videos ready',
    'files_available': 'Files available',
    'some_files_removed': 'Some files removed',
    'all_files_removed': 'All files removed',
    'upload_at_least_one': 'Please upload at least one video',
    'video_loaded': 'Video loaded successfully',
    'both_videos_loaded': 'Both videos loaded successfully',
    'video_already_loaded': 'Video already loaded',
    'both_videos_already_loaded': 'Both videos already loaded'
}

# Error messages
ERROR_MESSAGES = {
    'file_not_found': 'File does not exist',
    'invalid_format': 'File must be MP4 format',
    'validation_failed': 'Validation failed',
    'loading_failed': 'Loading failed',
    'unexpected_error': 'Unexpected error',
    'missing_keys': 'Missing required keys',
    'wrong_shape': 'Wrong shape',
    'invalid_matrix': 'Invalid matrix'
}

# Success messages
SUCCESS_MESSAGES = {
    'video_valid': 'Valid MP4 file',
    'calibration_valid': 'Valid JSON file',
    'compatibility_ok': 'Videos are compatible',
    'files_saved': 'Files saved successfully'
}

# UI Labels
UI_LABELS = {
    'video_1_title': '📹 Video 1 (Primary Camera)',
    'video_2_title': '📹 Video 2 (Secondary Camera)',
    'upload_video_1': 'Upload Video 1',
    'upload_video_2': 'Upload Video 2',
    'video_1_player': 'Video 1 Player',
    'video_2_player': 'Video 2 Player',
    'frame_control': '🎬 Frame Control',
    'frame_selector': 'Frame Selector',
    'frame_number': 'Current Frame',
    'sync_settings': '⚙️ Sync Settings',
    'sync_offset': 'Sync Offset (frames)',
    'video_status': 'Video Status',
    'upload_status': 'Upload Status',
    'calibration_file': 'Upload Calibration File',
    'refresh_display': '🔄 Refresh Display'
}
