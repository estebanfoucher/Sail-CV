"""
Custom exceptions for MVS web application
"""


class MVSException(Exception):
    """Base exception for MVS application"""
    pass


class ValidationError(MVSException):
    """Raised when validation fails"""
    pass


class VideoProcessingError(MVSException):
    """Raised when video processing fails"""
    pass


class FileHandlingError(MVSException):
    """Raised when file operations fail"""
    pass


class ConfigurationError(MVSException):
    """Raised when configuration is invalid"""
    pass


class CalibrationError(MVSException):
    """Raised when calibration data is invalid"""
    pass


class VideoCompatibilityError(MVSException):
    """Raised when videos are not compatible"""
    pass
