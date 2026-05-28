"""Minimal stdlib-logging stand-in for loguru's `logger`.

Avoids an external loguru dependency (not installable on the Jetson's
pip-less system Python). Supports the debug/info/warning/error calls used
across mv_utils.
"""

import logging

logger = logging.getLogger("mv_utils")
