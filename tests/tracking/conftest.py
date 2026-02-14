"""Tracking test configuration.

Adds src/tracking to sys.path so flat imports work
(e.g. from models import Detection, from detector import Detector).

Must NOT coexist with src/reconstruction on the same sys.path
to avoid 'models' namespace collision with mast3r/dust3r/croco/models.
"""

import sys
from pathlib import Path

_TRACKING_SRC = str(Path(__file__).resolve().parents[2] / "src" / "tracking")
if _TRACKING_SRC not in sys.path:
    sys.path.insert(0, _TRACKING_SRC)
