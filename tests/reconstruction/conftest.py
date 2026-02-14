"""Reconstruction test configuration.

Adds src/reconstruction to sys.path so flat imports work
(e.g. from calibration import Scene, from stereo.image import ...).

Must NOT coexist with src/tracking on the same sys.path
because tracking's 'models' package collides with mast3r/dust3r/croco/models.
"""

import sys
from pathlib import Path

_RECONSTRUCTION_SRC = str(Path(__file__).resolve().parents[2] / "src" / "reconstruction")
if _RECONSTRUCTION_SRC not in sys.path:
    sys.path.insert(0, _RECONSTRUCTION_SRC)
