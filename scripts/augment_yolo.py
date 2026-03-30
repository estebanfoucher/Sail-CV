#!/usr/bin/env python3
"""Entry point for YOLO augmentation CLI (ensures `src` is on sys.path)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train.augmentation.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
