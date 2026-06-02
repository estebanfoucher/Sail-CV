"""Preview tiles/strips and optional YOLO box overlays."""

from __future__ import annotations

import math
from pathlib import Path  # noqa: TC003

import cv2
import numpy as np


def draw_yolo_boxes_rgb(
    image_rgb: np.ndarray,
    bboxes: list[list[float]],
    class_labels: list[int] | None = None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw YOLO boxes on a copy (RGB)."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    if class_labels is None:
        class_labels = [0] * len(bboxes)
    for box, cls in zip(bboxes, class_labels, strict=True):
        xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            str(cls),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def resize_max_side(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image_rgb
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def horizontal_strip(
    images: list[np.ndarray],
    *,
    gap: int = 6,
    max_side: int = 1280,
) -> np.ndarray:
    """Concatenate RGB panels left-to-right; equalize height."""
    if not images:
        raise ValueError("images must be non-empty")
    resized = [resize_max_side(im, max_side) for im in images]
    mh = min(im.shape[0] for im in resized)
    normed: list[np.ndarray] = []
    for im in resized:
        if im.shape[0] != mh:
            nw = max(1, int(im.shape[1] * mh / im.shape[0]))
            normed.append(cv2.resize(im, (nw, mh), interpolation=cv2.INTER_AREA))
        else:
            normed.append(im)
    sep = np.full((mh, gap, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, im in enumerate(normed):
        parts.append(im)
        if i < len(normed) - 1:
            parts.append(sep)
    return np.hstack(parts)


def save_strip_png(path: Path, strip_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Failed to write {path}")


def tile_grid(
    images: list[np.ndarray],
    *,
    cols: int | None = None,
    gap: int = 6,
    max_tile_side: int = 640,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Lay RGB panels in a grid (cols x rows). Tiles are letterboxed to a uniform size.

    Args:
        images: list of HxWx3 RGB panels.
        cols: number of columns. Default: ``ceil(sqrt(N))``.
        gap: spacing in pixels between tiles.
        max_tile_side: longest side of a single tile after resize.
        bg: background color filling gaps and letterboxing (RGB).
    """
    if not images:
        raise ValueError("images must be non-empty")

    n = len(images)
    if cols is None or cols <= 0:
        cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))

    resized = [resize_max_side(im, max_tile_side) for im in images]
    tile_w = max(im.shape[1] for im in resized)
    tile_h = max(im.shape[0] for im in resized)

    bg_arr = np.array(bg, dtype=np.uint8)
    grid_h = tile_h * rows + gap * (rows - 1)
    grid_w = tile_w * cols + gap * (cols - 1)
    canvas = np.full((grid_h, grid_w, 3), bg_arr, dtype=np.uint8)

    for idx, im in enumerate(resized):
        r, c = divmod(idx, cols)
        ih, iw = im.shape[:2]
        ox = c * (tile_w + gap) + (tile_w - iw) // 2
        oy = r * (tile_h + gap) + (tile_h - ih) // 2
        canvas[oy : oy + ih, ox : ox + iw] = im
    return canvas


def save_tiles_png(path: Path, tiles_rgb: np.ndarray) -> None:
    """Alias of :func:`save_strip_png` kept for naming clarity."""
    save_strip_png(path, tiles_rgb)


def write_preview_index_html(preview_dir: Path, rel_pngs: list[str]) -> None:
    """Write static gallery; rel_pngs paths relative to preview_dir."""
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Augmentation previews</title>",
        "<style>body{font-family:sans-serif} img{max-width:100%;border:1px solid #ccc;margin:8px 0}</style>",
        "</head><body><h1>Augmentation previews</h1>",
    ]
    for rel in rel_pngs:
        lines.append(f'<div><img src="{rel}" alt="preview" loading="lazy"/></div>')
    lines.append("</body></html>")
    (preview_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")
