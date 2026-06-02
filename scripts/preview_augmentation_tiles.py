"""Build a tiled preview per original image showing the 12 augmentations.

For every original image in ``<dataset>/<split>/images/<stem>.jpg`` (filenames
without the ``_aug*`` suffix), we assemble a 4×4 grid containing:

- the original (top-left, labelled ``ORIGINAL``),
- the 12 augmented variants ``<stem>_aug0.jpg`` … ``<stem>_aug11.jpg`` (labelled ``aug00`` … ``aug11``).

YOLO bounding boxes from the corresponding ``<split>/labels/<stem>.txt`` are
overlayed on each tile, so you can visually check that augmentations kept the
labels consistent.

Output: ``<dataset>/previews/<split>/<stem>.jpg``.

Example::

    uv run python scripts/preview_augmentation_tiles.py \
        --dataset custom_augmented_ds_from_colab_17_epoch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUM_AUGS = 12
GRID_COLS = 4
GRID_ROWS = 4
THUMB_WIDTH = 480
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2
LABEL_COLOR = (255, 255, 255)
LABEL_BG = (0, 0, 0)
LABEL_PAD = 6


def _load_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def _draw_boxes(img: np.ndarray, boxes: list[tuple[int, float, float, float, float]]) -> None:
    h, w = img.shape[:2]
    for _cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)


def _make_tile(
    img_path: Path,
    label_path: Path,
    tile_w: int,
    tile_h: int,
    caption: str,
    missing: bool = False,
) -> np.ndarray:
    if missing or not img_path.is_file():
        tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
        cv2.putText(
            tile,
            "MISSING",
            (20, tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return _add_caption(tile, caption)

    img = cv2.imread(str(img_path))
    if img is None:
        return _make_tile(img_path, label_path, tile_w, tile_h, caption, missing=True)

    boxes = _load_yolo_boxes(label_path)
    _draw_boxes(img, boxes)

    h, w = img.shape[:2]
    scale = min(tile_w / w, tile_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    off_x = (tile_w - new_w) // 2
    off_y = (tile_h - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return _add_caption(canvas, caption)


def _add_caption(tile: np.ndarray, text: str) -> np.ndarray:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x1, y1 = LABEL_PAD, LABEL_PAD
    x2, y2 = x1 + tw + 2 * LABEL_PAD, y1 + th + 2 * LABEL_PAD
    cv2.rectangle(tile, (x1, y1), (x2, y2), LABEL_BG, -1)
    cv2.putText(
        tile,
        text,
        (x1 + LABEL_PAD, y2 - LABEL_PAD),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        LABEL_COLOR,
        2,
        cv2.LINE_AA,
    )
    return tile


def _build_grid(tiles: list[np.ndarray], cols: int, rows: int) -> np.ndarray:
    assert len(tiles) == cols * rows
    th, tw = tiles[0].shape[:2]
    grid = np.zeros((th * rows, tw * cols, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
    return grid


def _process_split(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    thumb_width: int,
) -> int:
    if not images_dir.is_dir():
        return 0

    originals = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "_aug" not in p.stem
    )
    if not originals:
        return 0

    probe = cv2.imread(str(originals[0]))
    if probe is None:
        raise RuntimeError(f"Cannot read {originals[0]}")
    h, w = probe.shape[:2]
    tile_w = thumb_width
    tile_h = max(1, int(round(h * (thumb_width / w))))

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for orig in originals:
        stem = orig.stem
        tiles: list[np.ndarray] = []

        tiles.append(
            _make_tile(
                orig,
                labels_dir / f"{stem}.txt",
                tile_w,
                tile_h,
                "ORIGINAL",
            )
        )
        for i in range(NUM_AUGS):
            aug_img = images_dir / f"{stem}_aug{i}.jpg"
            aug_label = labels_dir / f"{stem}_aug{i}.txt"
            tiles.append(
                _make_tile(
                    aug_img,
                    aug_label,
                    tile_w,
                    tile_h,
                    f"aug{i:02d}",
                )
            )

        total_slots = GRID_COLS * GRID_ROWS
        while len(tiles) < total_slots:
            blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            tiles.append(blank)

        grid = _build_grid(tiles, GRID_COLS, GRID_ROWS)
        out_path = output_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{out_path} ({tile_w * GRID_COLS}x{tile_h * GRID_ROWS})")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "custom_augmented_ds_from_colab_17_epoch",
        help="Root of the YOLO dataset (contains train/ and val/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir. Default: <dataset>/previews/",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=THUMB_WIDTH,
        help=f"Thumbnail width per tile in px (default {THUMB_WIDTH})",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_root = (args.output or (dataset / "previews")).resolve()

    total = 0
    for split in args.splits:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        out_dir = output_root / split
        added = _process_split(images_dir, labels_dir, out_dir, args.thumb_width)
        print(f"[{split}] {added} preview(s) -> {out_dir}")
        total += added

    print(f"Done. {total} preview(s) under {output_root}")


if __name__ == "__main__":
    main()
