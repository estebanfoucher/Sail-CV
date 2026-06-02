"""Read/write YOLO-format images and label files for augmentation."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import cv2
import numpy as np  # noqa: TC002

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as uint8 RGB HWC."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Failed to write image: {path}")


def parse_yolo_label_file(label_path: Path) -> tuple[list[list[float]], list[int]]:
    """
    Parse YOLO txt: each line class xc yc w h (normalized).

    Returns:
        bboxes: list of [xc, yc, w, h] floats in [0,1]
        class_labels: list of int class ids
    """
    bboxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.is_file():
        return bboxes, classes
    for raw in label_path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        bboxes.append([xc, yc, w, h])
        classes.append(cls_id)
    return bboxes, classes


def write_yolo_label_file(
    label_path: Path,
    bboxes: list[list[float]],
    class_labels: list[int],
) -> None:
    """Write YOLO lines; bboxes and class_labels same length."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for cls_id, box in zip(class_labels, bboxes, strict=True):
        xc, yc, w, h = box
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def list_yolo_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    out: list[Path] = []
    for f in sorted(images_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(f)
    return out


def yolo_dataset_root(data_dir: Path) -> Path:
    """
    Resolve a YOLO *split* root: a directory that directly contains ``images/`` and ``labels/``.

    If ``data_dir`` does not, looks for exactly one immediate subdirectory that does
    (handles an extra wrapper folder inside ``train/`` after some zips).
    """
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {data_dir}")

    if (data_dir / "images").is_dir() and (data_dir / "labels").is_dir():
        return data_dir

    nested = [
        p
        for p in sorted(data_dir.iterdir())
        if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()
    ]
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        names = [p.name for p in nested]
        raise FileNotFoundError(
            f"{data_dir} has no images/ + labels/ at its root; found multiple nested "
            f"splits with images+labels: {names}. Pass one split explicitly, e.g. "
            f".../train or .../val."
        )

    try:
        listing = sorted(p.name for p in data_dir.iterdir())
    except OSError:
        listing = ["<unreadable>"]
    raise FileNotFoundError(
        f"Expected {data_dir}/images and {data_dir}/labels (flat YOLO export per split), "
        f"or a single subdirectory under {data_dir} that contains both. "
        f"Top-level entries: {listing[:30]}{'...' if len(listing) > 30 else ''}"
    )
