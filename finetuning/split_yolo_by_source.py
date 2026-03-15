"""
Split a non-split YOLO dataset by source (video/photo) and reduce to one class.

Reads annotator export (images/ + labels/) and:
- Groups images by source (video basename from frame prefix, or single-image group).
- Splits by group (e.g. 80% train, 20% val) so the same video does not span both splits.
- Writes train/ and val/ with labels reduced to class_id 0 (telltale), and data.yaml (nc: 1).

Usage:
    python scripts/split_yolo_by_source.py /path/to/annotator_export --output /path/to/split_dataset
"""

import argparse
import random
import re
import shutil
from pathlib import Path

# Project root for importing reduce_to_one_class
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from reduce_to_one_class import SINGLE_CLASS_ID, SINGLE_CLASS_NAME

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Frame name pattern: {source}_{HH}_{MM}_{SS}_frame_{N}
VIDEO_FRAME_PATTERN = re.compile(r"^(.+)_(\d{2})_(\d{2})_(\d{2})_frame_\d+$")


def get_source_from_stem(stem: str) -> str:
    """
    Derive source id from image stem for grouping.
    Video frames: myboat_00_01_30_frame_0 -> myboat.
    Photos: photo_0 -> photo_0 (each its own group).
    """
    m = VIDEO_FRAME_PATTERN.match(stem)
    if m:
        return m.group(1)
    return stem


def group_images_by_source(images_dir: Path) -> dict[str, list[Path]]:
    """Group image paths by source (video basename or photo stem)."""
    groups: dict[str, list[Path]] = {}
    for f in images_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source = get_source_from_stem(f.stem)
        groups.setdefault(source, []).append(f)
    return groups


def split_groups(
    groups: dict[str, list[Path]],
    train_ratio: float,
    seed: int | None,
) -> tuple[list[Path], list[Path]]:
    """Split grouped images into train and val by group. Returns (train_paths, val_paths)."""
    source_keys = sorted(groups.keys())
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(source_keys)
    else:
        random.shuffle(source_keys)
    n_train = max(1, int(len(source_keys) * train_ratio))
    train_sources = set(source_keys[:n_train])
    val_sources = set(source_keys[n_train:])
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for source, paths in groups.items():
        if source in train_sources:
            train_paths.extend(paths)
        else:
            val_paths.extend(paths)
    return train_paths, val_paths


def write_split_and_reduce(
    source_images_dir: Path,
    source_labels_dir: Path,
    output_dir: Path,
    train_paths: list[Path],
    val_paths: list[Path],
    class_id: int = SINGLE_CLASS_ID,
) -> None:
    """
    Write train/ and val/ with images and labels; labels are written with class_id
    replaced by the single class (0). Creates data.yaml and classes.txt at output_dir.
    """
    out_train_img = output_dir / "train" / "images"
    out_train_lbl = output_dir / "train" / "labels"
    out_val_img = output_dir / "val" / "images"
    out_val_lbl = output_dir / "val" / "labels"
    for d in (out_train_img, out_train_lbl, out_val_img, out_val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    def write_split(paths: list[Path], img_out: Path, lbl_out: Path) -> None:
        for img_path in paths:
            shutil.copy2(img_path, img_out / img_path.name)
            label_name = img_path.stem + ".txt"
            src_label = source_labels_dir / label_name
            dest_label = lbl_out / label_name
            if src_label.exists():
                with open(src_label) as f:
                    lines = f.readlines()
                modified = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        parts[0] = str(class_id)
                        modified.append(" ".join(parts))
                with open(dest_label, "w") as f:
                    f.write("\n".join(modified) + "\n")
            else:
                dest_label.touch()

    write_split(train_paths, out_train_img, out_train_lbl)
    write_split(val_paths, out_val_img, out_val_lbl)

    # data.yaml for YOLO (path is dataset root; train/val relative to it)
    data_yaml = f"""path: {output_dir.resolve()}
train: train/images
val: val/images
nc: 1
names: ['{SINGLE_CLASS_NAME}']
"""
    (output_dir / "data.yaml").write_text(data_yaml)
    (output_dir / "classes.txt").write_text(SINGLE_CLASS_NAME + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset by source (video/photo) and reduce to one class (telltale)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to annotator export (must contain images/ and labels/).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for train/val and data.yaml.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of groups to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible split.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output.resolve()

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    print("Grouping images by source...")
    groups = group_images_by_source(images_dir)
    if not groups:
        raise SystemExit("No images found in images/.")
    total = sum(len(p) for p in groups.values())
    print(f"  {len(groups)} source(s), {total} image(s)")

    print("Splitting by group (train/val)...")
    train_paths, val_paths = split_groups(groups, args.train_ratio, args.seed)
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Writing train/val and reducing labels to one class...")
    write_split_and_reduce(
        images_dir,
        labels_dir,
        output_dir,
        train_paths,
        val_paths,
    )
    print(f"Done. Output: {output_dir}")
    print("  train/images, train/labels")
    print("  val/images, val/labels")
    print("  data.yaml (nc=1, names=['telltale']), classes.txt")


if __name__ == "__main__":
    main()
