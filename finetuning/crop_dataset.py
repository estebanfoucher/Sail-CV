"""
Build Ultralytics classification dataset (train|val/<class_id>/*.jpg) from YOLO splits.

Each split directory must contain images/ and labels/ (YOLO bbox format).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _iter_images(img_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pat in IMAGE_GLOBS:
        out.extend(img_dir.glob(pat))
    return out


def _max_class_in_labels(lbl_dir: Path) -> int:
    """Highest class index seen in any label file under lbl_dir (-1 if none)."""
    best = -1
    if not lbl_dir.exists():
        return best
    for lf in lbl_dir.glob("*.txt"):
        for raw in lf.read_text().splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 5:
                best = max(best, int(float(parts[0])))
    return best


def _nc_from_data_yaml(yaml_path: Path) -> int | None:
    if not yaml_path.exists():
        return None
    text = yaml_path.read_text()
    m = re.search(r"^\s*nc:\s*(\d+)\s*$", text, re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


def prepare_balanced_dataset(
    base_path: str | Path,
    output_path: str | Path,
    *,
    padding: float = 0.15,
    multipliers: dict[int, int] | None = None,
    min_width: int = 10,
    min_height: int = 10,
) -> None:
    """
    Extract class crops from one YOLO split (images/ + labels/) into output_path/<cls_id>/.

    Args:
        base_path: Directory containing images/ and labels/.
        output_path: Root for class subfolders (e.g. cls_dataset/train).
        padding: Fractional pad around each box.
        multipliers: Per-class repeat count (0 = skip class). Missing classes default to 1.
        min_width, min_height: Skip crops smaller than this (avoids Ultralytics corrupt warnings).
    """
    base_dir = Path(base_path)
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")

    max_lbl = _max_class_in_labels(lbl_dir)
    nc_hint = _nc_from_data_yaml(base_dir.parent / "data.yaml")
    upper = max(max_lbl, (nc_hint or 1) - 1, 0)

    if multipliers is None:
        multipliers = dict.fromkeys(range(upper + 1), 1)
    else:
        multipliers = {int(k): int(v) for k, v in multipliers.items()}

    for cls_id in range(upper + 1):
        if multipliers.get(cls_id, 1) > 0:
            (out_dir / str(cls_id)).mkdir(parents=True, exist_ok=True)

    images = _iter_images(img_dir)
    for img_file in tqdm(images, desc=f"Crops ({base_dir.name})"):
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w, _ = img.shape

        with open(lbl_file) as f:
            lines = f.readlines()
        for obj_i, line in enumerate(lines):
            parts = line.split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            count = multipliers.get(cls_id, 1)
            if count <= 0:
                continue

            x_c, y_c, bw, bh = map(float, parts[1:6])
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)
            x1_p, y1_p = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2_p, y2_p = min(w, x2 + pad_w), min(h, y2 + pad_h)

            crop = img[y1_p:y2_p, x1_p:x2_p]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            if cw < min_width or ch < min_height:
                continue

            for r in range(count):
                output_crop = crop.copy()
                if r > 0 and r % 2 == 0:
                    output_crop = cv2.flip(output_crop, 1)
                save_name = f"{img_file.stem}_obj{obj_i}_copy{r}.jpg"
                cv2.imwrite(str(out_dir / str(cls_id) / save_name), output_crop)


def _parse_multipliers_json(s: str) -> dict[int, int]:
    raw = json.loads(s)
    return {int(k): int(v) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO split (images+labels) -> Ultralytics cls folders train|val/<id>/*.jpg"
    )
    parser.add_argument(
        "--train",
        type=Path,
        help="Train split directory (contains images/ and labels/).",
    )
    parser.add_argument(
        "--val",
        type=Path,
        help="Val split directory (contains images/ and labels/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output root; writes train/ and val/ class subfolders here.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding around boxes as fraction of box size (default 0.15).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(10, 10),
        help="Minimum crop width and height in pixels (default 10 10).",
    )
    parser.add_argument(
        "--multipliers",
        type=str,
        default=None,
        help='JSON object mapping class id to repeat count, e.g. \'{"0":1,"1":5,"2":12}\'.',
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Use multiplier 1 for every class. For default fused paths, overrides preset 1/5/12.",
    )
    args = parser.parse_args()
    if args.no_balance and args.multipliers:
        parser.error("Do not combine --no-balance with --multipliers")

    project_root = Path(__file__).resolve().parent.parent
    triple = args.train is not None and args.val is not None and args.output is not None
    if (
        args.train is not None or args.val is not None or args.output is not None
    ) and not triple:
        parser.error("Provide all of --train, --val, and --output together")

    if triple:
        train_in = args.train.resolve()
        val_in = args.val.resolve()
        out_root = args.output.resolve()
        mult = _parse_multipliers_json(args.multipliers) if args.multipliers else None

        mw, mh = args.min_size
        (out_root / "train").mkdir(parents=True, exist_ok=True)
        (out_root / "val").mkdir(parents=True, exist_ok=True)
        prepare_balanced_dataset(
            train_in,
            out_root / "train",
            padding=args.padding,
            multipliers=mult,
            min_width=mw,
            min_height=mh,
        )
        prepare_balanced_dataset(
            val_in,
            out_root / "val",
            padding=args.padding,
            multipliers=mult,
            min_width=mw,
            min_height=mh,
        )
        print(f"Done. Classification dataset at {out_root}")
        return

    base_data_dir = project_root / "data" / "splitted_datasets" / "fused"
    output_base_dir = project_root / "data" / "cls_dataset"
    train_input = base_data_dir / "train"
    val_input = base_data_dir / "val"
    train_output = output_base_dir / "train"
    val_output = output_base_dir / "val"

    if args.multipliers:
        default_mult = _parse_multipliers_json(args.multipliers)
    else:
        default_mult = None if args.no_balance else {0: 1, 1: 5, 2: 12}
    mw, mh = args.min_size

    if train_input.exists() and (train_input / "images").exists():
        print(f"Processing train from: {train_input}")
        prepare_balanced_dataset(
            train_input,
            train_output,
            padding=args.padding,
            multipliers=default_mult,
            min_width=mw,
            min_height=mh,
        )
    else:
        print(f"Train dataset not found at {train_input}")
        print(
            "Pass --train, --val, and --output, or place fused data under data/splitted_datasets/fused/."
        )

    if val_input.exists() and (val_input / "images").exists():
        print(f"Processing val from: {val_input}")
        prepare_balanced_dataset(
            val_input,
            val_output,
            padding=args.padding,
            multipliers=default_mult,
            min_width=mw,
            min_height=mh,
        )
    else:
        print(f"Val dataset not found at {val_input}")


if __name__ == "__main__":
    main()
