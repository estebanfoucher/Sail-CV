"""CLI: preview strips or full YOLO export from a zip or directory."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from .apply import augment_yolo_sample
from .export_worker import export_flat_one_image
from .pipelines import build_augmentation_pipeline, seed_augmentation_globals
from .viz import (
    draw_yolo_boxes_rgb,
    save_tiles_png,
    tile_grid,
    write_preview_index_html,
)
from .yolo_io import (
    list_yolo_image_paths,
    load_image_rgb,
    parse_yolo_label_file,
    save_image_rgb,
    write_yolo_label_file,
    yolo_dataset_root,
)


def _find_flat_yolo_root(base: Path) -> Path:
    if (base / "images").is_dir() and (base / "labels").is_dir():
        return base
    for child in sorted(base.iterdir()):
        if (
            child.is_dir()
            and (child / "images").is_dir()
            and (child / "labels").is_dir()
        ):
            return child
    raise FileNotFoundError(
        f"No YOLO layout (images/ + labels/) under {base}. "
        "Unzip so the export root contains images/ and labels/."
    )


def _prepare_data_root(*, zip_path: Path | None, data_dir: Path | None) -> Path:
    if zip_path is not None and data_dir is not None:
        raise SystemExit("Use only one of --zip or --data-dir")
    if zip_path is not None:
        tmp = Path(tempfile.mkdtemp(prefix="sailcv_aug_"))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        return _find_flat_yolo_root(tmp)
    if data_dir is not None:
        return yolo_dataset_root(data_dir.resolve())
    raise SystemExit("Provide --zip or --data-dir")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment flat YOLO dataset (preview PNG strips or full export)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path, help="Path to YOLO export zip")
    src.add_argument(
        "--data-dir", type=Path, help="Unzipped root with images/ and labels/"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output_augmented"),
        help="Output root (previews/ and/or images+labels)",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only write preview PNG strips + optional index.html (no YOLO export)",
    )
    parser.add_argument(
        "--max-images", type=int, default=0, help="Cap images (0 = all)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Augmented panels per strip / variants"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Base RNG seed (reproducible)"
    )
    parser.add_argument(
        "--draw-boxes", action="store_true", help="Overlay YOLO boxes on panels"
    )
    parser.add_argument(
        "--no-index", action="store_true", help="Skip writing previews/index.html"
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Max side (px) of a single tile after resize",
    )
    parser.add_argument(
        "--tile-cols",
        type=int,
        default=0,
        help="Columns in the tile grid. 0 = auto (ceil(sqrt(N)))",
    )
    parser.add_argument(
        "--preset", type=str, default="sail_default", help="Augmentation preset"
    )
    parser.add_argument(
        "--no-bbox-blur",
        action="store_true",
        help="Disable bbox-only motion blur after Albumentations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel processes for full export (not preview). "
            "Use 0 for all CPUs (os.cpu_count()). "
            "Preview strips require --workers 1."
        ),
    )
    args = parser.parse_args()

    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    if workers < 1:
        raise SystemExit("--workers must be >= 0 (0 = auto)")
    if args.preview_only and workers > 1:
        print(
            "Note: --preview-only uses a single worker (strip assembly is sequential)."
        )
        workers = 1

    root = _prepare_data_root(zip_path=args.zip, data_dir=args.data_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"
    image_paths = list_yolo_image_paths(images_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    out_dir = args.out_dir.resolve()
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    export_img = out_dir / "images"
    export_lbl = out_dir / "labels"
    if not args.preview_only:
        export_img.mkdir(parents=True, exist_ok=True)
        export_lbl.mkdir(parents=True, exist_ok=True)
        yaml_src = root / "data.yaml"
        if yaml_src.is_file():
            shutil.copy2(yaml_src, out_dir / "data.yaml")
        cls_src = root / "classes.txt"
        if cls_src.is_file():
            shutil.copy2(cls_src, out_dir / "classes.txt")

    rel_pngs: list[str] = []

    if not args.preview_only and workers > 1:
        tasks = [
            (
                str(img_path),
                str(labels_dir),
                str(export_img),
                str(export_lbl),
                idx,
                args.repeats,
                args.seed,
                args.no_bbox_blur,
                args.preset,
            )
            for idx, img_path in enumerate(image_paths)
        ]
        print(f"Exporting with {workers} worker process(es) (preview strips skipped).")
        with Pool(processes=workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(export_flat_one_image, tasks, chunksize=2),
                total=len(tasks),
                desc="Export",
            ):
                pass
    else:
        _compose = None
        if not args.preview_only:
            _compose = build_augmentation_pipeline(seed=None, preset=args.preset)

        for idx, img_path in enumerate(image_paths):
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"
            image_rgb = load_image_rgb(img_path)
            bboxes, class_labels = parse_yolo_label_file(label_path)

            panels: list = []
            orig = image_rgb
            if args.draw_boxes and bboxes:
                orig = draw_yolo_boxes_rgb(orig, bboxes, class_labels)
            panels.append(orig)

            if not args.preview_only:
                save_image_rgb(export_img / f"{stem}.jpg", image_rgb)
                write_yolo_label_file(export_lbl / f"{stem}.txt", bboxes, class_labels)

            for r in range(args.repeats):
                sub_seed = None
                if args.seed is not None:
                    sub_seed = args.seed + idx * 10_007 + r * 1_003
                if not args.preview_only and _compose is not None:
                    if sub_seed is not None:
                        _compose.set_random_seed(sub_seed)
                        seed_augmentation_globals(sub_seed)
                    compose = _compose
                else:
                    compose = build_augmentation_pipeline(
                        seed=sub_seed, preset=args.preset
                    )
                rng = (
                    random.Random(sub_seed) if sub_seed is not None else random.Random()
                )
                aug, bb, cl = augment_yolo_sample(
                    image_rgb,
                    bboxes,
                    class_labels,
                    compose,
                    rng=rng,
                    bbox_motion_blur=not args.no_bbox_blur,
                )
                if args.draw_boxes and bb:
                    aug = draw_yolo_boxes_rgb(aug, bb, cl)
                panels.append(aug)

                if not args.preview_only:
                    out_stem = f"{stem}_aug{r}"
                    save_image_rgb(export_img / f"{out_stem}.jpg", aug)
                    write_yolo_label_file(export_lbl / f"{out_stem}.txt", bb, cl)

            tile_cols = args.tile_cols if args.tile_cols > 0 else None
            tiles = tile_grid(panels, cols=tile_cols, max_tile_side=args.max_side)
            tiles_path = preview_dir / f"{stem}_tiles.png"
            save_tiles_png(tiles_path, tiles)
            rel_pngs.append(tiles_path.name)

    if not args.no_index:
        write_preview_index_html(preview_dir, sorted(rel_pngs))

    print(f"Processed {len(image_paths)} image(s).")
    print(f"Previews: {preview_dir}")
    if not args.preview_only:
        print(f"Export: {export_img} , {export_lbl}")


if __name__ == "__main__":
    main()
