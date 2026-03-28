"""
Reduce YOLO datasets from multiple classes to a single class "telltale".

Converts all class IDs in label files to 0 and updates classes.txt to only contain "telltale".
Use as a library (constants + helpers) or run as CLI: ``uv run python finetuning/reduce_to_one_class.py ...``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SINGLE_CLASS_NAME = "telltale"
SINGLE_CLASS_ID = 0


def reduce_dataset_to_one_class(
    source_dir: Path,
    output_dir: Path,
    class_name: str = SINGLE_CLASS_NAME,
    class_id: int = SINGLE_CLASS_ID,
) -> dict:
    """
    Reduce a dataset to a single class by converting all labels to class_id.

    Expects ``source_dir/images`` and ``source_dir/labels`` (flat YOLO layout).
    """
    print("=" * 60)
    print(f"Reducing Dataset to Single Class: '{class_name}'")
    print("=" * 60)
    print()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_images_dir = source_dir / "images"
    source_labels_dir = source_dir / "labels"

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {source_images_dir}")
    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {source_labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        f
        for f in source_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images to process")
    print()

    summary: dict = {
        "total_images": len(image_files),
        "labels_processed": 0,
        "labels_modified": 0,
        "empty_labels": 0,
        "class_name": class_name,
        "class_id": class_id,
    }

    for img_path in image_files:
        dest_img = output_images_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        label_name = img_path.stem + ".txt"
        src_label = source_labels_dir / label_name
        dest_label = output_labels_dir / label_name

        if src_label.exists():
            with open(src_label) as f:
                lines = f.readlines()

            modified_lines: list[str] = []
            for raw in lines:
                stripped = raw.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                if len(parts) >= 5:
                    parts[0] = str(class_id)
                    modified_lines.append(" ".join(parts))

            if modified_lines:
                with open(dest_label, "w") as f:
                    f.write("\n".join(modified_lines) + "\n")
                summary["labels_processed"] += 1
                summary["labels_modified"] += 1
            else:
                dest_label.touch()
                summary["labels_processed"] += 1
                summary["empty_labels"] += 1
        else:
            dest_label.touch()
            summary["labels_processed"] += 1
            summary["empty_labels"] += 1

    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w") as f:
        f.write(f"{class_name}\n")
    print(f"Created classes.txt with single class: '{class_name}'")

    for file_name in [
        "data.yaml",
        "split_summary.json",
        "fusion_summary.json",
        "renaming_summary.json",
    ]:
        src_file = source_dir / file_name
        if not src_file.exists():
            continue
        if file_name == "data.yaml":
            with open(src_file) as f:
                content = f.read()

            lines = content.split("\n")
            updated_lines: list[str] = []
            for line in lines:
                if line.startswith("nc:"):
                    updated_lines.append("nc: 1")
                elif line.startswith("names:"):
                    updated_lines.append(f"names: ['{class_name}']")
                else:
                    updated_lines.append(line)

            with open(output_dir / file_name, "w") as f:
                f.write("\n".join(updated_lines))
            print(f"Updated {file_name} (set nc=1)")
        else:
            shutil.copy2(src_file, output_dir / file_name)
            print(f"Copied {file_name}")

    print()
    print("=" * 60)
    print("Reduction Summary")
    print("=" * 60)
    print()
    print(f"Total images:        {summary['total_images']}")
    print(f"Labels processed:   {summary['labels_processed']}")
    print(f"Labels modified:     {summary['labels_modified']}")
    print(f"Empty labels:        {summary['empty_labels']}")
    print(f"Class name:          {class_name}")
    print(f"Class ID:            {class_id}")
    print()
    print("=" * 60)
    print(f"Reduced dataset saved to: {output_dir}")
    print("=" * 60)

    return summary


def process_splitted_dataset(
    source_dir: Path,
    output_dir: Path,
    class_name: str = SINGLE_CLASS_NAME,
    class_id: int = SINGLE_CLASS_ID,
) -> dict:
    """
    Process a split dataset with ``train`` / ``val`` / ``test`` subdirectories.

    Each split must contain ``images/`` and ``labels/``.
    """
    print("=" * 60)
    print(f"Reducing Splitted Dataset to Single Class: '{class_name}'")
    print("=" * 60)
    print()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_summary: dict = {
        "train": {},
        "val": {},
        "test": {},
        "class_name": class_name,
        "class_id": class_id,
    }

    for split_type in ["train", "val", "test"]:
        split_source_dir = source_dir / split_type
        if not split_source_dir.exists():
            print(f"Skipping {split_type} (directory not found)")
            continue

        print(f"Processing {split_type} split...")
        print("-" * 60)

        split_output_dir = output_dir / split_type
        split_summary = reduce_dataset_to_one_class(
            source_dir=split_source_dir,
            output_dir=split_output_dir,
            class_name=class_name,
            class_id=class_id,
        )

        overall_summary[split_type] = split_summary
        print()

    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w") as f:
        f.write(f"{class_name}\n")
    print(f"Created classes.txt with single class: '{class_name}'")

    data_yaml_path = source_dir / "data.yaml"
    if data_yaml_path.exists():
        with open(data_yaml_path) as f:
            content = f.read()

        lines = content.split("\n")
        updated_lines: list[str] = []
        for yaml_line in lines:
            if yaml_line.startswith("nc:"):
                updated_lines.append("nc: 1")
            elif yaml_line.startswith("names:"):
                updated_lines.append(f"names: ['{class_name}']")
            else:
                updated_lines.append(
                    yaml_line.replace(str(source_dir), str(output_dir))
                )

        with open(output_dir / "data.yaml", "w") as f:
            f.write("\n".join(updated_lines))
        print("Updated data.yaml (set nc=1, updated paths)")

    for file_name in ["split_summary.json", "fusion_summary.json", "renaming_summary.json"]:
        src_file = source_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, output_dir / file_name)
            print(f"Copied {file_name}")

    print()
    print("=" * 60)
    print("Overall Reduction Summary")
    print("=" * 60)
    print()
    for split_type in ["train", "val", "test"]:
        s = overall_summary.get(split_type)
        if s:
            print(f"{split_type.capitalize()}:")
            print(f"  Images: {s.get('total_images', 0)}")
            print(f"  Labels modified: {s.get('labels_modified', 0)}")
    print()
    print(f"Class name: {class_name}")
    print(f"Class ID: {class_id}")
    print()
    print("=" * 60)
    print(f"Reduced dataset saved to: {output_dir}")
    print("=" * 60)

    return overall_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce dataset(s) to a single class 'telltale'"
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        type=str,
        help="Path(s) to dataset directory(ies) to reduce",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_telltale",
        help="Suffix to add to output directory names (default: _telltale)",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=SINGLE_CLASS_NAME,
        help=f"Name of the single class (default: {SINGLE_CLASS_NAME})",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=SINGLE_CLASS_ID,
        help=f"Class ID to use for all annotations (default: {SINGLE_CLASS_ID})",
    )

    args = parser.parse_args()

    for dataset_path in args.dataset_paths:
        source_dir = Path(dataset_path)

        if not source_dir.exists():
            print(f"Warning: Dataset not found: {source_dir}, skipping...")
            continue

        is_splitted = (source_dir / "train").exists() or (source_dir / "val").exists()

        output_dir = source_dir.parent / f"{source_dir.name}{args.output_suffix}"

        print()
        print(f"Processing: {source_dir}")
        print(f"Output: {output_dir}")
        print()

        if is_splitted:
            process_splitted_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                class_name=args.class_name,
                class_id=args.class_id,
            )
        else:
            reduce_dataset_to_one_class(
                source_dir=source_dir,
                output_dir=output_dir,
                class_name=args.class_name,
                class_id=args.class_id,
            )

        print()


if __name__ == "__main__":
    main()
