"""
Script to reduce datasets from multiple classes to a single class "telltale".

Converts all class IDs in label files to 0 and updates classes.txt to only contain "telltale".
This creates identical datasets with just one class for binary classification.
"""
import sys
from pathlib import Path
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Single class name
SINGLE_CLASS_NAME = "telltale"
SINGLE_CLASS_ID = 0


def reduce_dataset_to_one_class(
    source_dir: Path,
    output_dir: Path,
    class_name: str = SINGLE_CLASS_NAME,
    class_id: int = SINGLE_CLASS_ID
) -> dict:
    """
    Reduce a dataset to a single class by converting all labels to class_id 0.

    Args:
        source_dir: Path to source dataset directory
        output_dir: Path to output directory for reduced dataset
        class_name: Name of the single class (default: "telltale")
        class_id: Class ID to use for all annotations (default: 0)

    Returns:
        Dictionary containing reduction summary
    """
    print("=" * 60)
    print(f"Reducing Dataset to Single Class: '{class_name}'")
    print("=" * 60)
    print()

    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_images_dir = source_dir / 'images'
    source_labels_dir = source_dir / 'labels'

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {source_images_dir}")
    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {source_labels_dir}")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in source_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images to process")
    print()

    summary = {
        'total_images': len(image_files),
        'labels_processed': 0,
        'labels_modified': 0,
        'empty_labels': 0,
        'class_name': class_name,
        'class_id': class_id
    }

    # Process each image-label pair
    for img_path in image_files:
        # Copy image
        dest_img = output_images_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # Process corresponding label
        label_name = img_path.stem + '.txt'
        src_label = source_labels_dir / label_name
        dest_label = output_labels_dir / label_name

        if src_label.exists():
            # Read original label file
            with open(src_label, 'r') as f:
                lines = f.readlines()

            # Process each line: replace class ID with 0
            modified_lines = []
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                if len(parts) >= 5:  # YOLO format: class_id x y w h
                    # Replace first number (class_id) with target class_id
                    parts[0] = str(class_id)
                    modified_lines.append(' '.join(parts))

            # Write modified label file
            if modified_lines:
                with open(dest_label, 'w') as f:
                    f.write('\n'.join(modified_lines) + '\n')
                summary['labels_processed'] += 1
                summary['labels_modified'] += 1
            else:
                # Empty label file (no annotations)
                dest_label.touch()  # Create empty file
                summary['labels_processed'] += 1
                summary['empty_labels'] += 1
        else:
            # No label file, create empty one
            dest_label.touch()
            summary['labels_processed'] += 1
            summary['empty_labels'] += 1

    # Create/overwrite classes.txt with single class
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w') as f:
        f.write(f"{class_name}\n")
    print(f"Created classes.txt with single class: '{class_name}'")

    # Copy other files if they exist (data.yaml, etc.)
    for file_name in ['data.yaml', 'split_summary.json', 'fusion_summary.json', 'renaming_summary.json']:
        src_file = source_dir / file_name
        if src_file.exists():
            # For data.yaml, update class count
            if file_name == 'data.yaml':
                with open(src_file, 'r') as f:
                    content = f.read()

                # Update number of classes to 1
                lines = content.split('\n')
                updated_lines = []
                for line in lines:
                    if line.startswith('nc:'):
                        updated_lines.append('nc: 1')
                    elif line.startswith('names:'):
                        updated_lines.append(f"names: ['{class_name}']")
                    else:
                        updated_lines.append(line)

                with open(output_dir / file_name, 'w') as f:
                    f.write('\n'.join(updated_lines))
                print(f"Updated {file_name} (set nc=1)")
            else:
                shutil.copy2(src_file, output_dir / file_name)
                print(f"Copied {file_name}")

    # Print summary
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
    class_id: int = SINGLE_CLASS_ID
) -> dict:
    """
    Process a splitted dataset (with train/val/test subdirectories).

    Args:
        source_dir: Path to source splitted dataset directory
        output_dir: Path to output directory for reduced dataset
        class_name: Name of the single class (default: "telltale")
        class_id: Class ID to use for all annotations (default: 0)

    Returns:
        Dictionary containing reduction summary for all splits
    """
    print("=" * 60)
    print(f"Reducing Splitted Dataset to Single Class: '{class_name}'")
    print("=" * 60)
    print()

    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_summary = {
        'train': {},
        'val': {},
        'test': {},
        'class_name': class_name,
        'class_id': class_id
    }

    # Process each split (train, val, test)
    for split_type in ['train', 'val', 'test']:
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
            class_id=class_id
        )

        overall_summary[split_type] = split_summary
        print()

    # Create/overwrite classes.txt at root level
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w') as f:
        f.write(f"{class_name}\n")
    print(f"Created classes.txt with single class: '{class_name}'")

    # Copy and update data.yaml if it exists
    data_yaml_path = source_dir / 'data.yaml'
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            content = f.read()

        # Update paths and class count
        lines = content.split('\n')
        updated_lines = []
        for line in lines:
            if line.startswith('nc:'):
                updated_lines.append('nc: 1')
            elif line.startswith('names:'):
                updated_lines.append(f"names: ['{class_name}']")
            else:
                # Update paths to point to output directory
                line = line.replace(str(source_dir), str(output_dir))
                updated_lines.append(line)

        with open(output_dir / 'data.yaml', 'w') as f:
            f.write('\n'.join(updated_lines))
        print("Updated data.yaml (set nc=1, updated paths)")

    # Copy other summary files
    for file_name in ['split_summary.json', 'fusion_summary.json', 'renaming_summary.json']:
        src_file = source_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, output_dir / file_name)
            print(f"Copied {file_name}")

    # Print overall summary
    print()
    print("=" * 60)
    print("Overall Reduction Summary")
    print("=" * 60)
    print()
    for split_type in ['train', 'val', 'test']:
        if split_type in overall_summary and overall_summary[split_type]:
            summary = overall_summary[split_type]
            print(f"{split_type.capitalize()}:")
            print(f"  Images: {summary.get('total_images', 0)}")
            print(f"  Labels modified: {summary.get('labels_modified', 0)}")
    print()
    print(f"Class name: {class_name}")
    print(f"Class ID: {class_id}")
    print()
    print("=" * 60)
    print(f"Reduced dataset saved to: {output_dir}")
    print("=" * 60)

    return overall_summary


def main():
    """Main function to reduce datasets to single class."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reduce dataset(s) to a single class 'telltale'"
    )
    parser.add_argument(
        'dataset_paths',
        nargs='+',
        type=str,
        help='Path(s) to dataset directory(ies) to reduce'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='_telltale',
        help='Suffix to add to output directory names (default: _telltale)'
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default=SINGLE_CLASS_NAME,
        help=f'Name of the single class (default: {SINGLE_CLASS_NAME})'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=SINGLE_CLASS_ID,
        help=f'Class ID to use for all annotations (default: {SINGLE_CLASS_ID})'
    )

    args = parser.parse_args()

    for dataset_path in args.dataset_paths:
        source_dir = Path(dataset_path)

        if not source_dir.exists():
            print(f"Warning: Dataset not found: {source_dir}, skipping...")
            continue

        # Determine if it's a splitted dataset (has train/val subdirectories)
        is_splitted = (source_dir / 'train').exists() or (source_dir / 'val').exists()

        # Create output directory name
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
                class_id=args.class_id
            )
        else:
            reduce_dataset_to_one_class(
                source_dir=source_dir,
                output_dir=output_dir,
                class_name=args.class_name,
                class_id=args.class_id
            )

        print()


if __name__ == "__main__":
    main()
