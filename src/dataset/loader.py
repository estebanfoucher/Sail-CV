"""YOLO dataset loader for image-label pairs."""

import random
from pathlib import Path

import cv2
import numpy as np

from models import BoundingBox, Detection, Image


class YOLODatasetLoader:
    """
    Loader for YOLO format datasets.

    Expects dataset structure:
        dataset_path/
            images/
                *.jpg (or other image formats)
            labels/
                *.txt (YOLO format: class_id x_center y_center width height)
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize the YOLO dataset loader.

        Args:
            dataset_path: Path to the dataset root directory containing images/ and labels/ folders
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

    def _get_image_files(self) -> list[Path]:
        """Get all image files from the images directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in self.images_dir.iterdir() if f.suffix.lower() in image_extensions
        ]
        return sorted(image_files)

    def _get_label_path(self, image_path: Path) -> Path:
        """Get corresponding label file path for an image."""
        label_name = image_path.stem + ".txt"
        return self.labels_dir / label_name

    def _parse_yolo_label(
        self, label_path: Path, img_width: int, img_height: int
    ) -> list[Detection]:
        """
        Parse YOLO format label file and convert to Detection objects.

        YOLO format: class_id x_center y_center width height (all normalized 0-1)

        Args:
            label_path: Path to label file
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of Detection objects with confidence=1.0
        """
        detections = []

        if not label_path.exists():
            return detections

        with open(label_path) as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                parts = stripped_line.split()
                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert normalized YOLO format to absolute XYXY coordinates
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(x1 + 1, min(x2, img_width))
                    y2 = max(y1 + 1, min(y2, img_height))

                    # Create bounding box and detection
                    bbox = BoundingBox.from_numpy(np.array([x1, y1, x2, y2]))
                    detection = Detection(bbox=bbox, confidence=1.0, class_id=class_id)
                    detections.append(detection)

                except (ValueError, IndexError):
                    # Skip invalid lines
                    continue

        return detections

    def load_pair(
        self, image_name: str | None = None
    ) -> tuple[str, Image, list[Detection]]:
        """
        Load an image-label pair from the dataset.

        Args:
            image_name: Name of the image file (without path). If None, loads a random image.

        Returns:
            Tuple of (image_name, Image object, list[Detection])
        """
        image_files = self._get_image_files()

        if not image_files:
            raise ValueError(f"No image files found in {self.images_dir}")

        if image_name is None:
            # Load random image
            image_path = random.choice(image_files)
        else:
            # Load specific image
            image_path = self.images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img_array = cv2.imread(str(image_path))
        if img_array is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_height, img_width = img_array.shape[:2]
        image = Image(image=img_array, rgb_bgr="BGR")

        # Load and parse label
        label_path = self._get_label_path(image_path)
        detections = self._parse_yolo_label(label_path, img_width, img_height)

        return image_path.name, image, detections

    def get_all_pairs(self) -> list[tuple[str, Image, list[Detection]]]:
        """
        Get all image-label pairs from the dataset.

        Returns:
            List of tuples (image_name, Image, list[Detection])
        """
        image_files = self._get_image_files()
        pairs = []

        for image_path in image_files:
            try:
                img_array = cv2.imread(str(image_path))
                if img_array is None:
                    continue

                img_height, img_width = img_array.shape[:2]
                image = Image(image=img_array, rgb_bgr="BGR")

                label_path = self._get_label_path(image_path)
                detections = self._parse_yolo_label(label_path, img_width, img_height)

                pairs.append((image_path.name, image, detections))
            except Exception:
                # Skip problematic files
                continue

        return pairs

    def get_random_pairs(self, n: int) -> list[tuple[str, Image, list[Detection]]]:
        """
        Get n random image-label pairs from the dataset.

        Args:
            n: Number of pairs to return

        Returns:
            List of tuples (image_name, Image, list[Detection])
        """
        all_pairs = self.get_all_pairs()

        if len(all_pairs) < n:
            return all_pairs

        return random.sample(all_pairs, n)
