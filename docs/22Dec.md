# December 22, 2025

## Notes

We will focus first on one video C1.mp4

The idea is to perform tracking and detection under a certain prior, being the object to detect layout.

## YOLO Convention

**Coordinate system**: Origin `(0, 0)` is **top-left** corner of the image.
- X increases → right
- Y increases ↓ down

YOLO bounding box format uses **normalized center coordinates**:

```
x_center, y_center, width, height
```

| Field | Description | Range |
|-------|-------------|-------|
| `x_center` | Center X coordinate (normalized) | 0.0 - 1.0 |
| `y_center` | Center Y coordinate (normalized) | 0.0 - 1.0 |
| `width` | Box width (normalized) | 0.0 - 1.0 |
| `height` | Box height (normalized) | 0.0 - 1.0 |

**Normalization**: Values are relative to image dimensions:
- `x_center = pixel_x / image_width`
- `y_center = pixel_y / image_height`

**Detection output** typically includes: `x_center, y_center, width, height, confidence, class_id`

## Layout-Based Tracker

The `LayoutTracker` assigns detections to predefined layout positions using a weighted cost function:

```
cost = alpha * normalized_distance + beta * (1 - confidence)
```

- **alpha** (default 0.7): Weight for spatial distance
- **beta** (default 0.3): Weight for detection confidence
- Uses Hungarian algorithm for optimal 1-to-1 assignment
- Track IDs are layout position IDs (e.g., "TL", "TR")

## Running the Plotter

### Basic Usage (solid background)

```bash
conda run -n base python plot_detections.py \
  --detections assets/raw_detection/C1_raw_detection.json \
  --layout assets/layouts/C1_layout.json \
  --output assets/C1_plot.mp4
```

### With Layout Tracker

```bash
conda run -n base python plot_detections.py \
  --detections assets/raw_detection/C1_raw_detection.json \
  --layout assets/layouts/C1_layout.json \
  --output assets/C1_tracked.mp4 \
  --use-tracker \
  --conf-thresh 0.3
```

### With Video Background + Tracker

```bash
conda run -n base python plot_detections.py \
  --detections assets/raw_detection/C1_raw_detection.json \
  --layout assets/layouts/C1_layout.json \
  --output assets/C1_tracked_overlay.mp4 \
  --use-tracker \
  --conf-thresh 0.3 \
  --video assets/DS_6/C1.mp4
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--detections` | (required) | Path to raw detection JSON file |
| `--layout` | (required) | Path to layout JSON file |
| `--output` | `output_plot.mp4` | Output video path |
| `--video` | None | Background video (uses solid color if not provided) |
| `--use-tracker` | False | Enable LayoutTracker for assignment |
| `--conf-thresh` | 0.0 | Minimum detection confidence |
| `--alpha` | 0.7 | Distance weight in cost function |
| `--beta` | 0.3 | Confidence weight in cost function |
| `--max-distance` | 0.2 | Max normalized distance for valid match |
| `--width` | 1280 | Frame width (ignored if --video provided) |
| `--height` | 720 | Frame height (ignored if --video provided) |
| `--fps` | 30.0 | Output FPS (ignored if --video provided) |
| `--bg-color` | 30 30 30 | Background BGR color |

