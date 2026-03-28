# Sail-CV

[![GitHub](https://img.shields.io/badge/GitHub-Sail--CV-181717?logo=github)](https://github.com/estebanfoucher/Sail-CV)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Weights-yellow)](https://huggingface.co/estefoucher/tell-tale-detector)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Looking is measuring : embedded computer vision measurement of sails aerodynamic performance.


This work introduces an embedded computer-vision framework that quantitatively measures
1.  3D sail geometry via a photogrammetry-based reconstruction method
2.  boundary-layer behavior through continuous tell-tale state tracking, and.

The two modules operate independently yet
share the same minimal hardware requirements, enabling practical, plug-and-play deployment across a broad
range of yachts.


---

## 3D Reconstruction Module

The 3D reconstruction module aims to recover accurate, metric point clouds of the sail surface from calibrated
stereo imagery. The method leverages two core components: (i) the ability of AI-based reconstruction models to
generate dense point correspondences between two viewpoints, and (ii) precise intrinsic and extrinsic calibration
of a general two-camera setup, enabling accurate triangulation and conversion of correspondences into 3D
coordinates. This approach eliminates the need for applied texture or detailed geometric priors on the sail.
The paper presents the details of the methods and the training of the tell-tale detector. Generic results taken
in real conditions on yachts are showed together with a more in-depth analysis of tell-tales on the rigid wings of
a model wind-powered vessel, comparing with another tell-tales detection method and pressure measurements.

### Example : mainsail sheeting

| Combined View (input)|
|:-------------:|
| ![Combined View](assets/reconstruction/scene_7/gifs/combined.gif) |

| Front View |
|:----------:|
| <img src="assets/reconstruction/scene_7/gifs/render_front.gif" width="50%" alt="Front View"> |

| Bottom View | Top View |
|:-----------:|:--------:|
| ![Bottom View](assets/reconstruction/scene_7/gifs/render_bot.gif) | ![Top View](assets/reconstruction/scene_7/gifs/render_top.gif) |

### Example : jibsail tacking

| Combined View (input)|
|:-------------:|
| ![Combined View](assets/reconstruction/scene_8/gifs/combined.gif) |

| Front View |
|:----------:|
| <img src="assets/reconstruction/scene_8/gifs/render_front.gif" width="50%" alt="Front View"> |

| Bottom View | Top View |
|:-----------:|:--------:|
| ![Bottom View](assets/reconstruction/scene_8/gifs/render_bot.gif) | ![Top View](assets/reconstruction/scene_8/gifs/render_top.gif) |

---

## Tell tales tracker module

The tell-tale tracking module uses a single-camera, detection-plus-tracking pipeline. A vision model trained on a purpose-built dataset (bounding boxes for attached, detached, and leech tell-tales) feeds a tracker that turns per-frame detections into time-series for aerodynamic interpretation. The approach is robust to variations in color, sail type, illumination, and motion.

### Model weights (detector & classifier)

Weights can be supplied in two ways:

- **Local path** — Put checkpoint files in `checkpoints/` (or another path) and set `model_path` in your parameters YAML (e.g. `parameters/default_classifier.yml`). Paths are resolved from the project root.
- **Hugging Face** — If a file is not found locally, the code tries to download it from [estefoucher/tell-tale-detector](https://huggingface.co/estefoucher/tell-tale-detector) (folder `weights/`). Use the exact filename in your config:
  - **Classifier:** `sailcv-yolo11n-cls224.pt`
  - **Detector:** `sailcv-rtdetrl1088.pt`, `sailcv-rtdetrl640.pt`

You can keep weights in `checkpoints/` or rely on automatic download by using these filenames in your config.

### User guide

To run the tracker on a new video, create a layout with the layout annotator, then run `analyze_video.py` with a chosen parameters file. **Get Started → Quick Start — Tell-Tales Tracking** below gives the full steps and example commands (layout creation, pipeline run, and config options).

**Colab:** Open [notebooks/telltale_tracker_colab.ipynb](notebooks/telltale_tracker_colab.ipynb) in Google Colab to run the pipeline with default parameters: choose classifier or vector mode, use built-in samples (C1 or C4) or upload your own video and layout.

## Get Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg
- CUDA-compatible GPU (recommended, required for real-time tracking)

### Installation

```bash
git clone https://github.com/estebanfoucher/sail-CV.git
cd sail-CV
git submodule update --init --recursive
```

Install Python dependencies:

```bash
uv sync --all-extras
```

Install ffmpeg:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```


### Quick Start — 3D Reconstruction

Download model checkpoint:

```bash
mkdir -p checkpoints/

# MASt3R (3D reconstruction)
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

Set up the Python path for reconstruction (run from project root):

```bash
export PYTHONPATH="${PWD}/src/reconstruction:${PWD}/mast3r:${PWD}/mast3r/dust3r"
```

Reconstruct a point cloud from a calibrated scene:

```bash
uv run python src/reconstruction/reconstruct_pair.py --scene scene_10
```

With more options:

```bash
uv run python src/reconstruction/reconstruct_pair.py \
  --scene scene_10 --render-cameras --save-matches --subsample 4
```

Launch the web interface:

```bash
uv run python web_app/main.py
```

### Quick Start — Tell-Tales Tracking

**Try in Colab:** [notebooks/telltale_tracker_colab.ipynb](notebooks/telltale_tracker_colab.ipynb) — clone, install, then run the pipeline (classifier or vector) on built-in samples or your own video and layout, without a local setup.

When you have a **new video and no layout** (e.g. `assets/tracking/DS_6/C4.mp4`), follow these two steps. All commands are run from the project root.

**Step 1: Create the layout**

Run the layout annotator (from project root):

```bash
uv run python src/tracking/annotate_layout_opencv.py \
  --video assets/tracking/DS_6/C4.mp4 \
  --time-sec 3 \
  --output output/tracking_layouts/DS_6/C4_layout.json
```

- **Left-click** on the frame to add a tell-tale position; the terminal will prompt for **id** (e.g. `TL`) and **name** (e.g. `top left`). Repeat for every tell-tale.
- **s** — save layout to the output file and exit.
- **q** or **Esc** — quit without saving.
- **u** — undo last point; **c** — clear all points.
- Optionally use `--direction dx dy` to set a direction prior in the JSON. Use `--frame N` instead of `--time-sec` to pick a frame by index.

**Step 2: Run the tracking pipeline**

Behavior (detector, classifier, masks, PCA arrows, output options) is controlled by the **parameters file**. The single entry point is `analyze_video.py`:

```bash
uv run python src/tracking/analyze_video.py \
  --video assets/tracking/DS_6/C4.mp4 \
  --layout output/tracking_layouts/DS_6/C4_layout.json
```

This uses the default config `parameters/default_classifier.yml` (bboxes + classifier labels, no masks/arrows). For masks and PCA arrows without the classifier, use `parameters/default_vector.yml`:

```bash
uv run python src/tracking/analyze_video.py \
  --video assets/tracking/DS_6/C4.mp4 \
  --layout output/tracking_layouts/DS_6/C4_layout.json \
  --parameters parameters/default_vector.yml
```

The `test_classif.yml` and `test_vector.yml` configs use fixture detections for fast runs (e.g. CI).

Output is written to `output/tracking/` by default (JSON plus optional tracking video and fgmask). Use `--output` to change the directory.

**Frame range:** By default all frames are processed (`--frame-start 0`, `--frame-end -1`; `-1` means last frame). To analyze only the first 10 frames (optionally with a specific config):

```bash
uv run python src/tracking/analyze_video.py --video assets/tracking/DS_6/C4.mp4 --layout output/tracking_layouts/DS_6/C4_layout.json --frame-start 0 --frame-end 9 --parameters parameters/default_vector.yml
```


**Parameters:** Files in `parameters/` (`default_classifier.yml`, `default_vector.yml`, and test configs) define the detector, classifier, crop module, and output options (main tracking video, mask overlay, PCA arrows). See `parameters/` for available configs.

If you already have a layout (e.g. from the fixtures):

```bash
uv run python src/tracking/analyze_video.py \
  --video fixtures/C1_fixture.mp4 \
  --layout fixtures/C1_layout.json
```

### Finetuning the detector

Export annotations from the annotator, turn them into train/val if needed, then (for RT-DETR finetuning) use **one class** and a **split** layout. Tools: `finetuning/annotator/`, `split_yolo_by_source.py`, `reduce_to_one_class.py`, `finetune_rtdetr.ipynb`.

**1. Create your custom dataset**

- Open `finetuning/annotator/index.html` in a browser (or serve the `finetuning/annotator/` folder).
- Load photos and/or videos. For video: seek to a frame, click **Add current frame** (frames get a time prefix for grouping).
- Annotate boxes with the three classes: **attached**, **detached**, **leech**.
- **Export YOLO zip** — you get `images/`, `labels/`, and `data.yaml` (flat layout, 3 classes).

**2. From zip to folders**

Unzip so you have a single directory (call it `EXPORT`) that contains `images/` and `labels/`. Run the commands below from the **project root**. If unzipping drops `images/` and `labels/` in the current directory, use `.` instead of `EXPORT`.

Two separate ideas:

- **Split** — build `train/` and `val/` so one video does not appear in both (script: `split_yolo_by_source.py`).
- **Reduce** — rewrite every box to class `telltale` / `nc: 1` (default behavior of the split script, or `reduce_to_one_class.py` alone).

What to run:

| You want | Command |
|----------|---------|
| Split **and** 1 class (usual input for `finetune_rtdetr.ipynb`) | `uv run python finetuning/split_yolo_by_source.py EXPORT -o OUT` |
| Split **without** changing classes (still 3 classes) | Same command, add `--no-reduce` (export must include `data.yaml` or `classes.txt`) |
| 1 class **without** train/val split | `uv run python finetuning/reduce_to_one_class.py EXPORT` → creates `EXPORT_telltale` next to `EXPORT` |

Split script options: `--train-ratio 0.8`, `--seed 42`. If you already have a **3-class split** folder and only need 1 class, run `reduce_to_one_class.py` on **that folder** (the one with `train/` and `val/`), not on the original flat export.

**Example** (`sailcv_annotations_yolo.zip` at project root):

```bash
unzip -o sailcv_annotations_yolo.zip
uv run python finetuning/split_yolo_by_source.py sailcv_annotations_yolo -o sailcv_fm_split --seed 42
```

**3. Finetune**

- Open `finetuning/finetune_rtdetr.ipynb` (Colab or local Jupyter).
- Run all cells. On Colab: enable GPU (Runtime → Change runtime type → GPU), then when prompted upload the zip of your **split** custom dataset (Step 2 output).
- The notebook downloads the main dataset ([estefoucher/sail-cv-telltales](https://huggingface.co/datasets/estefoucher/sail-cv-telltales)), fuses it with your data, downloads the 640 checkpoint ([estefoucher/tell-tale-detector](https://huggingface.co/estefoucher/tell-tale-detector)), and trains 10–15 epochs. Best weights: `runs/train/finetune_rtdetr/weights/best.pt`.

**4. Use the new weights**

Set `model_path` in your parameters YAML to that `best.pt`, or copy it to `checkpoints/` and reference by filename.

### Finetuning the classifier (3 classes, Colab)

Tell-tale **state** classification uses the same three classes as the annotator. Split the export **without** reducing to one class:

| Step | Command / action |
|------|------------------|
| Split | `uv run python finetuning/split_yolo_by_source.py EXPORT -o OUT --no-reduce --seed 42` |
| Zip | Zip `OUT/` so it contains `train/`, `val/`, and `data.yaml` (`nc` / `names` must match the HF dataset below) |
| Train | Open `finetuning/train_classifier.ipynb` on Colab (GPU). It downloads **`telltale_3_classes_fused.zip`** from [estefoucher/sail-cv-telltales](https://huggingface.co/datasets/estefoucher/sail-cv-telltales), uploads your zip and `finetuning/crop_dataset.py`, fuses, crops, and fine-tunes `yolo11n-cls`. Best weights: `runs/classify/train/weights/best.pt`. |

**Local crops only** (YOLO split → Ultralytics cls folders): `uv run python finetuning/crop_dataset.py --train PATH/train --val PATH/val --output data/cls_dataset` (add `--multipliers '{"0":1,"1":5}'` or omit for 1:1 per class).

### Docker

For containerized deployment on Jetson hardware:

```bash
cd docker/
docker compose build
docker compose up -d
```

### Development

```bash
uv sync --all-extras --group dev   # Install all dependencies
uv run pytest tests/               # Run full test suite
uv run ruff check src/ tests/      # Lint
uv run ruff format src/ tests/     # Format
```



---


## Acknowledgments

This project builds upon the excellent work of [MASt3R](https://github.com/naver/mast3r) (Grounding Image Matching in 3D with MASt3R) by Naver Labs.


## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Creative Commons Attribution-NonCommercial-ShareAlike 4.0), consistent with the [MASt3R](https://github.com/naver/mast3r) license from Naver Corporation.

See [LICENSE](LICENSE) for full details.
