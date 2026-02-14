# Sail-CV

[![GitHub](https://img.shields.io/badge/GitHub-Sail--CV-181717?logo=github)](https://github.com/estebanfoucher/Sail-CV)
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

The tell-tale tracking module —requiring only a single camera— uses a detection-plus-tracking pipeline. A
vision model is trained on a purpose-built dataset annotated with bounding boxes for attached, detached, and
leech tell-tales, as shown in Figure 1. A tracker then converts per-frame detections into time-series suitable for
aerodynamic interpretation. This machine-learning-based approach offers the robustness necessary to handle
variations in color, sail type, illumination, and object motion, showing promising behavior for reliable field use.


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

Download model checkpoints:

```bash
mkdir -p checkpoints/

# MASt3R (3D reconstruction)
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

### Quick Start — 3D Reconstruction

**CLI** — reconstruct a point cloud from a calibrated scene:

```bash
uv run python src/reconstruction/reconstruct_pair.py --scene scene_10
```

Options:

| Flag | Description |
|------|-------------|
| `--scene <name>` | Scene directory under `assets/reconstruction/` |
| `--subsample <n>` | Point correspondence subsample factor (default: 8) |
| `--render-cameras` | Export camera pyramids alongside point cloud |
| `--save-matches` | Save match correspondence renders |
| `--sam-checkpoint <path>` | Enable SAM masking with a FastSAM checkpoint |

Or use custom paths:

```bash
uv run python src/reconstruction/reconstruct_pair.py \
  --image1 path/to/cam1.png \
  --image2 path/to/cam2.png \
  --calibration path/to/calibration.json \
  --output output/my_pair
```

**Web Interface:**

```bash
cd web_app && uv run python main.py
```

### Quick Start — Tell-Tales Tracking

**CLI** — run the tracking pipeline on the C1 fixture with the classifier:

```bash
uv run python src/tracking/analyze_video.py \
  --video fixtures/C1_fixture.mp4 \
  --layout fixtures/C1_layout.json \
  --parameters parameters/default.yaml \
  --output output/tracking
```

Options:

| Flag | Description |
|------|-------------|
| `--video <path>` | Input video file |
| `--layout <path>` | Layout JSON with tell-tale positions |
| `--parameters <path>` | YAML config (default: `parameters/default.yaml`) |
| `--output <path>` | Output directory (default: `output/tracking`) |

The `default.yaml` config includes the classifier (`checkpoints/classifyer_224.pt`). To run without it, remove the `classifier:` section from the YAML.

### Docker

For containerized deployment on Jetson hardware:

```bash
make docker-build      # Build all images
make docker-up         # Start services
make docker-down       # Stop services
```

### Development

```bash
make dev               # Install dependencies + dev setup
make check             # Run all code quality checks (ruff, mypy)
make test              # Run full test suite
make test-reconstruction   # Run reconstruction tests only
make test-tracking         # Run tracking tests only
```



---


## Acknowledgments

This project builds upon the excellent work of [MASt3R](https://github.com/naver/mast3r) (Grounding Image Matching in 3D with MASt3R) by Naver Labs.


## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Creative Commons Attribution-NonCommercial-ShareAlike 4.0), consistent with the [MASt3R](https://github.com/naver/mast3r) license from Naver Corporation.

See [LICENSE](LICENSE) for full details.
