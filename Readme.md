# Sail-CV

## Looking is measuring : embedded computer vision measurement of sails aerodynamic performance.

Understanding the interaction between wind flow and sails is essential for optimizing sailing performance.
Sailors traditionally rely on two qualitative visual cues —tell-tales and sail shape— but obtaining quantitative,
real-time measurements of these indicators remains challenging. Moreover, these visual cues are seldom recorded
in forms suitable for post-navigation analysis.
This work introduces an embedded computer-vision framework that quantitatively measures two key aerodynamic features: (1) boundary-layer behavior through continuous tell-tale state tracking, and (2) 3D sail
geometry via a photogrammetry-based reconstruction method. The two modules operate independently yet
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

## Get Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg
- Optional: CUDA-compatible GPU (recommended for optimal performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/estebanfoucher/sail-CV.git
   cd sail-CV
   git submodule update --init --recursive
   ```

2. **Install dependencies**

   Set up python:
   ```bash
   uv sync
   ```
   Install ffmpeg:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # MacOS
   brew install ffmpeg
   ```

3. **Activate environment**
   ```bash
   source .venv/bin/activate
   ```

4. **Add external modules to python environment**
   ```bash
   export PYTHONPATH="${PWD}/src:${PWD}/mast3r:${PWD}/mast3r/dust3r:${PYTHONPATH}"
   ```
   Note: Run this command from the project root directory

5. **Download MASt3R model checkpoint** (for 3D reconstruction)
   ```bash
   mkdir -p checkpoints/
   wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
   ```

### Quick Start — 3D Reconstruction

**Web Interface (Recommended)**
```bash
cd web_app
python main.py
```
Open your browser to `http://localhost:{PORT}` to access the interactive web interface.

### Docker (Alternative)

For containerized deployment, especially on Jetson hardware (tested on Jetson Nano):

```bash
cd docker/
docker compose build
docker compose up -d
```

### Development

```bash
make dev           # Install dependencies and setup development environment
make check         # Run all code quality checks
make test          # Run test suite
```

## Acknowledgments

This project builds upon the excellent work of [MASt3R](https://github.com/naver/mast3r) (Grounding Image Matching in 3D with MASt3R) by Naver Labs.


## Tell tales tracker module

The tell-tale tracking module —requiring only a single camera— uses a detection-plus-tracking pipeline. A
vision model is trained on a purpose-built dataset annotated with bounding boxes for attached, detached, and
leech tell-tales, as shown in Figure 1. A tracker then converts per-frame detections into time-series suitable for
aerodynamic interpretation. This machine-learning-based approach offers the robustness necessary to handle
variations in color, sail type, illumination, and object motion, showing promising behavior for reliable field use.
