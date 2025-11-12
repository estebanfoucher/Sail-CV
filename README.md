# SailCV-tell-tales-tracking

## Overview

This repository provides a pytorch-based tell-tale tracker for aerodynamics monitoring of separated flow on a sail.

It relies on a **fine tuned RT-DETR** as detector and a **custom python-based tracker**.

## Get Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg
- optional : CUDA-compatible GPU (recommended for optimal performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/estebanfoucher/SailCV-tell-tales-tracking.git &&
   cd SailCV-tell-tales-tracking
   ```

2. **Install dependencies**

   set up python:
   ```bash
   uv sync
   ```
   install ffmpeg:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # MacOS
   brew install ffmpeg
   ```

## How to use  

### Basic usage

`source .venv/bin/activate && python main.py`

### Docker 

`cd docker && docker compose -f docker-compose.yml build && docker compose -f docker-compose.yml up -d && docker exec -it docker-tell-tales-tracking-1 bash`

then execute

`python3 main.py`