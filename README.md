# Multiview Structure

A Docker-based application for processing multiview video data on Jetson devices.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA Jetson device (for GPU acceleration)

### Setup
1. Clone the repository
2. Create output directory:
   ```bash
   mkdir -p output
   ```
3. Build and run:
   ```bash
   cd docker
   docker-compose build
   docker-compose up
   ```

## Project Structure
```
multiviewstructure/
├── src/           # Source code
├── data/          # Input video files
├── output/        # Extracted frames and results
└── docker/        # Docker configuration
```

## Usage
The application will:
- Process MP4 files from the data directory
- Extract frames using ffmpeg
- Save results to the output directory

## Data Format
- Place MP4 files in `data/scene_X/camera_Y/` directories
- Supported formats: `.mp4`, `.MP4`

## Output
- Extracted frames saved as JPEG images
- Files persist in the `output/` directory after container stops
