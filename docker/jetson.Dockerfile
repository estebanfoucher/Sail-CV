FROM dustynv/l4t-pytorch:r36.4.0

LABEL description="Docker container for MVS app with MASt3R and SAM integration. JETSON VERSION"
ENV DEVICE="cuda"
ENV MODEL="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
ARG DEBIAN_FRONTEND=noninteractive

# Install uv package manager (latest version via official installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory structure
WORKDIR /app
RUN mkdir -p /app/checkpoints /app/tmp/mast3r_test /app/tmp/mast3r_output

# Clone MASt3R repository to /app/mast3r for dependencies
RUN git clone --recursive https://github.com/naver/mast3r /app/mast3r
WORKDIR /app/mast3r/dust3r
RUN uv pip install --system -r requirements.txt

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install VPI via apt (NVIDIA's Vision Programming Interface)
RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
    add-apt-repository "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
    apt-get update && apt-get install -y --no-install-recommends \
      libegl1-mesa \
      libnvvpi3 vpi3-dev vpi3-samples python3.10-vpi3 && \
    add-apt-repository -r "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Skip requirements_optional.txt - not needed for basic inference
RUN uv pip install --system opencv-python==4.8.0.74 pupil-apriltags loguru ultralytics

WORKDIR /app/mast3r/dust3r/croco/models/curope/
RUN python3 setup.py build_ext --inplace

WORKDIR /app/mast3r
RUN uv pip install --system -r requirements.txt

# Download model checkpoint
RUN wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P /app/checkpoints/

# Create non-root user for better security and file management
RUN useradd -m -u 1000 app_user && \
    usermod -aG sudo app_user

# Change ownership of directories before copying files
RUN chown app_user:app_user /app/checkpoints && \
    chown app_user:app_user /app/tmp

# Copy the refactored source code
COPY --chown=app_user:app_user src/ /app/src/

# Set working directory to src
WORKDIR /app/src

# Add mast3r to Python path
ENV PYTHONPATH="/app/mast3r:$PYTHONPATH"

# Set Ultralytics config directory to avoid warnings
ENV YOLO_CONFIG_DIR="/app/"

# Switch to non-root user
USER app_user

# Keep container running
CMD ["tail", "-f", "/dev/null"]
