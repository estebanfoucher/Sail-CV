FROM dustynv/l4t-pytorch:r36.4.0 AS base

LABEL description="Sail-CV base image for Jetson hardware"
ARG DEBIAN_FRONTEND=noninteractive

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory structure
WORKDIR /app

# Copy project files for dependency installation
COPY pyproject.toml uv.lock* README.md /app/

# Install main project dependencies
RUN uv pip install --system -e .

# Install setuptools explicitly for building extensions
RUN uv pip install --system setuptools

# Create non-root user for better security and file management
RUN useradd -m -u 1000 app_user && \
    usermod -aG sudo app_user

# Create common directories
RUN mkdir -p /app/output /app/.cache

# Set Ultralytics config directory
ENV YOLO_CONFIG_DIR="/app/.cache"
ENV TORCH_HOME="/app/.cache/torch"
