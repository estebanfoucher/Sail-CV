FROM dustynv/l4t-pytorch:r36.4.0

LABEL description="Sail-CV Tell-Tales Tracking service (Jetson)"
ENV DEVICE="cuda"
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

# Install main project dependencies (with tracking extras)
RUN uv pip install --system -e ".[tracking]"

# Install setuptools
RUN uv pip install --system setuptools

# Install VPI (NVIDIA Vision Programming Interface)
RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
   add-apt-repository "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
       apt-get update && apt-get install -y --no-install-recommends \
     libegl1-mesa \
     libnvvpi3 vpi3-dev vpi3-samples python3.10-vpi3 && \
   add-apt-repository -r "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
   rm -rf /var/lib/apt/lists/* && \
   apt-get clean

# Create non-root user
RUN useradd -m -u 1000 app_user && \
    usermod -aG sudo app_user

# Create directories
RUN mkdir -p /app/output /app/.cache

# Copy source files
COPY src/tracking/ /app/src/tracking/
COPY analyze_video.py /app/

# Set proper ownership and permissions
RUN chown -R app_user:app_user /app && \
    chmod -R 755 /app

# Set environment
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV YOLO_CONFIG_DIR="/app/.cache"
ENV TORCH_HOME="/app/.cache/torch"

WORKDIR /app/

# Switch to non-root user
USER app_user

CMD ["tail", "-f", "/dev/null"]
