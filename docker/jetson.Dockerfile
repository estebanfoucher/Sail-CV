FROM dustynv/l4t-pytorch:r36.4.0

LABEL description="Docker container for tell-tales tracking. JETSON VERSION"
ENV DEVICE="cuda"
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

# Copy project files for dependency installation
COPY pyproject.toml uv.lock* README.md main.py /app/

# Install main project dependencies
WORKDIR /app
RUN uv pip install --system -e .

# Install setuptools explicitly for building extensions
RUN uv pip install --system setuptools

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install VPI via apt (NVIDIA's Vision Programming Interface)
# RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
#    add-apt-repository "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
#        apt-get update && apt-get install -y --no-install-recommends \
#      libegl1-mesa \
#      libnvvpi3 vpi3-dev vpi3-samples python3.10-vpi3 && \
#    add-apt-repository -r "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
#    rm -rf /var/lib/apt/lists/* && \
#    apt-get clean

# Create non-root user for better security and file management
RUN useradd -m -u 1000 app_user && \
    usermod -aG sudo app_user

# Create directories first
RUN mkdir -p /app/output

# Copy source files
COPY src/ /app/src/

# Set proper ownership and permissions for all directories
RUN chown -R app_user:app_user /app/src /app/output && \
    chmod -R 755 /app/output

# Set working directory to src
WORKDIR /app/

COPY generate_raw_detection.py /app/

# Set Ultralytics config directory to avoid warnings
ENV YOLO_CONFIG_DIR="/app/"

# Switch to non-root user
USER app_user

CMD ["tail", "-f", "/dev/null"]
