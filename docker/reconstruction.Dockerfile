FROM dustynv/l4t-pytorch:r36.4.0

LABEL description="Sail-CV 3D Reconstruction service (Jetson)"
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
COPY mast3r/ /app/mast3r/

# Install mast3r dependencies first
WORKDIR /app/mast3r
RUN uv pip install --system -r requirements.txt

# Install dust3r dependencies
WORKDIR /app/mast3r/dust3r
RUN uv pip install --system -r requirements.txt

# Install main project dependencies (with reconstruction extras)
WORKDIR /app
RUN uv pip install --system -e ".[reconstruction]"

# Install setuptools for building extensions
RUN uv pip install --system setuptools

# Build croco extension
WORKDIR /app/mast3r/dust3r/croco/models/curope/
RUN python3 setup.py build_ext --inplace

# Create non-root user
RUN useradd -m -u 1000 app_user && \
    usermod -aG sudo app_user

# Create directories
RUN mkdir -p /app/output

# Copy source files
COPY src/reconstruction/ /app/src/reconstruction/

# Copy web app files
COPY web_app/ /app/web_app/

# Set proper ownership and permissions
RUN chown -R app_user:app_user /app/src /app/output /app/web_app && \
    chmod -R 755 /app/output /app/web_app

# Set working directory to src
WORKDIR /app/src

# Add mast3r and dust3r to Python path
ENV PYTHONPATH="/app/src:/app/mast3r:/app/mast3r/dust3r:$PYTHONPATH"

# Switch to non-root user
USER app_user

# Install web app dependencies
RUN uv pip install --system -r /app/web_app/requirements.txt

EXPOSE 7860

WORKDIR /app/web_app
CMD ["tail", "-f", "/dev/null"]
