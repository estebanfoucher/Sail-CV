FROM sail-cv-base AS base

LABEL description="Sail-CV 3D Reconstruction service (Jetson)"

WORKDIR /app

# Copy mast3r submodule
COPY mast3r/ /app/mast3r/

# Install mast3r dependencies
WORKDIR /app/mast3r
RUN uv pip install --system -r requirements.txt

# Install dust3r dependencies
WORKDIR /app/mast3r/dust3r
RUN uv pip install --system -r requirements.txt

# Install main project dependencies (with reconstruction extras)
WORKDIR /app
RUN uv pip install --system -e ".[reconstruction]"

# Build croco extension
WORKDIR /app/mast3r/dust3r/croco/models/curope/
RUN python3 setup.py build_ext --inplace

# Copy source files
WORKDIR /app
COPY src/reconstruction/ /app/src/reconstruction/

# Copy web app files
COPY web_app/ /app/web_app/

# Install web app dependencies
RUN uv pip install --system -r /app/web_app/requirements.txt

# Set proper ownership and permissions
RUN chown -R app_user:app_user /app/src /app/output /app/web_app && \
    chmod -R 755 /app/output /app/web_app

# Add mast3r, dust3r, and reconstruction source to Python path
ENV PYTHONPATH="/app/src/reconstruction:/app/src:/app/mast3r:/app/mast3r/dust3r:$PYTHONPATH"

# Switch to non-root user
USER app_user

EXPOSE 7860

WORKDIR /app/web_app
CMD ["tail", "-f", "/dev/null"]
