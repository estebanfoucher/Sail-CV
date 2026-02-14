FROM sail-cv-base AS base

LABEL description="Sail-CV Tell-Tales Tracking service (Jetson)"
ENV DEVICE="cuda"

WORKDIR /app

# Install main project dependencies (with tracking extras)
RUN uv pip install --system -e ".[tracking]"

# Install VPI (NVIDIA Vision Programming Interface)
RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
   add-apt-repository "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
       apt-get update && apt-get install -y --no-install-recommends \
     libegl1-mesa \
     libnvvpi3 vpi3-dev vpi3-samples python3.10-vpi3 && \
   add-apt-repository -r "deb https://repo.download.nvidia.com/jetson/common r36.4 main" && \
   rm -rf /var/lib/apt/lists/* && \
   apt-get clean

# Copy source files
COPY src/tracking/ /app/src/tracking/
COPY analyze_video.py /app/

# Set proper ownership and permissions
RUN chown -R app_user:app_user /app && \
    chmod -R 755 /app

# Add tracking source to Python path
ENV PYTHONPATH="/app/src/tracking:/app/src:$PYTHONPATH"

# Switch to non-root user
USER app_user

WORKDIR /app/
CMD ["tail", "-f", "/dev/null"]
