FROM dustynv/l4t-pytorch:r36.4.0

LABEL description="Docker container for MASt3R with dependencies installed. JETSON VERSION"
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

RUN git clone --recursive https://github.com/naver/mast3r /mast3r
WORKDIR /mast3r/dust3r
RUN uv pip install --system -r requirements.txt
# Skip requirements_optional.txt - not needed for basic inference
RUN uv pip install --system opencv-python==4.8.0.74

WORKDIR /mast3r/dust3r/croco/models/curope/
RUN python3 setup.py build_ext --inplace

WORKDIR /mast3r
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system loguru
RUN uv pip install --system ultralytics
RUN python3 -c "from ultralytics import FastSAM; FastSAM('FastSAM-x.pt')"

# Download model checkpoint
RUN mkdir -p checkpoints/
RUN wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/

# Create non-root user for better security and file management
RUN useradd -m -u 1000 mast3r_user && \
    usermod -aG sudo mast3r_user

# Create tmp directory (will be mounted as volume)
RUN mkdir -p /mast3r/tmp/mast3r_test /mast3r/tmp/mast3r_output

# Copy the modular Python inference scripts
COPY mast3r_inference_core.py /mast3r/mast3r_inference_core.py
COPY mast3r_postprocess.py /mast3r/mast3r_postprocess.py
COPY mast3r_saver.py /mast3r/mast3r_saver.py
COPY mast3r_utils.py /mast3r/mast3r_utils.py
COPY run_mast3r.py /mast3r/run_mast3r.py
COPY convert_calibration.py /mast3r/convert_calibration.py
COPY triangulate_matches.py /mast3r/triangulate_matches.py
COPY image.py /mast3r/image.py
COPY main.py /mast3r/main.py

# Copy the SAM module
COPY sam.py /mast3r/sam.py
COPY test_fastsam.py /mast3r/test_fastsam.py
RUN chmod +x /mast3r/run_mast3r.py
RUN chmod +x /mast3r/convert_calibration.py
RUN chmod +x /mast3r/triangulate_matches.py
RUN chmod +x /mast3r/image.py
RUN chmod +x /mast3r/main.py

# Change ownership of only the necessary files and directories
RUN chown mast3r_user:mast3r_user /mast3r/tmp && \
    chown -R mast3r_user:mast3r_user /mast3r/tmp/mast3r_test && \
    chown -R mast3r_user:mast3r_user /mast3r/tmp/mast3r_output && \
    chown mast3r_user:mast3r_user /mast3r/mast3r_inference_core.py && \
    chown mast3r_user:mast3r_user /mast3r/mast3r_postprocess.py && \
    chown mast3r_user:mast3r_user /mast3r/mast3r_saver.py && \
    chown mast3r_user:mast3r_user /mast3r/mast3r_utils.py && \
    chown mast3r_user:mast3r_user /mast3r/run_mast3r.py && \
    chown mast3r_user:mast3r_user /mast3r/convert_calibration.py

# Switch to non-root user
USER mast3r_user

ENTRYPOINT ["python3", "/mast3r/main.py"]
