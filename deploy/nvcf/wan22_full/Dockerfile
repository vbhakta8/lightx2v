# WAN 2.2 Full Model (Vanilla/Unquantized) - NVCF Deployment
# Uses official Wan-AI/Wan2.2-I2V-A14B model with on-demand loading
# Requires: 48GB VRAM GPU (L40, A100, etc.)
# Base image: LightX2V Docker image with all dependencies pre-installed

FROM lightx2v/lightx2v:25111101-cu128

ENV DEBIAN_FRONTEND=noninteractive

# Additional system deps for NVCF
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

# Copy VAE and tokenizer (shared with distilled version)
COPY models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth /app/models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth
COPY models/Wan2.2-I2V-A14B/google /app/models/Wan2.2-I2V-A14B/google

# Copy full unquantized models (vanilla FP16/BF16)
# High-noise model (~28.6GB)
COPY models/Wan2.2-I2V-A14B/high_noise_model /app/models/Wan2.2-I2V-A14B/high_noise_model

# Low-noise model (~28.6GB)
COPY models/Wan2.2-I2V-A14B/low_noise_model /app/models/Wan2.2-I2V-A14B/low_noise_model

# T5 encoder (unquantized)
COPY models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth \
    /app/models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth

# Install LightX2V from GitHub (pinned to stable commit)
RUN pip install --no-cache-dir git+https://github.com/ModelTC/LightX2V.git@9f544e743c86a1f4f7d2c42c55e94ccb4d2f88b6

# Pin numpy to 1.x for tritonserver Python backend compatibility
RUN pip install --no-cache-dir "numpy<2.0,>=1.24"
RUN pip install nvidia-pytriton==0.7.0

# Copy triton server
RUN mkdir -p wombo/triton
COPY wombo/triton/wan_i2v_full.py ./wombo/triton/wan_i2v_full.py

# Runtime environment
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
ENV PRODUCTION=True
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WOMBO_LOG_LEVEL=DEBUG

# WAN 2.2 Full Model Configuration
ENV WAN_MODEL_PATH=/app/models/Wan2.2-I2V-A14B
ENV WAN_MODEL_CLS=wan2.2_moe
ENV WAN_TASK=i2v

# Full model uses 40 steps (not 4 like distilled)
ENV WAN_INFER_STEPS=40
ENV WAN_ENABLE_CFG=false

# No quantization - vanilla FP16/BF16
ENV WAN_DIT_QUANTIZED=false
ENV WAN_T5_QUANTIZED=false

# Memory management for 48GB VRAM + limited RAM
# unload_modules: Load models from disk on-demand, unload after use
ENV WAN_UNLOAD_MODULES=true
ENV WAN_CPU_OFFLOAD=false
ENV WAN_TEXT_ENCODER_OFFLOAD=true
ENV WAN_VAE_OFFLOAD=false

# Default generation settings (720p, ~3.3s video)
ENV WAN_HEIGHT=720
ENV WAN_WIDTH=1280
ENV WAN_NUM_FRAMES=81
ENV WAN_FPS=24

# Model switching boundary (90% = step 36 of 40)
ENV WAN_BOUNDARY=0.900

# NVIDIA optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV DTYPE=BF16

# Output directory
ENV WAN_OUTPUT_DIR=/app/outputs
RUN mkdir -p /app/outputs

CMD ["python3", "-u", "wombo/triton/wan_i2v_full.py"]
