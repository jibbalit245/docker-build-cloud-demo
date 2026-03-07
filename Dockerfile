FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="westllc"
LABEL description="Omni-Stack: Qwen3-80B + Qwen2.5-VL-32B + Lilith-L3.3-70B + Wan2.2 — models pulled at runtime"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    OLLAMA_MODELS=/workspace/ollama_models \
    OLLAMA_HOST=0.0.0.0:11434 \
    VLLM_TP=1 \
    PATH="/usr/local/bin:$PATH"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ffmpeg openssh-server \
    libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Ollama — direct binary install (more reliable than install.sh in build envs)
RUN wget -q https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 \
    -O /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama

# Python deps — everything needed to run all 4 model servers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "vllm==0.6.3" \
    "transformers>=4.47.0" \
    "accelerate" \
    "bitsandbytes" \
    "auto-gptq" \
    "fastapi" \
    "uvicorn[standard]" \
    "httpx" \
    "pydantic" \
    "huggingface_hub[cli]" \
    "qwen-vl-utils" \
    "sentencepiece" \
    "tiktoken" \
    "einops" \
    "diffusers>=0.31.0" \
    "imageio" \
    "imageio-ffmpeg" \
    "decord" \
    "timm" \
    "numpy<2.0"

# Copy server scripts — models pulled at runtime into /workspace (network volume)
WORKDIR /app
COPY gateway.py /app/gateway.py
COPY vl_server.py /app/vl_server.py
COPY wan_server.py /app/wan_server.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000 8001 8002 8003 8888 11434 22
VOLUME ["/workspace"]
CMD ["/start.sh"]
