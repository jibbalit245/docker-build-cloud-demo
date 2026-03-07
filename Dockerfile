FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="omni-stack"
LABEL description="Omni-Stack: deps only - app files pulled from GitHub at pod startup"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    OLLAMA_MODELS=/workspace/ollama_models \
    OLLAMA_HOST=0.0.0.0:11434 \
    VLLM_TP=1 \
    WEBUI_AUTH=False \
    OPENAI_API_BASE_URL=http://localhost:8000/v1 \
    OPENAI_API_KEY=none

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ffmpeg zstd openssh-server \
    libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "vllm==0.6.3" \
    "transformers>=4.47.0" \
    "accelerate" \
    "bitsandbytes" \
    "fastapi" \
    "uvicorn[standard]" \
    "httpx" \
    "pydantic" \
    "huggingface_hub[cli]" \
    "qwen-vl-utils" \
    "sentencepiece" \
    "tiktoken" \
    "einops" \
    "auto-gptq" \
    "diffusers>=0.31.0" \
    "imageio" \
    "imageio-ffmpeg" \
    "av" \
    "decord" \
    "easydict" \
    "ftfy" \
    "timm" \
    "numpy<2.0" \
    "open-webui"

# start.sh pulls app files from GitHub on first boot
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 3000 8000 8001 8002 8003 8888 11434 22
VOLUME ["/workspace"]
CMD ["/start.sh"]
