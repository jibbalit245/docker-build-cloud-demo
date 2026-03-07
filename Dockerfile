FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="omni-stack"
LABEL description="Unified AI stack: Qwen3-Next-80B-A3B (vLLM) + Qwen2.5-VL-32B (vision) + Lilith-L3.3-70B (Ollama) + Wan2.2 (video)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    OLLAMA_MODELS=/workspace/ollama_models \
    OLLAMA_HOST=0.0.0.0:11434 \
    MODEL_DIR=/workspace/models \
    VLLM_TP=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ffmpeg libsm6 libxext6 \
    libglib2.0-0 libgl1-mesa-glx build-essential openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python deps - core inference
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
    "auto-gptq"

# Python deps - Wan2.2 video generation
RUN pip install --no-cache-dir \
    "diffusers>=0.31.0" \
    "imageio" \
    "imageio-ffmpeg" \
    "av" \
    "decord" \
    "easydict" \
    "ftfy" \
    "timm" \
    "numpy<2.0"

WORKDIR /app
COPY gateway.py /app/gateway.py
COPY vl_server.py /app/vl_server.py
COPY wan_server.py /app/wan_server.py
COPY start.sh /start.sh

RUN chmod +x /start.sh

# Ports:
# 8000 = unified gateway (OpenAI-compatible)
# 8001 = vLLM (Qwen3-Next-80B-A3B)
# 8002 = vision server (Qwen2.5-VL-32B)
# 8003 = Wan2.2 video server
# 8888 = Jupyter Lab
# 11434 = Ollama (Lilith Whisper)
EXPOSE 8000 8001 8002 8003 8888 11434 22

VOLUME ["/workspace"]

CMD ["/start.sh"]
