FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="omni-stack"
LABEL description="Unified AI stack: Qwen3-Next-80B-A3B + Qwen2.5-VL-32B + Lilith-L3.3-70B + Wan2.2 + Open WebUI"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    OLLAMA_MODELS=/workspace/ollama_models \
    OLLAMA_HOST=0.0.0.0:11434 \
    MODEL_DIR=/workspace/models \
    VLLM_TP=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ffmpeg libsm6 libxext6 \
    libglib2.0-0 libgl1-mesa-glx build-essential \
    openssh-server zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

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
    "runpod"

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

RUN pip install --no-cache-dir open-webui

WORKDIR /app
COPY gateway.py /app/gateway.py
COPY vl_server.py /app/vl_server.py
COPY wan_server.py /app/wan_server.py
COPY runpod_handler.py /app/runpod_handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 3000 8000 8001 8002 8003 8888 11434 22

VOLUME ["/workspace"]

# Default: pod mode (supervisord via start.sh)
# Override CMD to run serverless: CMD ["python3", "-u", "/app/runpod_handler.py"]
CMD ["/start.sh"]
