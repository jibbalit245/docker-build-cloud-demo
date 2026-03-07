#!/bin/bash
set -e

echo "================================================"
echo "  OMNI-STACK: Multi-Model AI Pod Starting..."
echo "================================================"
echo "  Qwen3-Next-80B-A3B   -> port 8001 (vLLM)"
echo "  Qwen2.5-VL-32B       -> port 8002 (vision)"
echo "  Lilith-L3.3-70B      -> port 11434 (Ollama)"
echo "  Wan2.2               -> port 8003 (video)"
echo "  Unified Gateway      -> port 8000"
echo "================================================"

# Create dirs on persistent network volume
mkdir -p /workspace/hf_cache
mkdir -p /workspace/ollama_models
mkdir -p /workspace/models/wan2.2
mkdir -p /workspace/logs

# ── SSH ────────────────────────────────────────────────
mkdir -p /run/sshd
if [ -n "${PUBLIC_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd -D &

# ── 1. Ollama (Lilith Whisper L3.3-70B) ────────────────
echo "[1/5] Starting Ollama..."
OLLAMA_MODELS=/workspace/ollama_models ollama serve > /workspace/logs/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in $(seq 1 20); do
    sleep 2
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[1/5] Ollama ready."
        break
    fi
    echo "[1/5] Waiting for Ollama... ($i/20)"
done

# Pull Lilith if not already cached
if ! ollama list 2>/dev/null | grep -qi "lilith"; then
    echo "[1/5] Pulling Lilith-Whisper-L3.3-70B Q5_K_M..."
    ollama pull hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF 2>&1 \
        | tee -a /workspace/logs/ollama_pull.log \
        || echo "WARN: Ollama pull failed - run manually: ollama pull hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF"
fi

# ── 2. vLLM (Qwen3-Next-80B-A3B MoE) ─────────────────
echo "[2/5] Starting vLLM for Qwen3-Next-80B-A3B..."
python3 -m vllm.entrypoints.openai.api_server \
    --model huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated \
    --port 8001 \
    --host 0.0.0.0 \
    --tensor-parallel-size ${VLLM_TP:-1} \
    --gpu-memory-utilization 0.45 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --download-dir /workspace/hf_cache \
    --served-model-name qwen3-80b \
    > /workspace/logs/vllm.log 2>&1 &

# ── 3. Vision Server (Qwen2.5-VL-32B) ─────────────────
echo "[3/5] Starting Qwen2.5-VL-32B vision server..."
# NOTE: litmudoc MLX version is Apple Silicon only - not CUDA compatible.
# Using Qwen/Qwen2.5-VL-32B-Instruct-GPTQ-Int4 (NVIDIA-compatible).
# Override with env var: VL_MODEL_ID=<your-model>
python3 /app/vl_server.py \
    --model "${VL_MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct-GPTQ-Int4}" \
    --port 8002 \
    --gpu-frac 0.35 \
    > /workspace/logs/vl_server.log 2>&1 &

# ── 4. Wan2.2 Video Server ─────────────────────────────
echo "[4/5] Starting Wan2.2 video generation server..."
python3 /app/wan_server.py \
    --model-dir /workspace/models/wan2.2 \
    --port 8003 \
    > /workspace/logs/wan_server.log 2>&1 &

# ── 5. Unified Gateway ─────────────────────────────────
echo "[5/5] Starting unified gateway on :8000..."
sleep 8
uvicorn gateway:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    > /workspace/logs/gateway.log 2>&1 &

# ── Jupyter Lab (optional) ────────────────────────────
if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    echo "[+] Starting Jupyter Lab on :8888..."
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --NotebookApp.token="${JUPYTER_PASSWORD:-omnistack}" \
        --notebook-dir=/workspace \
        > /workspace/logs/jupyter.log 2>&1 &
fi

echo ""
echo "================================================"
echo "  All services launched. Logs at /workspace/logs/"
echo ""
echo "  GET  http://<pod>:8000/health       <- status"
echo "  GET  http://<pod>:8000/v1/models    <- model list"
echo "  POST http://<pod>:8000/v1/chat/completions"
echo "  POST http://<pod>:8000/v1/video/generate"
echo "================================================"

# Keep container alive
wait
