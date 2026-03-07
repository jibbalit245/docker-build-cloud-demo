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
echo "  Open WebUI           -> port 3000"
echo "================================================"

mkdir -p /workspace/hf_cache /workspace/ollama_models \
         /workspace/models/wan2.2 /workspace/logs \
         /workspace/webui

# ── SSH ────────────────────────────────────────────────
mkdir -p /run/sshd
if [ -n "${PUBLIC_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd -D &

# ── 1. Ollama (Lilith Whisper L3.3-70B) ────────────────
echo "[1/6] Starting Ollama..."
OLLAMA_MODELS=/workspace/ollama_models ollama serve \
    > /workspace/logs/ollama.log 2>&1 &

for i in $(seq 1 20); do
    sleep 2
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[1/6] Ollama ready."
        break
    fi
    echo "[1/6] Waiting for Ollama... ($i/20)"
done

if ! ollama list 2>/dev/null | grep -qi "lilith"; then
    echo "[1/6] Pulling Lilith-Whisper-L3.3-70B Q5_K_M..."
    ollama pull hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF \
        >> /workspace/logs/ollama_pull.log 2>&1 \
        || echo "WARN: Pull failed - run manually later"
fi

# ── 2. vLLM (Qwen3-Next-80B-A3B MoE) ─────────────────
echo "[2/6] Starting vLLM for Qwen3-Next-80B-A3B..."
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
echo "[3/6] Starting Qwen2.5-VL-32B vision server..."
python3 /app/vl_server.py \
    --model "${VL_MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct-GPTQ-Int4}" \
    --port 8002 \
    --gpu-frac 0.35 \
    > /workspace/logs/vl_server.log 2>&1 &

# ── 4. Wan2.2 Video Server ─────────────────────────────
echo "[4/6] Starting Wan2.2 video server..."
python3 /app/wan_server.py \
    --model-dir /workspace/models/wan2.2 \
    --port 8003 \
    > /workspace/logs/wan_server.log 2>&1 &

# ── 5. Unified Gateway ─────────────────────────────────
echo "[5/6] Starting unified gateway on :8000..."
sleep 8
uvicorn gateway:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    > /workspace/logs/gateway.log 2>&1 &

# ── 6. Open WebUI ──────────────────────────────────────
echo "[6/6] Starting Open WebUI on :3000..."
sleep 5
DATA_DIR=/workspace/webui \
WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-omnistack}" \
WEBUI_AUTH="${WEBUI_AUTH:-False}" \
OPENAI_API_BASE_URL="http://localhost:8000/v1" \
OPENAI_API_KEY="none" \
open-webui serve \
    --host 0.0.0.0 \
    --port 3000 \
    > /workspace/logs/webui.log 2>&1 &

echo ""
echo "================================================"
echo "  All services launched!"
echo "  Logs: /workspace/logs/"
echo ""
echo "  Open WebUI  -> http://0.0.0.0:3000  (USE THIS)"
echo "  Gateway     -> http://0.0.0.0:8000/health"
echo "  Models      -> http://0.0.0.0:8000/v1/models"
echo "================================================"

wait
