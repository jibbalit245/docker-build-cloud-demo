#!/bin/bash
set -e

echo "================================================"
echo "  OMNI-STACK: Multi-Model AI Pod Starting..."
echo "================================================"

mkdir -p /workspace/hf_cache /workspace/ollama_models \
         /workspace/models/wan2.2 /workspace/datasets \
         /workspace/logs /workspace/webui

# ── SSH ────────────────────────────────────────────────
mkdir -p /run/sshd
if [ -n "${PUBLIC_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd -D &

# ── HuggingFace token ──────────────────────────────────
if [ -n "${HF_TOKEN}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
fi

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache

# ══════════════════════════════════════════════════════
# MODELS — pull from HuggingFace on first boot
# ══════════════════════════════════════════════════════
download_model() {
    local REPO=$1
    local NAME=$(basename $REPO)
    local DIR=/workspace/hf_cache/$NAME
    if [ ! -d "$DIR" ]; then
        echo "[MODELS] Downloading $REPO ..."
        huggingface-cli download "$REPO" \
            --local-dir "$DIR" \
            --local-dir-use-symlinks False \
            >> /workspace/logs/model_downloads.log 2>&1 \
            && echo "[MODELS] $NAME done." \
            || echo "[MODELS] WARNING: $NAME failed — check logs"
    else
        echo "[MODELS] $NAME already cached, skipping."
    fi
}

download_model "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated"
download_model "huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated"
download_model "Wan-AI/Wan2.2-T2V-14B"

# Lilith Whisper via Ollama
OLLAMA_MODELS=/workspace/ollama_models ollama serve > /workspace/logs/ollama.log 2>&1 &
for i in $(seq 1 20); do
    sleep 2
    curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && break
    echo "[OLLAMA] Waiting... ($i/20)"
done
if ! ollama list 2>/dev/null | grep -qi "lilith"; then
    echo "[OLLAMA] Pulling Lilith-Whisper-L3.3-70B Q5_K_M..."
    ollama pull hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF \
        >> /workspace/logs/ollama_pull.log 2>&1 \
        && echo "[OLLAMA] Lilith done." \
        || echo "[OLLAMA] WARNING: Lilith pull failed"
fi

# ══════════════════════════════════════════════════════
# DATASETS — pull from HuggingFace on first boot
# ══════════════════════════════════════════════════════
download_dataset() {
    local REPO=$1
    local DEST=$2
    if [ ! -d "/workspace/datasets/$DEST" ]; then
        echo "[DATA] Downloading $REPO ..."
        huggingface-cli download "$REPO" \
            --repo-type dataset \
            --local-dir "/workspace/datasets/$DEST" \
            --local-dir-use-symlinks False \
            >> /workspace/logs/dataset_downloads.log 2>&1 \
            && echo "[DATA] $DEST done." \
            || echo "[DATA] WARNING: $DEST failed — check logs"
    else
        echo "[DATA] $DEST already cached, skipping."
    fi
}

download_dataset "HuggingFaceM4/ChartQA"                              "ChartQA"
download_dataset "m-a-p/Code-Feedback"                                 "Code-Feedback"
download_dataset "zwhe99/DeepMath-103K"                                "DeepMath-103K"
download_dataset "HuggingFaceM4/FineVision"                            "FineVision-Cauldron"
download_dataset "BAAI/Infinity-Instruct"                              "Infinity-Instruct"
download_dataset "liuhaotian/LLaVA-Instruct-150K"                     "LLaVA-Instruct-150K"
download_dataset "nvidia/Llama-Nemotron-Post-Training-Dataset"         "Llama-Nemotron-PostTrain"
download_dataset "Cognitive-Lab/MCP-Flow"                              "MCP-Flow"
download_dataset "MCPBench/MCPToolBench-Plus"                          "MCPToolBenchPP"
download_dataset "teknium/OpenHermes-2.5"                              "OpenHermes-2.5"
download_dataset "nvidia/OpenMathInstruct-2"                           "OpenMathInstruct-2"
download_dataset "SWE-bench/SWE-bench"                                 "SWE-rebench-V2-PRs"
download_dataset "arcee-ai/The-Tome"                                   "The-Tome"
download_dataset "ToolBench/ToolBench"                                 "ToolBenchBackbone"
download_dataset "ToolBench/ToolBench"                                 "ToolBenchCaller"
download_dataset "ToolBench/ToolBench"                                 "ToolBenchG2"
download_dataset "ToolBench/ToolBench"                                 "ToolBenchG3"
download_dataset "ToolBench/ToolBench"                                 "ToolBenchPlanner"
download_dataset "openbmb/UltraFeedback"                               "UltraFeedback"
download_dataset "HuggingFaceM4/WebSight"                              "WebSight"
download_dataset "microsoft/orca-agentinstruct-1M-v1"                  "orca-agentinstruct"
download_dataset "huihui-ai/abliterated-distill-30k"                   "abliterated-distill-30k"
download_dataset "huihui-ai/abliterated_dataset"                       "abliterated_dataset"
download_dataset "Cognitive-Lab/MCP-SFT"                               "MCP_SFT"
download_dataset "Cognitive-Lab/phi3-mcp-dataset"                      "phi3_mcp_dataset"
download_dataset "Cognitive-Lab/math-for-mcp"                          "math-for-mcp"
download_dataset "Cognitive-Lab/mcp-clients"                           "mcp-clients"
download_dataset "Cognitive-Lab/video-mcp"                             "video-mcp"
download_dataset "Cognitive-Lab/discover-tools"                        "discover-tools"
download_dataset "gradio-agents-mcp-hackathon/certificates"            "gradio-agents-mcp-hackathon-certificates"

# ── vLLM (Qwen3-Next-80B-A3B) ─────────────────────────
echo "[SVC] Starting vLLM..."
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/hf_cache/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated \
    --port 8001 \
    --host 0.0.0.0 \
    --tensor-parallel-size ${VLLM_TP:-1} \
    --gpu-memory-utilization 0.45 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --served-model-name qwen3-80b \
    > /workspace/logs/vllm.log 2>&1 &

# ── Vision Server (Qwen2.5-VL-32B) ─────────────────────
echo "[SVC] Starting Qwen2.5-VL-32B vision server..."
python3 /app/vl_server.py \
    --model /workspace/hf_cache/Qwen2.5-VL-32B-Instruct-abliterated \
    --port 8002 \
    --gpu-frac 0.35 \
    > /workspace/logs/vl_server.log 2>&1 &

# ── Wan2.2 Video Server ─────────────────────────────────
echo "[SVC] Starting Wan2.2..."
python3 /app/wan_server.py \
    --model-dir /workspace/hf_cache/Wan2.2-T2V-14B \
    --port 8003 \
    > /workspace/logs/wan_server.log 2>&1 &

# ── Unified Gateway ─────────────────────────────────────
echo "[SVC] Starting gateway..."
sleep 8
uvicorn gateway:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    > /workspace/logs/gateway.log 2>&1 &

# ── Open WebUI ──────────────────────────────────────────
echo "[SVC] Starting Open WebUI..."
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
echo "  Downloads running in background."
echo "  Logs: /workspace/logs/"
echo "  Open WebUI  -> :3000"
echo "  Gateway     -> :8000/health"
echo "================================================"

wait
