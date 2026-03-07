#!/bin/bash
set -e
set -o pipefail

echo "================================================"
echo "  OMNI-STACK: Multi-Model AI Pod Starting..."
echo "================================================"

WORKFLOW_LOG=/workspace/logs/workflow.log
SERVICE_MAP=/workspace/logs/service_map.json
OLLAMA_MODEL_NAME_DEFAULT="hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.2a.i1-Q4_K_M-GGUF:latest"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME:-$OLLAMA_MODEL_NAME_DEFAULT}"

log_phase() {
        local MSG=$1
        echo "[WORKFLOW] $MSG" | tee -a "$WORKFLOW_LOG"
}

write_service_map() {
        cat > "$SERVICE_MAP" <<EOF
{
    "gateway": "http://0.0.0.0:8000",
    "routes": {
        "qwen3-80b": {
            "backend": "vllm",
            "url": "http://127.0.0.1:8001",
            "model": "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated"
        },
        "vision": {
            "backend": "vl_server",
            "url": "http://127.0.0.1:8002",
            "model": "huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated"
        },
        "wan2.2": {
            "backend": "wan_server",
            "url": "http://127.0.0.1:8003",
            "model": "Wan-AI/Wan2.2-T2V-14B"
        },
        "lilith": {
            "backend": "ollama",
            "url": "http://127.0.0.1:11434",
            "model": "$OLLAMA_MODEL_NAME"
        }
    }
}
EOF
}

mkdir -p /workspace/hf_cache /workspace/ollama_models \
         /workspace/models/wan2.2 /workspace/datasets \
         /workspace/logs /workspace/webui
touch "$WORKFLOW_LOG"

log_phase "phase=bootstrap status=started"

# ── SSH ────────────────────────────────────────────────
mkdir -p /run/sshd
if [ -n "${PUBLIC_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd -D &
log_phase "phase=bootstrap status=ssh_started"

# ── HuggingFace token ──────────────────────────────────
if [ -n "${HF_TOKEN}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
    log_phase "phase=auth status=hf_login_attempted"
else
    log_phase "phase=auth status=hf_token_missing"
fi

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:8001}"
export VL_URL="${VL_URL:-http://127.0.0.1:8002}"
export WAN_URL="${WAN_URL:-http://127.0.0.1:8003}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
export VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-}"
export VL_CUDA_DEVICES="${VL_CUDA_DEVICES:-}"
export WAN_CUDA_DEVICES="${WAN_CUDA_DEVICES:-}"

# ══════════════════════════════════════════════════════
# MODELS — pull from HuggingFace on first boot
# ══════════════════════════════════════════════════════
log_phase "phase=models status=download_started"

run_with_optional_cuda() {
    local CUDA_DEVICES=$1
    shift
    if [ -n "$CUDA_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$@"
    else
        "$@"
    fi
}

download_model() {
    local REPO=$1
    local MARKER_FILE=$2
    local NAME=$(basename $REPO)
    local DIR=/workspace/hf_cache/$NAME
    local MARKER_PATH="$DIR/$MARKER_FILE"

    if [ -d "$DIR" ] && [ ! -f "$MARKER_PATH" ]; then
        echo "[MODELS] $NAME cache exists but is incomplete (missing $MARKER_FILE). Re-downloading..."
        rm -rf "$DIR"
    fi

    if [ ! -d "$DIR" ]; then
        echo "[MODELS] Downloading $REPO ..."
        if huggingface-cli download "$REPO" \
            --local-dir "$DIR" \
            --local-dir-use-symlinks False \
            >> /workspace/logs/model_downloads.log 2>&1; then
            echo "[MODELS] $NAME done."
        else
            echo "[MODELS] FATAL: $NAME failed — check /workspace/logs/model_downloads.log"
            return 1
        fi
    else
        echo "[MODELS] $NAME already cached and complete, skipping."
    fi

    if [ ! -f "$MARKER_PATH" ]; then
        echo "[MODELS] FATAL: $NAME missing required file $MARKER_FILE after download"
        return 1
    fi
}

download_model "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated" "config.json"
download_model "huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated" "config.json"
download_model "Wan-AI/Wan2.2-T2V-14B" "model_index.json"
log_phase "phase=models status=download_completed"

# Lilith Whisper via Ollama
OLLAMA_MODELS=/workspace/ollama_models ollama serve > /workspace/logs/ollama.log 2>&1 &
log_phase "phase=models status=ollama_started_for_pull"
for i in $(seq 1 20); do
    sleep 2
    curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && break
    echo "[OLLAMA] Waiting... ($i/20)"
done
if ! ollama list 2>/dev/null | grep -qi "lilith"; then
    echo "[OLLAMA] Pulling $OLLAMA_MODEL_NAME ..."
    ollama pull "$OLLAMA_MODEL_NAME" \
        >> /workspace/logs/ollama_pull.log 2>&1 \
        && echo "[OLLAMA] Lilith done." \
        || {
            echo "[OLLAMA] FATAL: Lilith pull failed"
            exit 1
        }
fi
log_phase "phase=models status=ollama_pull_completed"

# ══════════════════════════════════════════════════════
# DATASETS — pull from HuggingFace on first boot
# ══════════════════════════════════════════════════════
log_phase "phase=datasets status=download_started"

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
log_phase "phase=datasets status=download_completed"

log_phase "phase=workflow status=downloads_complete_services_pending"

wait_for_url() {
    local NAME=$1
    local URL=$2
    local MAX_TRIES=${3:-60}
    local SLEEP_SECS=${4:-5}

    for i in $(seq 1 "$MAX_TRIES"); do
        if curl -sf "$URL" > /dev/null 2>&1; then
            echo "[READY] $NAME is ready."
            return 0
        fi
        echo "[READY] Waiting for $NAME... ($i/$MAX_TRIES)"
        sleep "$SLEEP_SECS"
    done

    echo "[READY] WARNING: $NAME did not become ready in time: $URL"
    return 1
}

log_phase "phase=workflow status=services_starting"

# ── vLLM-led model server orchestration (non-Ollama) ──
# vLLM starts first, then vision and Wan start in sequence.

# vLLM (Qwen3-Next-80B-A3B)
echo "[SVC] Starting vLLM..."
run_with_optional_cuda "$VLLM_CUDA_DEVICES" python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/hf_cache/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated \
    --port 8001 \
    --host 0.0.0.0 \
    --tensor-parallel-size "${VLLM_TP:-${TENSOR_PARALLEL:-1}}" \
    --gpu-memory-utilization 0.45 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --served-model-name qwen3-80b \
    > /workspace/logs/vllm.log 2>&1 &
VLLM_PID=$!
wait_for_url "vLLM" "$VLLM_URL/health" 180 5

# Vision Server (Qwen2.5-VL-32B)
echo "[SVC] Starting Qwen2.5-VL-32B vision server..."
run_with_optional_cuda "$VL_CUDA_DEVICES" python3 /app/vl_server.py \
    --model /workspace/hf_cache/Qwen2.5-VL-32B-Instruct-abliterated \
    --port 8002 \
    --gpu-frac 0.35 \
    > /workspace/logs/vl_server.log 2>&1 &
VL_PID=$!
wait_for_url "Vision" "$VL_URL/health" 180 5

# Wan2.2 Video Server
echo "[SVC] Starting Wan2.2..."
run_with_optional_cuda "$WAN_CUDA_DEVICES" python3 /app/wan_server.py \
    --model-dir /workspace/hf_cache \
    --port 8003 \
    > /workspace/logs/wan_server.log 2>&1 &
WAN_PID=$!
wait_for_url "Wan2.2" "$WAN_URL/health" 180 5

# ── Unified Gateway ─────────────────────────────────────
wait_for_url "Ollama" "$OLLAMA_URL/api/tags" 60 2

# Ensure the heavy model servers are truly loaded, not just listening.
if ! curl -sf "$VL_URL/health" | grep -q '"loaded":true'; then
    echo "[READY] FATAL: Vision server started but model not loaded"
    exit 1
fi
if ! curl -sf "$WAN_URL/health" | grep -q '"model_loaded":true'; then
    echo "[READY] FATAL: Wan2.2 server started but model not loaded"
    exit 1
fi
log_phase "phase=services status=backend_health_checks_completed"

echo "[SVC] Starting gateway..."
uvicorn gateway:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    > /workspace/logs/gateway.log 2>&1 &
GATEWAY_PID=$!

# ── Open WebUI ──────────────────────────────────────────
wait_for_url "Gateway" "http://127.0.0.1:8000/health" 60 2
log_phase "phase=services status=gateway_ready"

echo "[SVC] Starting Open WebUI..."
DATA_DIR=/workspace/webui \
WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-omnistack}" \
WEBUI_AUTH="${WEBUI_AUTH:-False}" \
ENABLE_OPENAI_API="${ENABLE_OPENAI_API:-True}" \
ENABLE_OLLAMA_API="${ENABLE_OLLAMA_API:-False}" \
OPENAI_API_BASE_URL="http://localhost:8000/v1" \
OPENAI_API_BASE_URLS="http://localhost:8000/v1" \
OPENAI_API_KEY="none" \
open-webui serve \
    --host 0.0.0.0 \
    --port 3000 \
    > /workspace/logs/webui.log 2>&1 &
WEBUI_PID=$!
log_phase "phase=services status=webui_started"

write_service_map
log_phase "phase=workflow status=service_map_written path=$SERVICE_MAP"

# Final gateway verification pass for route visibility.
if curl -sf http://localhost:8000/health > /workspace/logs/gateway_health_snapshot.json; then
    log_phase "phase=verify status=gateway_health_snapshot_written"
else
    log_phase "phase=verify status=gateway_health_snapshot_failed"
fi

if curl -sf http://localhost:8000/v1/models > /workspace/logs/gateway_models_snapshot.json; then
    log_phase "phase=verify status=gateway_models_snapshot_written"
else
    log_phase "phase=verify status=gateway_models_snapshot_failed"
fi

log_phase "phase=workflow status=ready"

echo ""
echo "================================================"
echo "  All services launched!"
echo "  Downloads running in background."
echo "  Logs: /workspace/logs/"
echo "  Open WebUI  -> :3000"
echo "  Gateway     -> :8000/health"
echo "  Service map -> /workspace/logs/service_map.json"
echo "  Workflow    -> /workspace/logs/workflow.log"
echo "================================================"

# Keep container alive while core services run, and fail fast if any exits.
SERVICE_PIDS=("$VLLM_PID" "$VL_PID" "$WAN_PID" "$GATEWAY_PID" "$WEBUI_PID")
wait -n "${SERVICE_PIDS[@]}"
EXIT_CODE=$?
echo "[WORKFLOW] A core service exited unexpectedly (code=$EXIT_CODE)."
exit "$EXIT_CODE"
