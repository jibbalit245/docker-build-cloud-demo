#!/bin/bash
set -e
set -o pipefail

echo "================================================"
echo "  OMNI-STACK TRAINING POD: Starting..."
echo "================================================"

WORKFLOW_LOG=/workspace/logs/workflow.log

# ── Configurable defaults ──────────────────────────────
# Override these at pod launch to train a different model.
TRAIN_BASE_MODEL="${TRAIN_BASE_MODEL:-huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-omnistack}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

log_phase() {
    local MSG=$1
    echo "[WORKFLOW] $MSG" | tee -a "$WORKFLOW_LOG"
}

mkdir -p /workspace/hf_cache /workspace/datasets \
         /workspace/logs /workspace/notebooks \
         /workspace/output/checkpoints /workspace/output/final_model
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
    log_phase "phase=auth status=hf_token_missing — gated models may fail"
fi

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache

# ── Weights & Biases (optional) ────────────────────────
if [ -n "${WANDB_API_KEY}" ]; then
    wandb login "${WANDB_API_KEY}" 2>/dev/null || true
    log_phase "phase=auth status=wandb_login_attempted"
fi

# ══════════════════════════════════════════════════════
# BASE MODEL — download once, reuse on subsequent boots
# ══════════════════════════════════════════════════════
log_phase "phase=model status=download_started model=$TRAIN_BASE_MODEL"

MODEL_NAME=$(basename "$TRAIN_BASE_MODEL")
MODEL_DIR=/workspace/hf_cache/$MODEL_NAME

if [ ! -d "$MODEL_DIR" ]; then
    echo "[MODEL] Downloading $TRAIN_BASE_MODEL ..."
    huggingface-cli download "$TRAIN_BASE_MODEL" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False \
        >> /workspace/logs/model_downloads.log 2>&1 \
        && echo "[MODEL] $MODEL_NAME downloaded." \
        || echo "[MODEL] WARNING: $MODEL_NAME download failed — check /workspace/logs/model_downloads.log"
else
    echo "[MODEL] $MODEL_NAME already cached, skipping."
fi

log_phase "phase=model status=download_complete path=$MODEL_DIR"

# ══════════════════════════════════════════════════════
# DATASETS — download all training datasets
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

# ── Copy starter notebook if not already in /workspace ─
if [ -f "/app/notebooks/quickstart_train.ipynb" ] && \
   [ ! -f "/workspace/notebooks/quickstart_train.ipynb" ]; then
    cp /app/notebooks/quickstart_train.ipynb /workspace/notebooks/quickstart_train.ipynb
fi

# ══════════════════════════════════════════════════════
# JUPYTERLAB — interactive training environment
# ══════════════════════════════════════════════════════
log_phase "phase=jupyterlab status=starting port=$JUPYTER_PORT"

echo "[SVC] Starting JupyterLab on port $JUPYTER_PORT ..."
jupyter lab \
    --ip=0.0.0.0 \
    --port="$JUPYTER_PORT" \
    --no-browser \
    --allow-root \
    --notebook-dir=/workspace \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --ServerApp.password="" \
    --ServerApp.allow_origin="*" \
    --ServerApp.disable_check_xsrf=True \
    > /workspace/logs/jupyter.log 2>&1 &
JUPYTER_PID=$!

log_phase "phase=jupyterlab status=started pid=$JUPYTER_PID"

echo ""
echo "================================================"
echo "  OMNI-STACK TRAINING POD READY"
echo "  Base model : $MODEL_DIR"
echo "  Datasets   : /workspace/datasets/"
echo "  Output     : /workspace/output/"
echo "  JupyterLab : http://0.0.0.0:$JUPYTER_PORT"
echo "  Token      : $JUPYTER_TOKEN"
echo "  Logs       : /workspace/logs/"
echo "================================================"

log_phase "phase=workflow status=ready"

# Keep container alive; exit if JupyterLab dies
wait "$JUPYTER_PID"
EXIT_CODE=$?
echo "[WORKFLOW] JupyterLab exited (code=$EXIT_CODE)."
exit "$EXIT_CODE"
