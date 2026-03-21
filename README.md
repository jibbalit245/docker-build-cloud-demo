# omni-stack
Docker Build Cloud CI — builds jibbalit/omni-stack:latest on every push to main.

## Models
- Qwen3-Next-80B-A3B (vLLM)
- Qwen2.5-VL-32B (vision)
- Lilith-L3.3-70B GGUF (Ollama)
- Wan2.2 T2V-14B (video)

## Training Pod

A separate training image (`westllc/omni-stack-train:latest`) is available for online fine-tuning.
It pre-downloads the base model and all 30 datasets on first boot, then launches **JupyterLab** on port **8888**.

### Quick start on RunPod

1. Create a new pod using the template in `.runpod/train_hub.json`
   (image: `westllc/omni-stack-train:latest`, network volume ≥ 1500 GB at `/workspace`).
2. Set environment variables:
   - `HF_TOKEN` — your HuggingFace token (required for gated models)
   - `TRAIN_BASE_MODEL` — model to fine-tune (default: Qwen3-80B)
   - `TRAIN_DATASET` — dataset folder under `/workspace/datasets/` (default: `OpenHermes-2.5`)
   - `JUPYTER_TOKEN` — JupyterLab access token (default: `omnistack`)
3. Open `http://<pod-ip>:8888?token=omnistack` and run `notebooks/quickstart_train.ipynb`.

### Supported training methods

| Method | Description | Key env vars |
|--------|-------------|-------------|
| `qlora` | QLoRA supervised fine-tuning via Unsloth + TRL SFTTrainer | `LORA_RANK`, `LORA_ALPHA` |
| `sft`   | Full supervised fine-tuning (no LoRA) | `LORA_RANK=0` |
| `dpo`   | Direct Preference Optimisation (requires chosen/rejected pairs) | — |
| `grpo`  | Group Relative Policy Optimisation (verifiable rewards) | — |

### Output paths

| Path | Contents |
|------|----------|
| `/workspace/output/lora_adapter/` | Saved LoRA adapter weights |
| `/workspace/output/final_model/`  | Merged full-precision model (when `MERGE_AFTER_TRAIN=true`) |
| `/workspace/output/checkpoints/`  | Per-epoch checkpoints |
| `/workspace/logs/`                | All service and download logs |
