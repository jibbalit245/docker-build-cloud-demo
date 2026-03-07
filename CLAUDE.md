# CLAUDE.md — Omni-Stack Repo Context

This file gives any Claude instance immediate context about this repository.
Read this before touching any file.

---

## What This Repo Is

A multi-model AI inference stack packaged as a single Docker image (`westllc/omni-stack:latest`)
and deployed as a RunPod GPU pod. Six services run together inside one container.

---

## Services

| Service        | Port  | Backend        | Model                                              |
|----------------|-------|----------------|----------------------------------------------------|
| Open WebUI     | 3000  | open-webui     | Frontend UI for all models                         |
| Gateway        | 8000  | gateway.py     | FastAPI router — single entry point for all models |
| vLLM           | 8001  | vLLM 0.6.3     | huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated |
| Vision Server  | 8002  | vl_server.py   | Qwen/Qwen2.5-VL-32B-Instruct-GPTQ-Int4            |
| Wan2.2 Video   | 8003  | wan_server.py  | Wan-AI/Wan2.2-T2V-14B                              |
| Ollama         | 11434 | Ollama         | Liliths-Whisper-L3.3-70b-0.1.Q5_K_M GGUF          |

---

## Files

| File                | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `Dockerfile`        | Base image + all framework installs. ARG/ENV HF_TOKEN for HF auth.    |
| `start.sh`          | Container entrypoint. Downloads models + datasets, then starts services.|
| `gateway.py`        | Unified FastAPI gateway on :8000. Routes by `model` field in request.  |
| `vl_server.py`      | Vision inference server for Qwen2.5-VL-32B on :8002.                  |
| `wan_server.py`     | Video generation server for Wan2.2-T2V-14B on :8003.                  |
| `runpod_handler.py` | RunPod serverless handler (alternative async invocation mode).         |
| `.github/workflows/ci.yml` | GitHub Actions CI — builds and pushes image to Docker Hub.    |
| `.runpod/`          | RunPod template and endpoint configuration.                            |

---

## Boot Sequence (start.sh)

On first boot the container:
1. Logs into HuggingFace using `HF_TOKEN` env var
2. Downloads 4 models to `/workspace/hf_cache/` (skipped if already present)
3. Downloads 31 datasets to `/workspace/datasets/` (skipped if already present)
4. Starts SSH, Ollama, vLLM, vision server, Wan2.2 server, gateway, Open WebUI

Downloads run in the background — services start immediately. On subsequent boots
with the network volume attached, steps 2–3 are instant skips.

---

## Models (HuggingFace)

```
huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated  → /workspace/hf_cache/
Qwen/Qwen2.5-VL-32B-Instruct-GPTQ-Int4                    → /workspace/hf_cache/
Wan-AI/Wan2.2-T2V-14B                                      → /workspace/hf_cache/
hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF → pulled via Ollama
```

---

## Datasets (HuggingFace, all to /workspace/datasets/)

ChartQA, Code-Feedback, DeepMath-103K, FineVision-Cauldron, Infinity-Instruct,
LLaVA-Instruct-150K, Llama-Nemotron-PostTrain, MCP-Flow, MCPToolBenchPP, MCP_SFT,
OpenHermes-2.5, OpenMathInstruct-2, SWE-rebench-V2-PRs, The-Tome, ToolBenchBackbone,
ToolBenchCaller, ToolBenchG2, ToolBenchG3, ToolBenchPlanner, UltraFeedback, WebSight,
abliterated-distill-30k, abliterated_dataset, discover-tools,
gradio-agents-mcp-hackathon-certificates, math-for-mcp, mcp-clients, orca-agentinstruct,
phi3_mcp_dataset, video-mcp

---

## Gateway API

```
GET  :8000/health
GET  :8000/v1/models
POST :8000/v1/chat/completions   model="qwen3-80b" | "vision" | "lilith" | "wan2.2"
POST :8000/v1/video/generate
POST :8000/v1/images/analyze
```

---

## CI/CD

- **Trigger**: push to `main`
- **Runner**: `ubuntu-latest` (standard GitHub-hosted)
- **Builder**: standard `docker/setup-buildx-action@v3` — no Docker Build Cloud
- **Pushes to**: `westllc/omni-stack:latest` on Docker Hub
- **Auth**: GitHub secret `DOCKER_PAT` (OAT) + variable `DOCKER_USER=westllc`
- **HF token**: GitHub secret `HF_TOKEN`, passed as build-arg and baked into image as ENV

### DO NOT use Docker Build Cloud for this repo
The Build Cloud endpoint `jibbalit/omni-agent` belongs to a different org than `westllc`
and will always return 403. Standard buildx works fine.

---

## RunPod Deployment

- **Image**: `westllc/omni-stack:latest` (private — requires registry auth on RunPod)
- **Network volume**: 1500GB, mount at `/workspace` — all models and datasets live here
- **Required env vars**:
  - `HF_TOKEN` — HuggingFace auth for gated models and datasets
  - `HF_HOME=/workspace/hf_cache`
  - `OLLAMA_MODELS=/workspace/ollama_models`
  - `OLLAMA_HOST=0.0.0.0:11434`
  - `WEBUI_AUTH=False`
- **Ports**: 3000/http, 8000/http, 8001/http, 8002/http, 8003/http, 11434/http, 22/tcp
- **Recommended GPU**: 2× A100 80GB or 1× H100 SXM (~140GB VRAM needed for all 4 models)
- **All services must bind 0.0.0.0** — RunPod proxy cannot reach localhost/127.0.0.1

---

## Critical Rules

1. **Never bake models into the Docker image** — weights live on the network volume only
2. **Never use Build Cloud** for this repo — use standard buildx
3. **Always free disk space** before the build step or the GitHub runner runs out
4. **All services bind 0.0.0.0** not localhost
5. **HF_TOKEN must be set** at runtime for gated model/dataset downloads to succeed
