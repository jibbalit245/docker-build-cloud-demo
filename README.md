# omni-stack

A multi-model AI inference stack packaged as a single Docker image and deployed as a RunPod GPU pod.
Six services run together inside one container, all exposed through a unified OpenAI-compatible gateway.

## Services

| Service        | Port  | Model |
|----------------|-------|-------|
| Open WebUI     | 3000  | Frontend UI for all models |
| Gateway        | 8000  | FastAPI router — single entry point |
| vLLM           | 8001  | huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated |
| Vision Server  | 8002  | huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated |
| Wan2.2 Video   | 8003  | Wan-AI/Wan2.2-T2V-14B |
| Ollama         | 11434 | Liliths-Whisper-L3.3-70B GGUF |

## Gateway API

```
GET  :8000/health
GET  :8000/v1/models
POST :8000/v1/chat/completions   model="qwen3-80b" | "vision" | "lilith" | "wan2.2"
POST :8000/v1/video/generate
POST :8000/v1/images/analyze
```

### Chat completions example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-80b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## RunPod Deployment

**Image:** `westllc/omni-stack:latest`

**Required environment variables:**

| Variable | Value |
|----------|-------|
| `HF_TOKEN` | HuggingFace access token (for gated models) |
| `HF_HOME` | `/workspace/hf_cache` |
| `OLLAMA_MODELS` | `/workspace/ollama_models` |
| `OLLAMA_HOST` | `0.0.0.0:11434` |

**Recommended GPU:** 2× A100 80GB or 1× H100 SXM (~140 GB VRAM)

**Network volume:** 1500 GB mounted at `/workspace` — all model weights live here, not in the image.

## CI/CD

Pushing to `main` triggers `.github/workflows/ci.yml`, which builds and pushes
`westllc/omni-stack:latest` to Docker Hub using standard `docker/setup-buildx-action`.

Requires GitHub secrets/variables:
- `DOCKER_PAT` — Docker Hub OAT
- `DOCKER_USER` — `westllc`

## Smoke test

See [RUNPOD_SMOKE_TEST.md](RUNPOD_SMOKE_TEST.md) for a 5-minute end-to-end validation checklist.

