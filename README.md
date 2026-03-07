# omni-stack
Docker Build Cloud CI — builds jibbalit/omni-stack:latest on every push to main.

## Models
- Qwen3-Next-80B-A3B (vLLM)
- Qwen2.5-VL-32B (vision)
- Lilith-L3.3-70B GGUF (Ollama)
- Wan2.2 T2V-14B (video)

## FastAPI pipeline flow (gateway interconnection)
The FastAPI gateway (`gateway.py`, port `8000`) is the pipeline router between clients and model backends.

```mermaid
flowchart TD
    A[Client / Open WebUI / RunPod job] --> N[NGINX edge<br/>TLS, rate-limit, optional auth]
    N --> B[FastAPI Router :8000]

    B -->|model=qwen3-80b| C[Qwen3-80B on vLLM :8001]
    B -->|model=vision / qwen2.5-vl| D[Qwen2.5-VL-32B (Transformers) :8002]
    B -->|model=lilith / whisper| E[Lilith-Whisper on Ollama :11434]
    B -->|model=wan2.2 / video| F[Wan2.2-T2V (Transformers) :8003]

    F --> P[Optional post-processing]
    C --> R[Unified response path]
    D --> R
    E --> R
    P --> R
```
