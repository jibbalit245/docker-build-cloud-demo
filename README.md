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
    A[Client / Open WebUI / RunPod job] --> B[FastAPI Gateway :8000]

    B -->|model=qwen3-80b| C[vLLM :8001]
    B -->|model=vision| D[Qwen2.5-VL server :8002]
    B -->|model=wan2.2| E[Wan2.2 server :8003]
    B -->|model=lilith/whisper| F[Ollama :11434]

    C --> G[Response]
    D --> G
    E --> G
    F --> G
```
