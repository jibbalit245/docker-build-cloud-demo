"""
Omni-Stack Unified Gateway
OpenAI-compatible API that routes requests to the right model by name.

Models:
  qwen3-80b / qwen3 / text    -> vLLM  :8001
  qwen2.5-vl / vision / vl    -> VL    :8002
  wan2.2 / video / wan        -> Wan   :8003
  lilith / llama / l3.3       -> Ollama:11434
"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

app = FastAPI(title="Omni-Stack Gateway", version="1.0.0")

SERVICES = {
    "qwen3-80b":   "http://localhost:8001",
    "qwen3":       "http://localhost:8001",
    "text":        "http://localhost:8001",
    "qwen2.5-vl":  "http://localhost:8002",
    "vision":      "http://localhost:8002",
    "vl":          "http://localhost:8002",
    "wan2.2":      "http://localhost:8003",
    "video":       "http://localhost:8003",
    "wan":         "http://localhost:8003",
    "lilith":      "http://localhost:11434",
    "llama":       "http://localhost:11434",
    "l3.3":        "http://localhost:11434",
    "whisper":     "http://localhost:11434",
}
DEFAULT_TEXT = "http://localhost:8001"

OLLAMA_MODEL_NAME = "lilith-whisper:latest"


def resolve_service(model: str) -> str:
    if not model:
        return DEFAULT_TEXT
    m = model.lower()
    for key, url in SERVICES.items():
        if key in m:
            return url
    return DEFAULT_TEXT


@app.get("/health")
async def health():
    statuses = {}
    checks = {
        "vllm":   "http://localhost:8001/health",
        "vl":     "http://localhost:8002/health",
        "wan":    "http://localhost:8003/health",
        "ollama": "http://localhost:11434/api/tags",
    }
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in checks.items():
            try:
                r = await client.get(url)
                statuses[name] = "ok" if r.status_code < 400 else f"error_{r.status_code}"
            except Exception:
                statuses[name] = "down"
    overall = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "gateway": "ok", "services": statuses}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "qwen3-80b",        "object": "model", "type": "text",   "backend": "vllm:8001"},
            {"id": "qwen2.5-vl-32b",   "object": "model", "type": "vision", "backend": "vl_server:8002"},
            {"id": "lilith-l3.3-70b",  "object": "model", "type": "text",   "backend": "ollama:11434"},
            {"id": "wan2.2",           "object": "model", "type": "video",  "backend": "wan_server:8003"},
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    target = resolve_service(model)
    if target == "http://localhost:11434":
        return await proxy_ollama_chat(body)
    return await proxy_openai(target, "/v1/chat/completions", body)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    target = resolve_service(model)
    return await proxy_openai(target, "/v1/completions", body)


@app.post("/v1/video/generate")
async def video_generate(request: Request):
    body = await request.json()
    return await proxy_openai("http://localhost:8003", "/generate", body)


@app.post("/v1/images/analyze")
async def image_analyze(request: Request):
    body = await request.json()
    return await proxy_openai("http://localhost:8002", "/v1/chat/completions", body)


async def proxy_openai(base: str, path: str, body: dict):
    stream = body.get("stream", False)
    async with httpx.AsyncClient(timeout=600.0) as c:
        if stream:
            async def streamer():
                async with c.stream("POST", base + path, json=body) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            return StreamingResponse(streamer(), media_type="text/event-stream")
        r = await c.post(base + path, json=body)
        return JSONResponse(content=r.json(), status_code=r.status_code)


async def proxy_ollama_chat(body: dict):
    """Translate OpenAI chat format -> Ollama /api/chat -> back to OpenAI format."""
    ollama_body = {
        "model": OLLAMA_MODEL_NAME,
        "messages": body.get("messages", []),
        "stream": False,
        "options": {
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
        }
    }
    async with httpx.AsyncClient(timeout=600.0) as c:
        r = await c.post("http://localhost:11434/api/chat", json=ollama_body)
        data = r.json()
    content = data.get("message", {}).get("content", "")
    return JSONResponse({
        "id": "chatcmpl-ollama-lilith",
        "object": "chat.completion",
        "model": OLLAMA_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": data.get("eval_count", {})
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
