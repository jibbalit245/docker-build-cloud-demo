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
import asyncio
import os
import uuid
import json

app = FastAPI(title="Omni-Stack Gateway", version="1.0.0")

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
VL_URL = os.getenv("VL_URL", "http://127.0.0.1:8002")
WAN_URL = os.getenv("WAN_URL", "http://127.0.0.1:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

SERVICES = {
    "qwen3-80b":   VLLM_URL,
    "qwen3":       VLLM_URL,
    "text":        VLLM_URL,
    "qwen2.5-vl":  VL_URL,
    "vision":      VL_URL,
    "vl":          VL_URL,
    "wan2.2":      WAN_URL,
    "video":       WAN_URL,
    "wan":         WAN_URL,
    "lilith":      OLLAMA_URL,
    "llama":       OLLAMA_URL,
    "l3.3":        OLLAMA_URL,
    "whisper":     OLLAMA_URL,
}
DEFAULT_TEXT = VLLM_URL

OLLAMA_MODEL_NAME = os.getenv(
    "OLLAMA_MODEL_NAME",
    "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.2a.i1-Q4_K_M-GGUF:latest",
)
OLLAMA_MODEL_CANDIDATES = [
    m.strip() for m in os.getenv("OLLAMA_MODEL_CANDIDATES", "").split(",") if m.strip()
]
RETRY_ATTEMPTS = int(os.getenv("GATEWAY_RETRY_ATTEMPTS", "3"))
RETRY_BASE_DELAY = float(os.getenv("GATEWAY_RETRY_BASE_DELAY", "0.5"))
FORCE_SYNTHETIC_SSE = os.getenv("FORCE_SYNTHETIC_SSE", "true").lower() == "true"
OLLAMA_ALIAS_MODELS = {
    "lilith",
    "lilith-whisper",
    "whisper",
    "llama",
    "l3.3",
    "lilith-l3.3-70b",
}
OLLAMA_MODEL_CACHE = None


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def _response_payload(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return {"error": "upstream returned non-json", "raw": resp.text[:2000]}


async def _post_with_retry(url: str, body: dict, timeout: float, headers: dict) -> httpx.Response:
    transient_status = {502, 503, 504}
    last_exc = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(url, json=body, headers=headers)
            if r.status_code in transient_status and attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return r
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ) as exc:
            last_exc = exc
            if attempt >= RETRY_ATTEMPTS - 1:
                break
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    raise last_exc or RuntimeError("upstream request failed")


def resolve_service(model: str) -> str:
    if not model:
        return DEFAULT_TEXT
    m = model.lower().strip()

    # Common model aliases from WebUI/client presets.
    alias_map = {
        "qwen3": "qwen3-80b",
        "qwen3-80b-instruct": "qwen3-80b",
        "huihui-ai/huihui-qwen3-next-80b-a3b-instruct-abliterated": "qwen3-80b",
        "qwen2.5-vl-32b": "qwen2.5-vl",
        "qwen2.5-vl": "qwen2.5-vl",
        "qwen2.5-vl-32b-instruct-abliterated": "qwen2.5-vl",
        "huihui-ai/qwen2.5-vl-32b-instruct-abliterated": "qwen2.5-vl",
        "vision": "qwen2.5-vl",
        "wan2.2-t2v": "wan2.2",
        "wan-ai/wan2.2-t2v-14b": "wan2.2",
        "wan": "wan2.2",
        "lilith": "lilith",
        "lilith-whisper": "lilith",
        "liliths-whisper-l3.3-70b-0.2a.i1-q4_k_m.gguf": "lilith",
        "liliths-whisper-l3.3-70b-0.2a.i1-q5_k_m.gguf": "lilith",
        "whisper": "lilith",
    }
    m = alias_map.get(m, m)

    for key, url in SERVICES.items():
        if key in m:
            return url
    return DEFAULT_TEXT


def _stream_chunk_from_response(payload: dict) -> dict:
    content = ""
    choices = payload.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""

    return {
        "id": payload.get("id", "chatcmpl-synth"),
        "object": "chat.completion.chunk",
        "model": payload.get("model", "unknown"),
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def _resolve_ollama_model(requested_model: str, req_id: str) -> str:
    """Pick an installed Ollama model, even when versioned tags differ."""
    global OLLAMA_MODEL_CACHE

    requested = (requested_model or "").strip()
    requested_lower = requested.lower()

    # If caller explicitly passed a concrete Ollama tag, honor it.
    if requested and requested_lower not in OLLAMA_ALIAS_MODELS and (":" in requested or "/" in requested):
        return requested

    if OLLAMA_MODEL_CACHE:
        return OLLAMA_MODEL_CACHE

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            data = _response_payload(r)
            names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        names = []

    candidates = []
    if OLLAMA_MODEL_NAME:
        candidates.append(OLLAMA_MODEL_NAME)
    candidates.extend(OLLAMA_MODEL_CANDIDATES)
    candidates.extend([
        "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q4_K_M-GGUF:latest",
        "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.2a.i1-Q4_K_M-GGUF:latest",
        "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF:latest",
        "lilith-whisper:latest",
    ])

    name_set = set(names)
    for candidate in candidates:
        if candidate in name_set:
            OLLAMA_MODEL_CACHE = candidate
            print(f"[gateway][{req_id}] ollama_model={OLLAMA_MODEL_CACHE} (exact)")
            return OLLAMA_MODEL_CACHE

    for name in names:
        lname = name.lower()
        if "lilith" in lname or "whisper" in lname or "l3.3" in lname:
            OLLAMA_MODEL_CACHE = name
            print(f"[gateway][{req_id}] ollama_model={OLLAMA_MODEL_CACHE} (discovered)")
            return OLLAMA_MODEL_CACHE

    if requested and requested_lower not in OLLAMA_ALIAS_MODELS:
        OLLAMA_MODEL_CACHE = requested
    elif OLLAMA_MODEL_NAME:
        OLLAMA_MODEL_CACHE = OLLAMA_MODEL_NAME
    elif names:
        OLLAMA_MODEL_CACHE = names[0]
    else:
        OLLAMA_MODEL_CACHE = "lilith-whisper:latest"

    print(f"[gateway][{req_id}] ollama_model={OLLAMA_MODEL_CACHE} (fallback)")
    return OLLAMA_MODEL_CACHE


@app.get("/health")
async def health():
    statuses = {}
    checks = {
        "vllm":   f"{VLLM_URL}/health",
        "vl":     f"{VL_URL}/health",
        "wan":    f"{WAN_URL}/health",
        "ollama": f"{OLLAMA_URL}/api/tags",
    }
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in checks.items():
            try:
                r = await client.get(url)
                statuses[name] = "ok" if r.status_code < 400 else f"error_{r.status_code}"
            except Exception:
                statuses[name] = "down"

    core_services = ("vllm", "ollama")
    core_ok = all(statuses.get(name) == "ok" for name in core_services)
    any_ok = any(v == "ok" for v in statuses.values())
    if core_ok:
        overall = "ok"
    elif any_ok:
        overall = "degraded"
    else:
        overall = "down"

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


@app.get("/workflow")
async def workflow():
    """Return the live routing workflow so operators can verify model-to-service links."""
    return {
        "gateway": {
            "v1_chat_completions": "/v1/chat/completions",
            "v1_video_generate": "/v1/video/generate",
            "v1_images_analyze": "/v1/images/analyze",
        },
        "services": {
            "vllm": VLLM_URL,
            "vision": VL_URL,
            "wan": WAN_URL,
            "ollama": OLLAMA_URL,
        },
        "aliases": {
            "text": ["qwen3-80b", "qwen3", "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated"],
            "vision": ["vision", "qwen2.5-vl", "huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated"],
            "video": ["wan2.2", "wan", "Wan-AI/Wan2.2-T2V-14B"],
            "ollama": ["lilith", "whisper", OLLAMA_MODEL_NAME],
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_id = _request_id(request)
    body = await request.json()
    model = body.get("model", "")
    target = resolve_service(model)
    print(f"[gateway][{req_id}] route=/v1/chat/completions model={model} target={target}")
    if target == OLLAMA_URL:
        return await proxy_ollama_chat(body, req_id)
    if target == WAN_URL:
        return await proxy_wan_chat(body, req_id)
    return await proxy_openai(target, "/v1/chat/completions", body, req_id)


@app.post("/v1/completions")
async def completions(request: Request):
    req_id = _request_id(request)
    body = await request.json()
    model = body.get("model", "")
    target = resolve_service(model)
    print(f"[gateway][{req_id}] route=/v1/completions model={model} target={target}")
    return await proxy_openai(target, "/v1/completions", body, req_id)


@app.post("/v1/video/generate")
async def video_generate(request: Request):
    req_id = _request_id(request)
    body = await request.json()
    print(f"[gateway][{req_id}] route=/v1/video/generate target={WAN_URL}")
    return await proxy_openai(WAN_URL, "/generate", body, req_id)


@app.post("/v1/images/analyze")
async def image_analyze(request: Request):
    req_id = _request_id(request)
    body = await request.json()
    print(f"[gateway][{req_id}] route=/v1/images/analyze target={VL_URL}")
    return await proxy_openai(VL_URL, "/v1/chat/completions", body, req_id)


async def proxy_openai(base: str, path: str, body: dict, req_id: str):
    stream = body.get("stream", False)
    headers = {"x-request-id": req_id}

    if stream and FORCE_SYNTHETIC_SSE:
        non_stream_body = dict(body)
        non_stream_body["stream"] = False
        try:
            r = await _post_with_retry(base + path, non_stream_body, timeout=600.0, headers=headers)
            payload = _response_payload(r)
            if r.status_code >= 400:
                return JSONResponse(
                    content=payload,
                    status_code=r.status_code,
                    headers={"x-request-id": req_id},
                )
        except Exception as exc:
            return JSONResponse(
                content={"error": str(exc), "request_id": req_id},
                status_code=502,
                headers={"x-request-id": req_id},
            )

        chunk = _stream_chunk_from_response(payload)

        async def synthetic_streamer():
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            synthetic_streamer(),
            media_type="text/event-stream",
            headers={"x-request-id": req_id},
        )

    if stream:
        async def streamer():
            last_exc = None
            for attempt in range(RETRY_ATTEMPTS):
                yielded = False
                try:
                    async with httpx.AsyncClient(timeout=600.0) as c:
                        async with c.stream("POST", base + path, json=body, headers=headers) as r:
                            r.raise_for_status()
                            async for chunk in r.aiter_bytes():
                                yielded = True
                                yield chunk
                            return
                except (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    httpx.WriteError,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    httpx.HTTPStatusError,
                ) as exc:
                    last_exc = exc
                    if yielded or attempt >= RETRY_ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

            if last_exc:
                raise last_exc

        return StreamingResponse(
            streamer(),
            media_type="text/event-stream",
            headers={"x-request-id": req_id},
        )

    try:
        r = await _post_with_retry(base + path, body, timeout=600.0, headers=headers)
        return JSONResponse(
            content=_response_payload(r),
            status_code=r.status_code,
            headers={"x-request-id": req_id},
        )
    except Exception as exc:
        return JSONResponse(
            content={"error": str(exc), "request_id": req_id},
            status_code=502,
            headers={"x-request-id": req_id},
        )


async def proxy_ollama_chat(body: dict, req_id: str):
    """Translate OpenAI chat format -> Ollama /api/chat -> back to OpenAI format."""
    ollama_model = await _resolve_ollama_model(body.get("model", ""), req_id)
    ollama_body = {
        "model": ollama_model,
        "messages": body.get("messages", []),
        "stream": False,
        "options": {
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
        }
    }
    headers = {"x-request-id": req_id}
    try:
        r = await _post_with_retry(f"{OLLAMA_URL}/api/chat", ollama_body, timeout=600.0, headers=headers)
        data = _response_payload(r)
    except Exception as exc:
        return JSONResponse(
            content={"error": str(exc), "request_id": req_id},
            status_code=502,
            headers=headers,
        )

    content = data.get("message", {}).get("content", "")
    return JSONResponse(
        {
            "id": "chatcmpl-ollama-lilith",
            "object": "chat.completion",
            "model": ollama_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": data.get("eval_count", {})
        },
        headers=headers,
    )


def _extract_prompt(messages: list) -> str:
    """Extract a text prompt from OpenAI-style messages for Wan generation."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                return "\n".join(text_parts)
    return ""


async def proxy_wan_chat(body: dict, req_id: str):
    """Translate OpenAI chat requests for wan models into Wan /generate calls."""
    prompt = body.get("prompt") or _extract_prompt(body.get("messages", []))
    if not prompt:
        return JSONResponse(
            {
                "error": "wan model requires a text prompt in request.prompt or user message content",
                "request_id": req_id,
            },
            status_code=400,
            headers={"x-request-id": req_id},
        )

    wan_req = {
        "prompt": prompt,
        "negative_prompt": body.get("negative_prompt", "blurry, low quality, distorted"),
        "num_frames": body.get("num_frames", 81),
        "height": body.get("height", 480),
        "width": body.get("width", 832),
        "num_inference_steps": body.get("num_inference_steps", 50),
        "guidance_scale": body.get("guidance_scale", 5.0),
        "seed": body.get("seed"),
        # Avoid huge payloads on chat endpoint.
        "output_format": body.get("output_format", "path"),
    }

    headers = {"x-request-id": req_id}
    try:
        r = await _post_with_retry(f"{WAN_URL}/generate", wan_req, timeout=1200.0, headers=headers)
        if r.status_code >= 400:
            return JSONResponse(
                content=_response_payload(r),
                status_code=r.status_code,
                headers=headers,
            )
        video = _response_payload(r)
    except Exception as exc:
        return JSONResponse(
            content={"error": str(exc), "request_id": req_id},
            status_code=502,
            headers=headers,
        )

    summary = (
        f"Video generated: {video.get('path', 'unknown path')} "
        f"({video.get('resolution', 'unknown res')}, {video.get('num_frames', 'n/a')} frames)."
    )
    return JSONResponse(
        {
            "id": "chatcmpl-wan2.2",
            "object": "chat.completion",
            "model": body.get("model", "wan2.2"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": summary,
                    },
                    "finish_reason": "stop",
                }
            ],
            "video": video,
        },
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
