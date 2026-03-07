import runpod
import subprocess
import time
import requests
import os
import uuid

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
VL_URL = os.getenv("VL_URL", "http://127.0.0.1:8002")
WAN_URL = os.getenv("WAN_URL", "http://127.0.0.1:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
RETRY_ATTEMPTS = int(os.getenv("HANDLER_RETRY_ATTEMPTS", "3"))
RETRY_BASE_DELAY = float(os.getenv("HANDLER_RETRY_BASE_DELAY", "0.5"))

# ── Start all backend services at worker boot ──────────────────────────────

def start_services():
    procs = []

    # Ollama
    procs.append(subprocess.Popen(["ollama", "serve"],
        env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434",
             "OLLAMA_MODELS": os.getenv("OLLAMA_MODELS", "/workspace/ollama_models")}))

    # vLLM (Qwen3-80B)
    procs.append(subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", os.getenv("VLLM_MODEL", "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated"),
        "--port", "8001", "--host", "0.0.0.0",
        "--download-dir", os.getenv("HF_HOME", "/workspace/hf_cache"),
        "--gpu-memory-utilization", os.getenv("GPU_MEMORY_UTILIZATION", "0.45"),
        "--tensor-parallel-size", os.getenv("TENSOR_PARALLEL", "2"),
    ]))

    # Vision server
    procs.append(subprocess.Popen([
        "python3", "/app/vl_server.py",
        "--model", os.getenv("VL_MODEL", "huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated"),
        "--port", "8002",
        "--gpu-frac", os.getenv("VL_GPU_FRAC", "0.35"),
    ]))

    # Wan2.2 server
    procs.append(subprocess.Popen([
        "python3", "/app/wan_server.py",
        "--model-dir", os.getenv("WAN_MODEL_BASE", "/workspace/hf_cache"),
        "--port", "8003",
    ]))

    return procs


def wait_for_service(url, timeout=300, interval=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code < 400:
                print(f"Ready: {url}")
                return True
        except Exception:
            pass
        print(f"Waiting for {url}...")
        time.sleep(interval)
    print(f"Timeout waiting for {url}")
    return False


print("Starting all services...")
procs = start_services()

# Wait for all services so requests do not fail while workers warm up.
wait_for_service(f"{VLLM_URL}/health", timeout=300)
wait_for_service(f"{VL_URL}/health", timeout=300)
wait_for_service(f"{WAN_URL}/health", timeout=300)
wait_for_service(f"{OLLAMA_URL}/api/tags", timeout=120)

print("All services ready — accepting jobs")


# ── Handler ────────────────────────────────────────────────────────────────

SERVICES = {
    "qwen3-80b":  VLLM_URL,
    "qwen3":      VLLM_URL,
    "text":       VLLM_URL,
    "vision":     VL_URL,
    "vl":         VL_URL,
    "video":      WAN_URL,
    "wan":        WAN_URL,
    "lilith":     OLLAMA_URL,
    "whisper":    OLLAMA_URL,
}
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL_NAME",
    "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.2a.i1-Q4_K_M-GGUF:latest",
)
OLLAMA_MODEL_CANDIDATES = [
    m.strip() for m in os.getenv("OLLAMA_MODEL_CANDIDATES", "").split(",") if m.strip()
]
OLLAMA_ALIAS_MODELS = {
    "lilith",
    "lilith-whisper",
    "whisper",
    "llama",
    "l3.3",
    "lilith-l3.3-70b",
}
OLLAMA_MODEL_CACHE = None


def resolve(model):
    m = (model or "").lower()
    for k, v in SERVICES.items():
        if k in m:
            return v
    return VLLM_URL


def resolve_ollama_model(requested_model, req_id):
    """Pick a valid installed Ollama tag even if exact version changed."""
    global OLLAMA_MODEL_CACHE

    requested = (requested_model or "").strip()
    requested_lower = requested.lower()

    if requested and requested_lower not in OLLAMA_ALIAS_MODELS and (":" in requested or "/" in requested):
        return requested

    if OLLAMA_MODEL_CACHE:
        return OLLAMA_MODEL_CACHE

    names = []
    try:
        data = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
        names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        names = []

    candidates = []
    if OLLAMA_MODEL:
        candidates.append(OLLAMA_MODEL)
    candidates.extend(OLLAMA_MODEL_CANDIDATES)
    candidates.extend([
        "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q4_K_M-GGUF:latest",
        "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.2a.i1-Q4_K_M-GGUF:latest",
        "hf.co/bartowski/Liliths-Whisper-L3.3-70b-0.1.Q5_K_M-GGUF:latest",
        "lilith-whisper:latest",
    ])

    for candidate in candidates:
        if candidate in names:
            OLLAMA_MODEL_CACHE = candidate
            print(f"[handler][{req_id}] ollama_model={OLLAMA_MODEL_CACHE} (exact)")
            return OLLAMA_MODEL_CACHE

    for name in names:
        lname = name.lower()
        if "lilith" in lname or "whisper" in lname or "l3.3" in lname:
            OLLAMA_MODEL_CACHE = name
            print(f"[handler][{req_id}] ollama_model={OLLAMA_MODEL_CACHE} (discovered)")
            return OLLAMA_MODEL_CACHE

    if requested and requested_lower not in OLLAMA_ALIAS_MODELS:
        OLLAMA_MODEL_CACHE = requested
    elif OLLAMA_MODEL:
        OLLAMA_MODEL_CACHE = OLLAMA_MODEL
    elif names:
        OLLAMA_MODEL_CACHE = names[0]
    else:
        OLLAMA_MODEL_CACHE = "lilith-whisper:latest"

    print(f"[handler][{req_id}] ollama_model={OLLAMA_MODEL_CACHE} (fallback)")
    return OLLAMA_MODEL_CACHE


def post_with_retry(url, payload, timeout, headers):
    transient_status = {502, 503, 504}
    last_exc = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, timeout=timeout, headers=headers)
            if response.status_code in transient_status and attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return response
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            last_exc = exc
            if attempt >= RETRY_ATTEMPTS - 1:
                break
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    raise last_exc or RuntimeError(f"request failed for {url}")


def handler(job):
    inp = job.get("input", {})
    req_id = inp.get("request_id") or job.get("id") or str(uuid.uuid4())
    headers = {"x-request-id": req_id}
    job_type = inp.get("type", "chat")      # chat | video | image_analyze
    model    = inp.get("model", "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated")
    prompt   = inp.get("prompt", "")
    messages = inp.get("messages", [{"role": "user", "content": prompt}])
    max_tokens = inp.get("max_tokens", 512)
    temperature = inp.get("temperature", 0.7)
    target = resolve(model)

    print(f"[handler][{req_id}] type={job_type} model={model} target={target}")

    try:
        if job_type == "video":
            r = post_with_retry(
                f"{WAN_URL}/generate",
                {"prompt": prompt, "num_frames": inp.get("num_frames", 81)},
                timeout=600,
                headers=headers,
            )
            r.raise_for_status()
            return r.json()

        elif job_type == "image_analyze":
            r = post_with_retry(
                f"{VL_URL}/v1/chat/completions",
                {"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=120,
                headers=headers,
            )
            r.raise_for_status()
            return r.json()

        elif target == OLLAMA_URL:
            ollama_model = resolve_ollama_model(model, req_id)
            r = post_with_retry(
                f"{OLLAMA_URL}/api/chat",
                {
                    "model": ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=300,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            return {"output": data.get("message", {}).get("content", ""),
                    "model": ollama_model,
                    "request_id": req_id}

        elif target == WAN_URL:
            r = post_with_retry(
                f"{WAN_URL}/generate",
                {
                    "prompt": prompt,
                    "num_frames": inp.get("num_frames", 81),
                    "height": inp.get("height", 480),
                    "width": inp.get("width", 832),
                    "num_inference_steps": inp.get("num_inference_steps", 50),
                    "guidance_scale": inp.get("guidance_scale", 5.0),
                    "seed": inp.get("seed"),
                    "output_format": inp.get("output_format", "path"),
                },
                timeout=1200,
                headers=headers,
            )
            r.raise_for_status()
            return r.json()

        else:
            r = post_with_retry(
                target + "/v1/chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=300,
                headers=headers,
            )
            r.raise_for_status()
            return r.json()

    except Exception as e:
        return {"error": str(e), "request_id": req_id}


runpod.serverless.start({"handler": handler})
