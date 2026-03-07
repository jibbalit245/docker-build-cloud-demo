import runpod
import subprocess
import time
import requests
import os

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
    procs.append(subprocess.Popen(["python3", "/app/vl_server.py"]))

    # Wan2.2 server
    procs.append(subprocess.Popen(["python3", "/app/wan_server.py"]))

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

# Wait for vLLM as the slowest service; others will be up by then
wait_for_service("http://localhost:8001/health", timeout=300)
wait_for_service("http://localhost:11434/api/tags", timeout=120)

print("All services ready — accepting jobs")


# ── Handler ────────────────────────────────────────────────────────────────

SERVICES = {
    "qwen3-80b":  "http://localhost:8001",
    "qwen3":      "http://localhost:8001",
    "text":       "http://localhost:8001",
    "vision":     "http://localhost:8002",
    "vl":         "http://localhost:8002",
    "video":      "http://localhost:8003",
    "wan":        "http://localhost:8003",
    "lilith":     "http://localhost:11434",
    "whisper":    "http://localhost:11434",
}
OLLAMA_MODEL = "lilith-whisper:latest"


def resolve(model):
    m = (model or "").lower()
    for k, v in SERVICES.items():
        if k in m:
            return v
    return "http://localhost:8001"


def handler(job):
    inp = job.get("input", {})
    job_type = inp.get("type", "chat")      # chat | video | image_analyze
    model    = inp.get("model", "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated")
    prompt   = inp.get("prompt", "")
    messages = inp.get("messages", [{"role": "user", "content": prompt}])
    max_tokens = inp.get("max_tokens", 512)
    temperature = inp.get("temperature", 0.7)

    try:
        if job_type == "video":
            r = requests.post("http://localhost:8003/generate",
                json={"prompt": prompt, "num_frames": inp.get("num_frames", 81)},
                timeout=600)
            r.raise_for_status()
            return r.json()

        elif job_type == "image_analyze":
            r = requests.post("http://localhost:8002/v1/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=120)
            r.raise_for_status()
            return r.json()

        elif resolve(model) == "http://localhost:11434":
            r = requests.post("http://localhost:11434/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages,
                      "stream": False,
                      "options": {"temperature": temperature}},
                timeout=300)
            r.raise_for_status()
            data = r.json()
            return {"output": data.get("message", {}).get("content", ""),
                    "model": OLLAMA_MODEL}

        else:
            r = requests.post(resolve(model) + "/v1/chat/completions",
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=300)
            r.raise_for_status()
            return r.json()

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
