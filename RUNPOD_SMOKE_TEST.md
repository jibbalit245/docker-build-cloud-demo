# RunPod Smoke Test (5 Minutes)

Use this checklist right after pod startup to validate the stack end-to-end.

## Preconditions
- Pod is running with `/workspace` network volume mounted.
- Runtime env includes `HF_TOKEN` if gated downloads are required.
- Exposed ports include `3000`, `8000`, `8001`, `8002`, `8003`, `11434`.

## 1. Verify startup reached ready state
- Check `/workspace/logs/workflow.log` contains `phase=workflow status=ready`.
- Check `/workspace/logs/service_map.json` exists.

## 2. Health endpoints
Run these from inside the container (or via mapped URLs):

```bash
curl -sf http://127.0.0.1:8000/health | jq .
curl -sf http://127.0.0.1:8000/v1/models | jq .
curl -sf http://127.0.0.1:8002/health | jq .
curl -sf http://127.0.0.1:8003/health | jq .
curl -sf http://127.0.0.1:11434/api/tags | jq .
```

Expected:
- Gateway returns `status` as `ok` or `degraded` with service details.
- Vision health includes `"loaded": true`.
- Wan health includes `"model_loaded": true`.
- Ollama tags include a Lilith model.

## 3. Text route smoke test (gateway -> vLLM)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-80b",
    "messages": [{"role":"user","content":"Reply with one word: ready"}],
    "max_tokens": 8,
    "temperature": 0
  }' | jq .
```

## 4. Vision route smoke test (gateway -> vl_server)

```bash
curl -s http://127.0.0.1:8000/v1/images/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vision",
    "messages": [
      {"role":"user","content":[
        {"type":"text","text":"What color is the square?"},
        {"type":"image_url","image_url":{"url":"https://dummyimage.com/256x256/0000ff/ffffff.png&text=blue"}}
      ]}
    ],
    "max_tokens": 64
  }' | jq .
```

## 5. Wan route smoke test (gateway -> wan_server)

```bash
curl -s http://127.0.0.1:8000/v1/video/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A short clip of ocean waves at sunrise",
    "num_frames": 16,
    "num_inference_steps": 10,
    "output_format": "path"
  }' | jq .
```

Expected:
- Response includes `"status": "ok"` and a `path` under `/workspace/videos/`.

## 6. Ollama route smoke test (gateway -> ollama)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lilith",
    "messages": [{"role":"user","content":"Reply with one word: online"}],
    "max_tokens": 16,
    "temperature": 0
  }' | jq .
```

## 7. Restart/cache reuse check
- Restart pod with the same `/workspace` volume.
- Confirm startup logs show model and dataset downloads are skipped.

## Failure triage
- `start.sh` exits early: inspect `/workspace/logs/workflow.log` and backend logs in `/workspace/logs/`.
- Vision not loaded: inspect `/workspace/logs/vl_server.log`.
- Wan not loaded: inspect `/workspace/logs/wan_server.log`.
- Ollama model missing: inspect `/workspace/logs/ollama_pull.log`.
