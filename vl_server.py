"""
Qwen2.5-VL-32B Vision-Language Server
OpenAI-compatible /v1/chat/completions with image support.

NOTE: litmudoc/Qwen2.5-VL-32B-Instruct-abliterated-MLX-Q8 is Apple Silicon (MLX) only.
This server uses huihui-ai/Qwen2.5-VL-32B-Instruct-abliterated for NVIDIA GPU compatibility.
Override with --model flag or VL_MODEL_ID env var.
"""
import argparse
import base64
import json
from io import BytesIO
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

app = FastAPI(title="Qwen2.5-VL Server", version="1.0.0")

model = None
processor = None
model_id = None


def load_model(model_id_: str, gpu_frac: float = 0.35):
    global model, processor, model_id
    model_id = model_id_
    print(f"[vl_server] Loading {model_id} (gpu_frac={gpu_frac})...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[vl_server] {model_id} loaded.")


class Message(BaseModel):
    role: str
    content: object


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok", "model": model_id, "loaded": model is not None}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    # Build messages in Qwen-VL format
    qwen_messages = []
    for msg in req.messages:
        content = msg["content"]
        role = msg["role"]
        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        parts.append({"type": "image", "image": url})
                    else:
                        parts.append({"type": "image", "image": url})
            qwen_messages.append({"role": role, "content": parts})

    text_prompt = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    response_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    return {
        "id": "chatcmpl-vl",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/hf_cache/Qwen2.5-VL-32B-Instruct-abliterated")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--gpu-frac", type=float, default=0.35)
    args = parser.parse_args()

    load_model(args.model, args.gpu_frac)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
