"""
Wan2.2 Video Generation Server
REST API for text-to-video and image-to-video generation.

Model auto-downloads from HuggingFace to /workspace/models/wan2.2 on first run.
Supports: Wan2.2-T2V-14B, Wan2.2-I2V-14B
"""
import argparse
import base64
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Wan2.2 Video Server", version="1.0.0")

pipe = None
model_dir = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_wan_model(model_dir_: str):
    global pipe, model_dir
    model_dir = model_dir_

    try:
        from diffusers import WanPipeline, WanImageToVideoPipeline
        t2v_path = os.path.join(model_dir_, "Wan2.2-T2V-14B")

        if not os.path.exists(t2v_path):
            print(f"[wan_server] Downloading Wan2.2-T2V-14B to {t2v_path}...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Wan-AI/Wan2.2-T2V-14B",
                local_dir=t2v_path,
                ignore_patterns=["*.md", "*.txt"],
            )

        print(f"[wan_server] Loading Wan2.2-T2V-14B from {t2v_path}...")
        pipe = WanPipeline.from_pretrained(
            t2v_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("[wan_server] Wan2.2 loaded.")
    except Exception as e:
        print(f"[wan_server] WARNING: Could not load Wan2.2: {e}")
        print("[wan_server] Server will start but /generate will return 503 until model loads.")


class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality, distorted"
    num_frames: Optional[int] = 81
    height: Optional[int] = 480
    width: Optional[int] = 832
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 5.0
    seed: Optional[int] = None
    output_format: Optional[str] = "base64"  # "base64" or "path"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "device": device,
        "model_dir": model_dir,
    }


@app.post("/generate")
def generate_video(req: VideoRequest):
    if pipe is None:
        return JSONResponse(
            {"error": "Wan2.2 model not loaded yet. Check /workspace/logs/wan_server.log"},
            status_code=503
        )

    gen = torch.Generator(device=device)
    if req.seed is not None:
        gen.manual_seed(req.seed)

    print(f"[wan_server] Generating: '{req.prompt[:60]}...' | {req.num_frames} frames")
    t0 = time.time()

    output = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=gen,
    )

    elapsed = time.time() - t0
    frames = output.frames[0]

    # Save video
    out_path = f"/workspace/videos/{uuid.uuid4().hex}.mp4"
    os.makedirs("/workspace/videos", exist_ok=True)

    import imageio
    writer = imageio.get_writer(out_path, fps=16, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    response = {
        "status": "ok",
        "prompt": req.prompt,
        "num_frames": len(frames),
        "resolution": f"{req.width}x{req.height}",
        "elapsed_seconds": round(elapsed, 2),
        "path": out_path,
    }

    if req.output_format == "base64":
        with open(out_path, "rb") as f:
            response["video_b64"] = base64.b64encode(f.read()).decode()

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/wan2.2")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--no-autoload", action="store_true",
                        help="Skip model loading at startup (load manually via /load)")
    args = parser.parse_args()

    if not args.no_autoload:
        load_wan_model(args.model_dir)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
