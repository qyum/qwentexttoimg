from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from diffusers import DiffusionPipeline
import torch
import io
from fastapi.middleware.cors import CORSMiddleware
import gc

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request model
# ----------------------------
class PromptRequest(BaseModel):
    prompt: str

# ----------------------------
# Model setup
# ----------------------------
model_name = "Qwen/Qwen-Image"

# Decide torch dtype based on device
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

# ----------------------------
# Memory optimizations
# ----------------------------
pipe.enable_attention_slicing()

# try:
#     pipe.enable_xformers_memory_efficient_attention()
# except Exception:
#     print("xformers not available, continuing without it")

# Optional: enable CPU offload if GPU is limited
pipe.enable_sequential_cpu_offload()  # Uncomment if GPU runs out of memory

# ----------------------------
# Device setup
# ----------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# if not hasattr(pipe, "_cpu_offload_enabled") or not pipe._cpu_offload_enabled:
#     pipe = pipe.to(device)
#print(f"Pipeline device: {device}")

device = "cpu"


# ----------------------------
# Helpers
# ----------------------------
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}
negative_prompt = ""  # empty negative prompt

aspect_ratios = {
    "1:1": (768, 768),
    "16:9": (224, 224),  # reduced to avoid OOM
    "9:16": (540, 960),
    "4:3": (896, 672),
    "3:4": (672, 896),
    "3:2": (960, 640),
    "2:3": (640, 960),
}
default_width, default_height = aspect_ratios["16:9"]

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"message": "Qwen Text-to-Image API"}

@app.post("/generate")
def generate_image(request: PromptRequest):
    prompt = request.prompt + positive_magic["en"]
    width, height = default_width, default_height

    # Try to generate image with fallback for OOM
    
    try:
        generator = torch.Generator(device=device).manual_seed(42)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
     
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM on attempt {attempt+1}. Retrying with smaller resolution...")
        torch.cuda.empty_cache()
        gc.collect()
        # fallback resolution
        width = max(128, width // 2)
        height = max(128, height // 2)
     
    # Save to buffer
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
