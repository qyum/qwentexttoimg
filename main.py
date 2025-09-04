from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from diffusers import DiffusionPipeline
import torch
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# FastAPI setup
# ----------------------------
#router = APIRouter()
# ----------------------------






class PromptRequest(BaseModel):
    prompt: str
# Model setup
# ----------------------------
model_name = "Qwen/Qwen-Image"

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

# Enable VRAM optimizations
pipe.enable_attention_slicing()
# pipe.enable_sequential_cpu_offload()  # saves a lot of VRAM

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not available")

 
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}
negative_prompt = " "  # empty negative prompt

# Different safe aspect ratios
aspect_ratios = {
    "1:1": (768, 768),
    "16:9": (512, 512),   # reduced to avoid OOM
    "9:16": (540, 960),
    "4:3": (896, 672),
    "3:4": (672, 896),
    "3:2": (960, 640),
    "2:3": (640, 960),
}

# Pick default aspect ratio
default_width, default_height = aspect_ratios["16:9"]


 


@app.get("/")
def get_request():
    return {"Hello": "texttoimage"}


# Move pipeline to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
print(device)


@app.post("/generate")
def generate_image(request: PromptRequest):
    prompt = request.prompt + positive_magic["en"]
    width, height = default_width, default_height

    try:
        # Make generator on the same device as pipeline
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
        print("CUDA OOM: retrying with smaller resolution...")
        # Optional: implement fallback logic here

    # Save to buffer
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")             
