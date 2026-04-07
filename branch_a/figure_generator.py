import os
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs/figures"

def load_pipeline():
    print("[Branch A] Loading Stable Diffusion pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    # Enable memory efficient attention (xformers)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def build_prompt(caption: str) -> tuple:
    prompt = f"A clean scientific diagram showing: {caption.strip()}. White background, professional research paper style, clear labels, technical illustration, high quality."
    negative_prompt = "blurry, photorealistic, photograph, dark background, watermark, text errors, low quality, amateur"
    return prompt, negative_prompt

def generate_figure(pipe, caption: str, figure_num: str, output_dir: str) -> str:
    print(f"\n[Branch A] Generating Figure {figure_num}...")
    prompt, negative_prompt = build_prompt(caption)
    
    with torch.no_grad():
        result = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=20, guidance_scale=7.5, width=512, height=512)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"figure_{figure_num}.png")
    result.images[0].save(output_path)
    return output_path

def run_branch_a(figure_captions: list, output_dir: str = OUTPUT_DIR) -> list:
    if not figure_captions: return []
    pipe = load_pipeline()
    generated_paths = []
    for fig in tqdm(figure_captions, desc="Generating figures"):
        try:
            path = generate_figure(pipe, fig["caption"], fig["figure_num"], output_dir)
            generated_paths.append({"figure_num": fig["figure_num"], "caption": fig["caption"], "generated_path": path})
        except Exception as e:
            print(f"[Branch A] ERROR on Figure {fig['figure_num']}: {e}")
    return generated_paths
