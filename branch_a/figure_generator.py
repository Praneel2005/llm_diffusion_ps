import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
branch_a/figure_generator.py — with VLM critic loop
"""

import os, json, torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_WEIGHTS  = {
    "architecture": "lora_weights/architecture/final",
    "flowchart":    "lora_weights/flowchart/final",
    "chart":        "lora_weights/chart/final",
    "conceptual":   "lora_weights/conceptual/final",
}
_pipelines = {}


def _load_pipeline(figure_type: str):
    if figure_type in _pipelines:
        return _pipelines[figure_type]
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    lora_path = LORA_WEIGHTS.get(figure_type)
    print(f"[BranchA] Loading SDXL for {figure_type}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID, torch_dtype=torch.float16,
        use_safetensors=True).to(device)
    if lora_path and os.path.exists(lora_path):
        print(f"[BranchA] Loading LoRA: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet = pipe.unet.merge_and_unload()
    else:
        print(f"[BranchA] No LoRA found at {lora_path}, using base SDXL.")
    pipe.enable_attention_slicing()
    _pipelines[figure_type] = pipe
    return pipe


def _overlay_labels(image, entities):
    draw = ImageDraw.Draw(image)
    W, H = image.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for entity in entities:
        label = entity.get("label", "")
        bbox  = entity.get("bbox", [])
        if len(bbox) != 4 or not label:
            continue
        x = int((max(0, min(bbox[0], 95)) / 100) * W)
        y = int((max(0, min(bbox[1], 95)) / 100) * H)
        tb = draw.textbbox((x, y), label, font=font)
        draw.rectangle([tb[0]-3, tb[1]-3, tb[2]+3, tb[3]+3],
                       fill=(255, 255, 255, 200))
        draw.text((x, y), label, fill=(20, 20, 20), font=font)
    return image


def generate_figure(fig: dict, output_dir: str) -> dict:
    """Single generation — used by VLM critic loop."""
    pipe        = _load_pipeline(fig.get("figure_type", "architecture"))
    fig_id      = fig["id"]
    sd_prompt   = fig.get("sd_prompt", "")
    figure_type = fig.get("figure_type", "architecture")
    entities    = fig.get("entities", [])

    if not sd_prompt:
        sd_prompt = (f"A clean scientific {figure_type} diagram. "
                     f"{fig.get('caption','')[:200]}. "
                     f"White background, professional research paper style.")

    full_prompt = (sd_prompt +
                   ", white background, clean lines, scientific diagram, "
                   "high resolution, professional academic paper figure")
    neg_prompt  = ("blurry, noisy, photorealistic, dark background, "
                   "watermark, distorted, handwritten")

    print(f"  [BranchA] Generating {fig_id} ({figure_type})...")
    with torch.inference_mode():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
        )
    image = result.images[0]
    if entities:
        image = _overlay_labels(image, entities)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{fig_id}.png")
    image.save(out_path)
    print(f"  [BranchA] Saved: {out_path}")
    return {"id": fig_id, "path": out_path,
            "figure_type": figure_type, "caption": fig.get("caption", "")}


def run_branch_a(planned_figures: list, output_dir: str,
                  use_vlm_critic: bool = True) -> list:
    """
    Generates all figures, optionally with VLM critic loop.
    use_vlm_critic=True  → quality refinement, slower
    use_vlm_critic=False → single pass, faster
    """
    if not planned_figures:
        print("[BranchA] No figures to generate.")
        return []

    if use_vlm_critic:
        from shared.vlm_critic import run_critic_loop_all
        print(f"\n[BranchA] Generating {len(planned_figures)} figures WITH VLM critic...")
        return run_critic_loop_all(planned_figures, generate_figure, output_dir)
    else:
        print(f"\n[BranchA] Generating {len(planned_figures)} figures (no critic)...")
        generated = []
        for i, fig in enumerate(planned_figures):
            print(f"\n  [{i+1}/{len(planned_figures)}]", end=" ")
            try:
                generated.append(generate_figure(fig, output_dir))
            except Exception as e:
                print(f"  ERROR: {e}")
                generated.append({"id": fig["id"], "path": None, "error": str(e)})
        return generated


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python branch_a/figure_generator.py <planned_json> [--no-critic]")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        planned = json.load(f)
    use_critic = "--no-critic" not in sys.argv
    base       = os.path.basename(sys.argv[1]).replace("_planned.json", "")
    results    = run_branch_a(planned, f"outputs/figures/{base}_vlm", use_critic)
    print("\n=== RESULTS ===")
    for r in results:
        score = r.get("vlm_score", "N/A")
        path  = r.get("final_path") or r.get("path") or "FAILED"
        print(f"  [{r['id']}] Score: {score}/10 | {path}")
