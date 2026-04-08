"""
branch_a/figure_generator.py — Layer 2
----------------------------------------
Takes planned figure layouts and generates images using SDXL.
Adds text labels using Pillow after generation (not via SDXL).

Pipeline position:
  prompt_planner → figure_generator → outputs/figures/

Input:  list of planned figures (from prompt_planner.py)
Output: list of {"id": ..., "path": ..., "figure_type": ...}
"""

import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_pipeline     = None   # loaded once, reused across calls


def _load_pipeline():
    """Load SDXL once and keep it in memory."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("[BranchA] Loading SDXL pipeline (first run downloads ~6GB)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    _pipeline = _pipeline.to(device)
    _pipeline.enable_attention_slicing()   # saves VRAM with no quality loss

    print(f"[BranchA] SDXL loaded on {device}.")
    return _pipeline


def _overlay_labels(image: Image.Image, entities: list) -> Image.Image:
    """
    Draw entity labels onto the image using Pillow.
    SDXL cannot reliably render text — we add it after generation.
    bbox format: [x_pct, y_pct, w_pct, h_pct] — all 0-100
    """
    draw  = ImageDraw.Draw(image)
    W, H  = image.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for entity in entities:
        label = entity.get("label", "")
        bbox  = entity.get("bbox", [])

        if len(bbox) != 4 or not label:
            continue

        x_pct, y_pct, w_pct, h_pct = bbox

        # clamp to valid range
        x_pct = max(0, min(x_pct, 95))
        y_pct = max(0, min(y_pct, 95))

        x = int((x_pct / 100) * W)
        y = int((y_pct / 100) * H)

        # draw background box for readability
        text_bbox = draw.textbbox((x, y), label, font=font)
        padding   = 3
        draw.rectangle(
            [text_bbox[0]-padding, text_bbox[1]-padding,
             text_bbox[2]+padding, text_bbox[3]+padding],
            fill=(255, 255, 255, 200)
        )
        draw.text((x, y), label, fill=(20, 20, 20), font=font)

    return image


def generate_figure(fig: dict, output_dir: str) -> dict:
    """
    Generates one figure image from its planned layout.

    Input:  planned figure dict (from prompt_planner)
    Output: {"id": ..., "path": ..., "figure_type": ..., "caption": ...}
    """
    pipe       = _load_pipeline()
    fig_id     = fig["id"]
    sd_prompt  = fig.get("sd_prompt", "")
    entities   = fig.get("entities", [])
    figure_type = fig.get("figure_type", "conceptual")

    if not sd_prompt:
        sd_prompt = (
            f"A clean scientific {figure_type} diagram. "
            f"{fig.get('caption', '')[:200]}. "
            f"White background, professional research paper style."
        )

    # append quality boosters
    full_prompt = (
        sd_prompt +
        ", white background, clean scientific illustration, "
        "high resolution, professional academic paper figure"
    )
    negative_prompt = (
        "blurry, noisy, low quality, photorealistic, photograph, "
        "dark background, watermark, text errors, distorted"
    )

    print(f"  [BranchA] Generating {fig_id} ({figure_type})...", flush=True)

    with torch.inference_mode():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024,
            width=1024,
        )

    image = result.images[0]

    # overlay text labels using Pillow
    if entities:
        image = _overlay_labels(image, entities)

    # save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{fig_id}.png")
    image.save(out_path)
    print(f"  [BranchA] Saved: {out_path}")

    return {
        "id":          fig_id,
        "path":        out_path,
        "figure_type": figure_type,
        "caption":     fig.get("caption", ""),
        "sd_prompt":   full_prompt,
    }


def run_branch_a(planned_figures: list, output_dir: str) -> list:
    """
    Generates all figures from planned layouts.

    Input:  planned_figures — output of plan_all_figures()
    Output: list of generated figure dicts with file paths
    """
    if not planned_figures:
        print("[BranchA] No figures to generate.")
        return []

    print(f"\n[BranchA] Generating {len(planned_figures)} figures...")
    generated = []

    for i, fig in enumerate(planned_figures):
        print(f"\n  [{i+1}/{len(planned_figures)}]", end=" ")
        try:
            result = generate_figure(fig, output_dir)
            generated.append(result)
        except Exception as e:
            print(f"  [BranchA] ERROR on {fig['id']}: {e}")
            generated.append({
                "id":    fig["id"],
                "path":  None,
                "error": str(e)
            })

    success = sum(1 for g in generated if g.get("path"))
    print(f"\n[BranchA] Done. {success}/{len(planned_figures)} figures generated.")
    return generated


# ──────────────────────────────────────────────────────────────
# Test directly:
# python branch_a/figure_generator.py data/extracted/2306.00800v3_planned.json
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python branch_a/figure_generator.py <planned_json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        planned = json.load(f)

    base = os.path.basename(sys.argv[1]).replace("_planned.json", "")
    output_dir = f"outputs/figures/{base}"

    generated = run_branch_a(planned, output_dir)

    print("\n=== GENERATED FIGURES ===")
    for g in generated:
        status = g["path"] if g.get("path") else f"FAILED: {g.get('error')}"
        print(f"  [{g['id']}] {status}")
