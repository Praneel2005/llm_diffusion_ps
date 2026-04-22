"""
branch_a/hybrid_generator.py — Structure-Guided Diffusion (The Final Fix)
"""
import os, json, torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLImg2ImgPipeline

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("[Hybrid] Loading Base SDXL Img2Img Pipeline (No LoRAs needed)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_ID, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        ).to(device)
        _pipeline.enable_attention_slicing()
    return _pipeline

def draw_blueprint(entities, relations, width=1024, height=1024):
    """Draws the layout with boxes and lines, NO TEXT."""
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    
    # Draw connections (lines) first so they go behind boxes
    centers = {}
    for e in entities:
        bbox = e.get("bbox", [0,0,10,10])
        x = int((bbox[0] / 100) * width)
        y = int((bbox[1] / 100) * height)
        w = int((bbox[2] / 100) * width)
        h = int((bbox[3] / 100) * height)
        centers[e["id"]] = (x + w//2, y + h//2)
        
    for rel in relations:
        src = centers.get(rel.get("from"))
        dst = centers.get(rel.get("to"))
        if src and dst:
            draw.line([src, dst], fill="#888888", width=8)

    # Draw colored boxes
    for e in entities:
        bbox = e.get("bbox", [0,0,10,10])
        x = int((bbox[0] / 100) * width)
        y = int((bbox[1] / 100) * height)
        w = int((bbox[2] / 100) * width)
        h = int((bbox[3] / 100) * height)
        draw.rounded_rectangle([x, y, x+w, y+h], radius=15, fill="#4A90D9")
        
    return img

def overlay_text(img, entities, width=1024, height=1024):
    """Stamps the exact labels back onto the stylized image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        
    for e in entities:
        label = e.get("label", "")
        bbox = e.get("bbox", [0,0,10,10])
        x = int((bbox[0] / 100) * width)
        y = int((bbox[1] / 100) * height)
        w = int((bbox[2] / 100) * width)
        h = int((bbox[3] / 100) * height)
        
        # Draw text exactly in the center of where the box is
        tb = draw.textbbox((0,0), label, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        
        tx = x + (w - text_w) // 2
        ty = y + (h - text_h) // 2
        
        # Add a subtle dark glow behind text for readability
        draw.text((tx+2, ty+2), label, fill="#000000", font=font)
        draw.text((tx, ty), label, fill="#FFFFFF", font=font)
        
    return img

def generate_hybrid_figure(fig_plan, output_dir):
    fig_id = fig_plan["id"]
    entities = fig_plan.get("entities", [])
    relations = fig_plan.get("relations", [])
    
    print(f"  [Hybrid] Processing {fig_id}...")
    
    # 1. Create the dummy blueprint
    blueprint = draw_blueprint(entities, relations)
    
    # 2. Stylize with SDXL (The magic step)
    pipe = get_pipeline()
    prompt = "A highly professional, modern scientific vector diagram, 3D glassmorphism, clean corporate UI/UX style, glowing edges, soft shadows, white background, high resolution, academic paper asset."
    negative_prompt = "messy, chaotic, distorted, noisy, organic, photo, handwritten, text"
    
    with torch.inference_mode():
        # strength=0.35 means 65% structure, 35% AI stylization
        result = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            image=blueprint, 
            strength=0.70,  
            guidance_scale=8.5,
            num_inference_steps=50
        )
    
    stylized_img = result.images[0]
    
    # 3. Add the perfect text back on top
    final_img = overlay_text(stylized_img, entities)
    
    # Save it
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{fig_id}_hybrid.png")
    final_img.save(out_path)
    print(f"  [Hybrid] Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hybrid_generator.py <planned_json>")
        sys.exit(1)
        
    with open(sys.argv[1]) as f:
        planned = json.load(f)
        
    base = os.path.basename(sys.argv[1]).replace("_planned.json", "")
    out_dir = f"outputs/figures/{base}_hybrid"
    
    print(f"\n=== Running Hybrid Image Generation ===")
    for fig in planned:
        generate_hybrid_figure(fig, out_dir)
