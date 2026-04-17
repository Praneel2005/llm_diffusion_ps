"""
Evaluates Branch A output against original paper figures.
Computes CLIP score (caption-image alignment).
"""
import os, json, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def compute_clip_score(image_path: str, caption: str) -> float:
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(text=[caption], images=image,
                      return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs    = model(**inputs)
        logits_per_image = outputs.logits_per_image
        score      = logits_per_image.item() / 100.0
    return score

def evaluate_paper(planned_json: str, figures_dir: str):
    with open(planned_json) as f:
        planned = json.load(f)
    
    scores = []
    print(f"\n=== CLIP Score Evaluation ===")
    print(f"{'Figure':<15} {'Type':<15} {'CLIP Score':<12}")
    print("-" * 45)
    
    for fig in planned:
        fig_id  = fig["id"]
        caption = fig.get("caption", "")
        fig_type = fig.get("figure_type", "")
        img_path = os.path.join(figures_dir, f"{fig_id}.png")
        
        if not os.path.exists(img_path):
            print(f"{fig_id:<15} {fig_type:<15} {'MISSING':<12}")
            continue
        
        score = compute_clip_score(img_path, caption)
        scores.append(score)
        print(f"{fig_id:<15} {fig_type:<15} {score:.4f}")
    
    if scores:
        print("-" * 45)
        print(f"{'Average':<30} {sum(scores)/len(scores):.4f}")
    return scores

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate_branch_a.py <planned_json> <figures_dir>")
        sys.exit(1)
    evaluate_paper(sys.argv[1], sys.argv[2])
