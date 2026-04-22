"""
scripts/prepare_spatial_dataset.py
-----------------------------------
Invincible SOTA extraction script. Uses recursive hunting to find bounding boxes 
no matter how the dataset authors nested them!
"""

import json
import os
import ast
from tqdm import tqdm

RAW_DATA_PATH = "/home/drive3/llm_diffusion_ps/data/paper2fig100k/paper2fig_train.json"
OUTPUT_JSONL  = "/home/drive3/llm_diffusion_ps/data/mistral_spatial_train.jsonl"

def extract_entities(obj):
    """Recursively hunts down any text/bbox combinations in arbitrary JSON structures."""
    entities = []
    if isinstance(obj, dict):
        # Pattern 1: Standard object {"text": "...", "bbox": "..."}
        if ("text" in obj or "label" in obj) and ("bbox" in obj or "box" in obj or "boxes" in obj):
            text = obj.get("text") or obj.get("label")
            bbox = obj.get("bbox") or obj.get("box") or obj.get("boxes")
            
            # Bbox might be a stringified list like "[[x,y],...]"
            if isinstance(bbox, str):
                try: 
                    bbox = ast.literal_eval(bbox)
                except: 
                    pass
                
            if text and isinstance(bbox, list):
                entities.append({"text": str(text).strip(), "box": bbox})
        else:
            # Pattern 2: Nested dictionaries or Key=Text, Value=Bbox
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    entities.extend(extract_entities(v))
                else:
                    if isinstance(v, str) and "[" in v and "]" in v:
                        try:
                            val = ast.literal_eval(v)
                            if isinstance(val, list):
                                entities.append({"text": str(k).strip(), "box": val})
                        except:
                            pass
    elif isinstance(obj, list):
        for item in obj:
            entities.extend(extract_entities(item))
            
    return entities

def process_dataset():
    print(f"🔍 Loading raw dataset from {RAW_DATA_PATH}...")
    
    with open(RAW_DATA_PATH, 'r') as f:
        raw_data = json.load(f)

    items = raw_data if isinstance(raw_data, list) else list(raw_data.values())
    print(f"📊 Processing {len(items)} total papers...")

    dataset_lines = []
    valid_count = 0

    for item in tqdm(items, desc="Formatting for Mistral"):
        captions = item.get("captions", [])
        caption = " ".join(captions) if isinstance(captions, list) else str(captions)
        caption = caption.strip()
        
        ocr_result = item.get("ocr_result", {})
        
        if not caption or not ocr_result:
            continue
            
        # Call the invincible recursive extractor!
        entities = extract_entities(ocr_result)

        if not entities:
            continue

        target_json = {"entities": entities}

        system_prompt = "You are a Spatial Architecture Planner. Given a scientific figure caption, output a precise JSON layout matrix containing entities and their bounding box polygons."
        user_prompt = f"Caption: \"{caption}\"\nGenerate the JSON layout matrix."
        
        chat_format = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(target_json)}
            ]
        }

        dataset_lines.append(chat_format)
        valid_count += 1

    print(f"\n💾 Saving {valid_count} training examples to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for line in dataset_lines:
            f.write(json.dumps(line) + '\n')
            
    print("✅ Dataset preparation complete. Ready for SOTA Mistral Fine-Tuning.")

if __name__ == "__main__":
    process_dataset()
