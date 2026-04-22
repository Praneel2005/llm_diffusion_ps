import json
import os
from datasets import load_dataset

def process_ai2d():
    print("Streaming AI2D-Caption dataset to bypass Hugging Face schema errors...")
    # Using streaming=True to bypass the strict Arrow casting crash
    ds = load_dataset('abhayzala/AI2D-Caption', split='train', streaming=True)
    
    output_file = "/home/drive3/llm_diffusion_ps/data/ai2d_mistral_train.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in ds:
            try:
                # 1. Extract inputs
                caption = row.get('caption', '')
                
                # 2. Fix the authors' mixed data types (List vs String)
                raw_rels = row.get('relationships', [])
                if isinstance(raw_rels, str):
                    try:
                        relationships = json.loads(raw_rels.replace("'", '"'))
                    except:
                        relationships = [raw_rels]
                else:
                    relationships = raw_rels
                
                # 3. Parse the raw entities JSON string
                entities_str = row.get('entities', '{}')
                if isinstance(entities_str, dict):
                    entities_dict = entities_str
                else:
                    entities_dict = json.loads(entities_str)
                
                # 4. Clean up targets for Mistral
                clean_entities = []
                for entity_id, entity_data in entities_dict.items():
                    clean_entities.append({
                        "label": entity_data.get("label", ""),
                        "type": entity_data.get("type", ""),
                        "bounds": entity_data.get("bounds", [])
                    })
                
                # 5. Format as Mistral ChatML
                user_prompt = f"Generate the spatial layout and bounding boxes for the following diagram.\n\nCaption: {caption}\n\nRelationships to maintain: {relationships}"
                assistant_response = json.dumps(clean_entities)
                
                chat_format = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                
                f.write(json.dumps(chat_format) + '\n')
                processed_count += 1
                
            except Exception as e:
                # Silently skip totally broken rows
                continue

    print(f"✅ Successfully formatted {processed_count} rows!")
    print(f"📁 Saved to: {output_file}")

if __name__ == "__main__":
    process_ai2d()
