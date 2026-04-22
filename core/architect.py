import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def generate_layout():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_path = "/home/drive3/llm_diffusion_ps/models/mistral_spatial_overfit"
    
    print("🧠 Loading Mistral Architect with 4-bit Quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("🎨 Attaching the Spatial Adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Test Prompt: Transformer Encoder
    prompt = "Generate the spatial layout and bounding boxes for the following diagram.\n\nCaption: A CNN architecture with a Conv2D layer, followed by MaxPooling, then a Flatten layer, and finally a Dense output layer."
    
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to("cuda")
    
    print("✍️ Architect is designing the blueprint...")
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extracting the JSON response after the [/INST] tag
    try:
        json_str = response.split("[/INST]")[-1].strip()
        print("\n--- ARCHITECT'S BLUEPRINT ---")
        print(json_str)
        
        layout = json.loads(json_str)
        return layout
    except Exception as e:
        print(f"\n--- RAW OUTPUT --- \n{response}")
        print(f"\n❌ Error parsing JSON: {e}")
        return None

if __name__ == "__main__":
    generate_layout()
