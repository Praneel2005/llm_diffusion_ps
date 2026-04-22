import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
"""
scripts/train_mistral_spatial.py
-----------------------------------
Trains Mistral-7B to act as a SOTA Spatial Architecture Planner using QLoRA.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "/home/drive3/llm_diffusion_ps/data/mistral_spatial_train.jsonl"
OUTPUT_DIR = "/home/drive3/llm_diffusion_ps/models/mistral_spatial_lora"

def train():
    print("🚀 Initializing SOTA Spatial Training...")

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.shuffle(seed=42).select(range(2000))
    print(f"📊 Loaded {len(dataset)} training examples.")

    print("🧠 Loading Mistral 7B Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # SOTA Fix 1: Lock padding to ID 0 to prevent vocabulary overflow
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # SOTA Fix 2: Manually format to block hidden out-of-bounds tokens
    def format_instruction(example):
        msgs = example["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        assistant = msgs[2]["content"]
        
        # Native Mistral format
        text = f"<s>[INST] System: {system}\n\nUser: {user} [/INST] {assistant}</s>"
        return {"text": text}

    print("🔧 Formatting dataset with native Mistral tags...")
    dataset = dataset.map(format_instruction, remove_columns=["messages"])
    
    # SOTA Fix 3: Drop corrupted rows
    dataset = dataset.filter(lambda x: len(x["text"]) > 10)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print("🧠 Loading Mistral 7B Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        max_steps=200, 
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    print("⚙️ Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("🔥 Starting Training Loop...")
    trainer.train()

    print(f"💾 Saving specialized LoRA adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()
