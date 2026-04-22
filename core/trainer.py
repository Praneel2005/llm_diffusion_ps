import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

def run_overfit_test():
    print("🚀 Initializing Mistral QLoRA Overfit Test (BF16 Stability Fix)...")
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir = "/home/drive3/llm_diffusion_ps/models/mistral_spatial_overfit"
    data_path = "/home/drive3/llm_diffusion_ps/data/ai2d_mistral_train.jsonl"

    dataset = load_dataset("json", data_files=data_path, split="train")

    # Use bfloat16 for computation - it's more stable for Mistral than float16
    compute_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=2e-4,
        bf16=True,        # Use BF16 training
        fp16=False,       # Disable FP16 to avoid the GradScaler error
        max_steps=15,
        save_strategy="no",
        report_to="none"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    def formatting_func(example):
        user_msg = example['messages'][0]['content']
        asst_msg = example['messages'][1]['content']
        return [f"<s>[INST] {user_msg} [/INST] {asst_msg} </s>"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
        args=training_args
    )

    print("🔥 Starting fast training loop in BF16...")
    trainer.train()
    
    trainer.model.save_pretrained(output_dir)
    print(f"✅ Overfit test complete. Adapter saved to {output_dir}")

if __name__ == "__main__":
    run_overfit_test()
