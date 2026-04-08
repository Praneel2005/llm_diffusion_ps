"""
scripts/train_lora.py  —  Week 2 LoRA Training
------------------------------------------------
Fine-tunes one LoRA adapter on SDXL for a specific figure type.
Run 4 times (once per type): architecture, flowchart, chart, conceptual.

Uses HuggingFace PEFT + Diffusers for LoRA fine-tuning.

Usage:
    # Test run (10 steps — verify it works without crashing):
    python scripts/train_lora.py --figure_type architecture --max_steps 10

    # Full training (run inside tmux):
    python scripts/train_lora.py --figure_type architecture --max_steps 1500

Training order recommendation (easiest → hardest):
    1. architecture  (clearest visual patterns)
    2. chart         (well-defined structure)
    3. flowchart     (directional flow)
    4. conceptual    (most abstract — train last)
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VALID_TYPES   = ["architecture", "flowchart", "chart", "conceptual"]
IMAGE_SIZE    = 1024


# ─── DATASET ──────────────────────────────────────────────────────────

class FigureDataset(Dataset):
    """
    Loads Paper2Fig100k figures filtered to one figure type.
    Expects: data_path/ with images, labeled_json with type labels.
    """

    def __init__(self, data_path: str, labeled_json: str,
                 figure_type: str, image_size: int = 512):
        self.data_path   = data_path
        self.image_size  = image_size
        self.figure_type = figure_type
        self.samples     = []

        print(f"[Dataset] Loading {figure_type} samples from {labeled_json}")

        with open(labeled_json) as f:
            all_data = json.load(f)

        # filter to this figure type only
        for item in all_data:
            if item.get("figure_type") != figure_type:
                continue

            img_path = os.path.join(data_path, item.get("image_filename", ""))
            caption  = item.get("caption", "")

            if os.path.exists(img_path) and caption:
                self.samples.append({
                    "image_path": img_path,
                    "caption":    caption
                })

        print(f"[Dataset] Found {len(self.samples)} {figure_type} samples.")

        if len(self.samples) < 10:
            print(f"[Dataset] WARNING: Very few samples ({len(self.samples)}).")
            print(f"[Dataset] Run label_dataset.py first to generate labeled_dataset.json")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load and resize image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # convert to tensor [-1, 1]
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        image_tensor = transform(image)

        return {
            "pixel_values": image_tensor,
            "caption":      sample["caption"]
        }


# ─── TRAINING ─────────────────────────────────────────────────────────

def train_lora(
    figure_type:  str,
    data_path:    str,
    labeled_json: str,
    output_dir:   str,
    max_steps:    int  = 1500,
    batch_size:   int  = 1,
    lr:           float = 1e-4,
    save_every:   int  = 500,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[LoRA Trainer] Device: {device}")
    print(f"[LoRA Trainer] Figure type: {figure_type}")
    print(f"[LoRA Trainer] Max steps: {max_steps}")

    if device == "cpu":
        print("[LoRA Trainer] WARNING: Training on CPU is extremely slow.")
        print("[LoRA Trainer] Expect hours per step. Use GPU for real training.")

    # load dataset
    dataset = FigureDataset(
        data_path=data_path,
        labeled_json=labeled_json,
        figure_type=figure_type,
        image_size=512   # use 512 for training to save memory
    )

    if len(dataset) == 0:
        print("[LoRA Trainer] ERROR: No samples found. Cannot train.")
        print("[LoRA Trainer] Run: python scripts/label_dataset.py first")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda")
    )

    print(f"\n[LoRA Trainer] Loading SDXL UNet for LoRA...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    unet      = pipe.unet
    vae       = pipe.vae
    tokenizer = pipe.tokenizer
    text_enc  = pipe.text_encoder

    # freeze everything except LoRA layers
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)

    # configure LoRA
    lora_config = LoraConfig(
        r=16,                         # LoRA rank — higher = more capacity
        lora_alpha=32,                # scaling factor
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # attention layers
        lora_dropout=0.1,
        bias="none"
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps
    )

    noise_scheduler = pipe.scheduler
    os.makedirs(output_dir, exist_ok=True)

    # training loop
    print(f"\n[LoRA Trainer] Starting training: {max_steps} steps")
    print(f"[LoRA Trainer] Save checkpoint every {save_every} steps")
    print("-" * 50)

    step      = 0
    data_iter = iter(dataloader)
    losses    = []

    with tqdm(total=max_steps, desc=f"Training {figure_type} LoRA") as pbar:
        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pixel_values = batch["pixel_values"].to(device)
            if device == "cuda":
                pixel_values = pixel_values.to(torch.float16)

            # encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # add noise
            noise      = torch.randn_like(latents)
            timesteps  = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # encode text
            with torch.no_grad():
                text_inputs = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)
                encoder_hidden_states = text_enc(text_inputs)[0]

            # predict noise
            added_cond_kwargs = {}
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # save checkpoint
            if step % save_every == 0 or step == max_steps:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_{step}")
                unet.save_pretrained(checkpoint_dir)
                print(f"\n[LoRA Trainer] Checkpoint saved: {checkpoint_dir}")
                print(f"[LoRA Trainer] Average loss (last 50): "
                      f"{sum(losses[-50:]) / min(len(losses), 50):.4f}")

    # save final adapter
    final_dir = os.path.join(output_dir, "final")
    unet.save_pretrained(final_dir)
    print(f"\n[LoRA Trainer] Final adapter saved: {final_dir}")
    print(f"[LoRA Trainer] Training complete. Final loss: {losses[-1]:.4f}")

    # push to HuggingFace
    print("\n[LoRA Trainer] Pushing weights to HuggingFace for backup...")
    os.system("./push_weights.sh")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapter for SDXL")

    parser.add_argument("--figure_type",  required=True,
                        choices=VALID_TYPES,
                        help="Figure type to train: architecture/flowchart/chart/conceptual")
    parser.add_argument("--data_path",    default="data/paper2fig100k/",
                        help="Path to Paper2Fig100k image folder")
    parser.add_argument("--labeled_json", default="data/paper2fig100k/labeled_dataset.json",
                        help="Path to labeled dataset JSON from label_dataset.py")
    parser.add_argument("--output_dir",   default=None,
                        help="Where to save LoRA weights (default: lora_weights/<type>/)")
    parser.add_argument("--max_steps",    type=int, default=1500,
                        help="Training steps (use 10 for testing, 1500 for real training)")
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--save_every",   type=int, default=500)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"lora_weights/{args.figure_type}/"

    train_lora(
        figure_type=args.figure_type,
        data_path=args.data_path,
        labeled_json=args.labeled_json,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
    )