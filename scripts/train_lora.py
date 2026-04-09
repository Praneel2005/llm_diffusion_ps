"""
scripts/train_lora.py — SDXL LoRA Training (NaN loss fixed)
"""

import os, json, argparse, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VALID_TYPES   = ["architecture", "flowchart", "chart", "conceptual"]


class FigureDataset(Dataset):
    def __init__(self, data_path, labeled_json, figure_type, image_size=512):
        self.data_path  = data_path
        self.image_size = image_size
        self.samples    = []
        print(f"[Dataset] Loading {figure_type} samples...")
        with open(labeled_json) as f:
            all_data = json.load(f)
        for item in all_data:
            if item.get("figure_type") != figure_type:
                continue
            img_path = os.path.join(data_path, "figures",
                                    item.get("image_filename", "") + ".png")
            caption  = item.get("caption", "")
            if os.path.exists(img_path) and caption:
                self.samples.append({"image_path": img_path, "caption": caption})
        print(f"[Dataset] Found {len(self.samples)} {figure_type} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        return {"pixel_values": transform(image), "caption": s["caption"]}


def encode_prompt_sdxl(prompt_batch, tokenizer_1, tokenizer_2,
                        text_encoder_1, text_encoder_2, device):
    tokens_1 = tokenizer_1(
        prompt_batch, padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    tokens_2 = tokenizer_2(
        prompt_batch, padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        enc_1_out = text_encoder_1(tokens_1, output_hidden_states=True)
        enc_2_out = text_encoder_2(tokens_2, output_hidden_states=True)
        hidden_1  = enc_1_out.hidden_states[-2]   # (B, 77, 768)
        hidden_2  = enc_2_out.hidden_states[-2]   # (B, 77, 1280)
        encoder_hidden_states = torch.cat([hidden_1, hidden_2], dim=-1)  # (B, 77, 2048)
        pooled_embeds = enc_2_out[0]               # (B, 1280)

    return encoder_hidden_states, pooled_embeds


def train_lora(figure_type, data_path, labeled_json, output_dir,
               max_steps=1500, batch_size=1, lr=1e-4, save_every=500):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[LoRA] Device: {device} | Type: {figure_type} | Steps: {max_steps}")

    dataset = FigureDataset(data_path, labeled_json, figure_type, image_size=512)
    if len(dataset) == 0:
        print("[LoRA] ERROR: No samples found.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    print("[LoRA] Loading SDXL components...")
    # Load in float32 — we use autocast for mixed precision, not forced float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)

    unet            = pipe.unet
    vae             = pipe.vae
    tokenizer_1     = pipe.tokenizer
    tokenizer_2     = pipe.tokenizer_2
    text_encoder_1  = pipe.text_encoder
    text_encoder_2  = pipe.text_encoder_2
    noise_scheduler = DDPMScheduler.from_pretrained(SDXL_MODEL_ID, subfolder="scheduler")

    # Freeze everything except LoRA layers
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1, bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    # GradScaler for stable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[LoRA] Training for {max_steps} steps...")

    step, losses = 0, []
    data_iter = iter(dataloader)

    with tqdm(total=max_steps, desc=f"Training {figure_type} LoRA") as pbar:
        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Keep pixel values in float32 for VAE encoding
            pixel_values = batch["pixel_values"].to(device)
            captions     = batch["caption"]

            # Encode images → latents (float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text (float32)
            encoder_hidden_states, pooled_embeds = encode_prompt_sdxl(
                list(captions), tokenizer_1, tokenizer_2,
                text_encoder_1, text_encoder_2, device
            )

            # SDXL conditioning
            bs = latents.shape[0]
            add_time_ids = torch.tensor(
                [[512, 512, 0, 0, 512, 512]], dtype=torch.float32, device=device
            ).repeat(bs, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids":    add_time_ids,
            }

            # Forward pass with autocast — prevents NaN from float16 overflow
            with torch.cuda.amp.autocast():
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            # Check for NaN before stepping
            if torch.isnan(loss):
                print(f"\n[LoRA] WARNING: NaN loss at step {step}, skipping batch.")
                optimizer.zero_grad()
                step += 1
                pbar.update(1)
                continue

            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if step % save_every == 0 or step == max_steps:
                ckpt = os.path.join(output_dir, f"checkpoint_{step}")
                unet.save_pretrained(ckpt)
                valid_losses = [l for l in losses[-50:] if l == l]
                avg = sum(valid_losses) / len(valid_losses) if valid_losses else float('nan')
                print(f"\n[LoRA] Checkpoint: {ckpt} | Avg loss: {avg:.4f}")

    final = os.path.join(output_dir, "final")
    unet.save_pretrained(final)
    valid = [l for l in losses if l == l]
    print(f"\n[LoRA] Done. Final: {final}")
    print(f"[LoRA] Final loss: {valid[-1]:.4f}" if valid else "[LoRA] All losses were NaN.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure_type",  required=True, choices=VALID_TYPES)
    parser.add_argument("--data_path",    default="data/paper2fig100k/")
    parser.add_argument("--labeled_json", default="data/paper2fig100k/labeled_dataset.json")
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--max_steps",    type=int,   default=1500)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--save_every",   type=int,   default=500)
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
