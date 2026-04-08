"""
scripts/label_dataset.py
--------------------------
Adds figure_type labels to Paper2Fig100k using Mistral 7B classifier.
Creates labeled_dataset.json — required input for train_lora.py.

Usage:
    # Test on 50 samples first:
    python scripts/label_dataset.py --test 50

    # Full run (run overnight — takes several hours for 100K):
    python scripts/label_dataset.py
"""

import os
import json
import time
import argparse
import requests
from tqdm import tqdm

OLLAMA_URL  = "http://localhost:11434"
MODEL_NAME  = "mistral"
VALID_TYPES = {"architecture", "flowchart", "chart", "conceptual"}

DATA_PATH    = "data/paper2fig100k/"
OUTPUT_PATH  = "data/paper2fig100k/labeled_dataset.json"


def classify_caption(caption: str) -> str:
    """Classify one caption using Mistral."""
    prompt = f"""Classify this scientific figure caption into exactly one category:
architecture, flowchart, chart, conceptual

Caption: "{caption[:300]}"

Reply with ONE word only:"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 5}
            },
            timeout=20
        )
        raw = resp.json().get("response", "").strip().lower().split()[0]
        raw = raw.rstrip(".,;:")
        return raw if raw in VALID_TYPES else "conceptual"
    except Exception:
        return "conceptual"


def load_paper2fig_metadata(data_path: str) -> list:
    """
    Load Paper2Fig100k metadata.
    The dataset has various formats — this handles the most common ones.
    """
    # Try loading captions.json
    for fname in ["captions.json", "metadata.json", "dataset.json"]:
        fpath = os.path.join(data_path, fname)
        if os.path.exists(fpath):
            print(f"[Labeler] Loading metadata from: {fpath}")
            with open(fpath) as f:
                return json.load(f)

    # Try loading captions.csv
    csv_path = os.path.join(data_path, "captions.csv")
    if os.path.exists(csv_path):
        import csv
        print(f"[Labeler] Loading metadata from: {csv_path}")
        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "image_filename": row.get("filename", row.get("image", "")),
                    "caption":        row.get("caption", row.get("text", ""))
                })
        return samples

    # Fallback — scan for images and pair with any txt files
    print("[Labeler] No metadata file found. Scanning for image files...")
    samples = []
    for fname in os.listdir(data_path):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            txt_path = os.path.join(data_path, fname.replace(".png", ".txt").replace(".jpg", ".txt"))
            caption = ""
            if os.path.exists(txt_path):
                with open(txt_path) as f:
                    caption = f.read().strip()
            samples.append({"image_filename": fname, "caption": caption})
    return samples


def run_labeling(test_n: int = None):
    print(f"\n[Labeler] Loading Paper2Fig100k metadata...")
    samples = load_paper2fig_metadata(DATA_PATH)

    if not samples:
        print("[Labeler] ERROR: No samples found in data/paper2fig100k/")
        print("[Labeler] Make sure the dataset is downloaded and extracted.")
        return

    if test_n:
        samples = samples[:test_n]
        print(f"[Labeler] TEST MODE: Processing only {test_n} samples")
    else:
        print(f"[Labeler] Full run: {len(samples)} samples")

    labeled    = []
    type_counts = {"architecture": 0, "flowchart": 0, "chart": 0, "conceptual": 0}

    for sample in tqdm(samples, desc="Labeling figures"):
        caption = sample.get("caption", "")
        if not caption:
            figure_type = "conceptual"
        else:
            figure_type = classify_caption(caption)

        type_counts[figure_type] = type_counts.get(figure_type, 0) + 1

        labeled.append({
            "image_filename": sample.get("image_filename", ""),
            "caption":        caption,
            "figure_type":    figure_type
        })

        time.sleep(0.1)

    # save results
    out_path = OUTPUT_PATH
    if test_n:
        out_path = OUTPUT_PATH.replace(".json", f"_test{test_n}.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(labeled, f, indent=2)

    print(f"\n[Labeler] Done. Saved to: {out_path}")
    print(f"[Labeler] Distribution: {type_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None,
                        help="Number of samples to process (default: all)")
    args = parser.parse_args()
    run_labeling(test_n=args.test)