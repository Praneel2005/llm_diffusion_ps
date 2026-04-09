"""
scripts/label_dataset.py
Uses the fixed figure_classifier prompt for consistent labeling.
"""

import os, json, time, argparse, sys
sys.path.insert(0, '/home/drive3/llm_diffusion_ps')
from shared.figure_classifier import _classify_one
from tqdm import tqdm

DATA_PATH   = "data/paper2fig100k/"
OUTPUT_PATH = "data/paper2fig100k/labeled_dataset.json"


def load_samples(data_path: str) -> list:
    for fname in ["paper2fig_train.json", "train.json"]:
        fpath = os.path.join(data_path, fname)
        if os.path.exists(fpath):
            print(f"[Labeler] Loading: {fpath}")
            with open(fpath) as f:
                raw = json.load(f)
            items = raw if isinstance(raw, list) else list(raw.values())
            samples = []
            for item in items:
                captions = item.get("captions", [])
                caption  = " ".join(captions) if isinstance(captions, list) else str(captions)
                if caption.strip():
                    samples.append({
                        "image_filename": item.get("figure_id", ""),
                        "caption": caption.strip()
                    })
            print(f"[Labeler] Loaded {len(samples)} samples.")
            return samples
    print("[Labeler] ERROR: No train JSON found.")
    return []


def run_labeling(test_n=None):
    samples = load_samples(DATA_PATH)
    if not samples:
        return
    if test_n:
        samples = samples[:test_n]
        print(f"[Labeler] TEST MODE: {test_n} samples")

    labeled     = []
    type_counts = {"architecture": 0, "flowchart": 0, "chart": 0, "conceptual": 0}

    for sample in tqdm(samples, desc="Labeling"):
        fig_type = _classify_one(sample["caption"], sample["image_filename"])
        type_counts[fig_type] = type_counts.get(fig_type, 0) + 1
        labeled.append({**sample, "figure_type": fig_type})
        time.sleep(0.05)

    out = OUTPUT_PATH if not test_n else OUTPUT_PATH.replace(".json", f"_test{test_n}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(labeled, f, indent=2)

    print(f"\n[Labeler] Saved: {out}")
    print(f"[Labeler] Distribution: {type_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None)
    args = parser.parse_args()
    run_labeling(test_n=args.test)
