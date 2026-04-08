"""
scripts/label_dataset.py
Adds figure_type labels to Paper2Fig100k using Mistral 7B.
Handles the actual dataset format: train.json / test.json with 'captions' list.
"""

import os, json, time, argparse, requests
from tqdm import tqdm

OLLAMA_URL   = "http://localhost:11434"
MODEL_NAME   = "mistral"
VALID_TYPES  = {"architecture", "flowchart", "chart", "conceptual"}
DATA_PATH    = "data/paper2fig100k/"
OUTPUT_PATH  = "data/paper2fig100k/labeled_dataset.json"


def classify_caption(caption: str) -> str:
    prompt = f"""Classify this scientific figure caption into exactly one category:
architecture, flowchart, chart, conceptual

Caption: "{caption[:300]}"

Reply with ONE word only:"""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.0, "num_predict": 5}},
            timeout=20
        )
        raw = resp.json().get("response", "").strip().lower().split()[0].rstrip(".,;:")
        return raw if raw in VALID_TYPES else "conceptual"
    except Exception:
        return "conceptual"


def load_samples(data_path: str) -> list:
    """Load Paper2Fig100k from its actual format: train.json with 'captions' list."""
    for fname in ["train.json", "test.json"]:
        fpath = os.path.join(data_path, fname)
        if os.path.exists(fpath):
            print(f"[Labeler] Loading from: {fpath}")
            with open(fpath) as f:
                raw = json.load(f)
            samples = []
            # raw is a list of dicts OR a dict keyed by figure_id
            items = raw if isinstance(raw, list) else list(raw.values())
            for item in items:
                fig_id  = item.get("figure_id", "")
                captions = item.get("captions", [])
                # captions is a list of strings — join them
                caption = " ".join(captions) if isinstance(captions, list) else str(captions)
                if caption.strip():
                    samples.append({
                        "image_filename": fig_id,
                        "caption": caption.strip()
                    })
            print(f"[Labeler] Loaded {len(samples)} samples.")
            return samples

    print("[Labeler] ERROR: train.json / test.json not found in", data_path)
    print("[Labeler] Make sure Paper2Fig100k.tar.gz is extracted.")
    return []


def run_labeling(test_n: int = None):
    samples = load_samples(DATA_PATH)
    if not samples:
        return

    if test_n:
        samples = samples[:test_n]
        print(f"[Labeler] TEST MODE: {test_n} samples")

    labeled     = []
    type_counts = {t: 0 for t in VALID_TYPES}

    for sample in tqdm(samples, desc="Labeling"):
        figure_type = classify_caption(sample["caption"])
        type_counts[figure_type] = type_counts.get(figure_type, 0) + 1
        labeled.append({**sample, "figure_type": figure_type})
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
