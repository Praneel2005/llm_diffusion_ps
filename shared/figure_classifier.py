"""
shared/figure_classifier.py  —  Layer 2
-----------------------------------------
Uses Mistral 7B (via Ollama) to classify each figure caption into one of 4 types:
    architecture  — neural network diagrams, system pipelines, model overviews
    flowchart     — process flows, decision trees, step-by-step diagrams
    chart         — bar charts, line graphs, plots, tables of results
    conceptual    — abstract concept diagrams, comparison figures, illustrations

This is SEPARATE from the parser (which only extracts).
This is SEPARATE from the planner (which writes SD prompts).

The classifier's job: caption → figure_type label
That label routes to the correct LoRA adapter in Layer 2 Branch A.

Usage:
    from shared.figure_classifier import classify_figures
    figures = [{"id": "fig_0", "caption": "Overview of our 3-stage pipeline..."}]
    classified = classify_figures(figures)
    # classified[0]["figure_type"] == "architecture"
"""

import requests
import json
import time

OLLAMA_URL  = "http://localhost:11434"
MODEL_NAME  = "mistral"

# The 4 valid categories — must match exactly what LoRA adapters are trained for
VALID_TYPES = {"architecture", "flowchart", "chart", "conceptual"}


def _classify_one(caption: str, fig_id: str) -> str:
    """
    Sends one figure caption to Mistral 7B and gets back a figure type.

    Uses a strict prompt that forces Mistral to output exactly one word.
    This is critical — Gemini's mistake was letting Mistral write freeform text
    inside the parser. Here we constrain the output tightly.
    """

    prompt = f"""You are a scientific figure classifier. Your only job is to classify a figure caption into exactly one of these 4 categories:

1. architecture  — neural network diagrams, system pipelines, model architecture overviews, encoder-decoder diagrams
2. flowchart     — process flows, step-by-step workflows, decision trees, training pipelines
3. chart         — bar charts, line graphs, scatter plots, result comparison plots, performance curves
4. conceptual    — abstract concept illustrations, comparison diagrams, motivational figures, visual examples

Figure caption:
"{caption}"

Rules:
- Reply with ONLY one word from this list: architecture, flowchart, chart, conceptual
- No explanation. No punctuation. No other text. Just one word.

Classification:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,   # zero temperature = deterministic
                    "num_predict": 5,     # we only need one word
                }
            },
            timeout=30
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "").strip().lower()

        # clean up — take only the first word in case Mistral adds punctuation
        first_word = raw_output.split()[0].rstrip(".,;:") if raw_output else ""

        if first_word in VALID_TYPES:
            return first_word
        else:
            # if Mistral hallucinates a wrong category, fall back to conceptual
            print(f"[Classifier] WARNING: Mistral returned '{raw_output}' for {fig_id}. Defaulting to 'conceptual'.")
            return "conceptual"

    except Exception as e:
        print(f"[Classifier] ERROR calling Mistral for {fig_id}: {e}")
        print("[Classifier] Defaulting to 'conceptual'.")
        return "conceptual"


def classify_figures(figures: list) -> list:
    """
    Takes a list of figure dicts (from pdf_parser) and adds a figure_type field.

    Input:
        [{"id": "fig_0", "caption": "Overview of our three-stage pipeline..."}, ...]

    Output:
        [{"id": "fig_0", "caption": "...", "figure_type": "architecture"}, ...]
    """
    if not figures:
        print("[Classifier] No figures to classify.")
        return []

    print(f"\n[Classifier] Classifying {len(figures)} figures using Mistral 7B...")

    classified = []
    for i, fig in enumerate(figures):
        print(f"  [{i+1}/{len(figures)}] {fig['id']}: ", end="", flush=True)

        figure_type = _classify_one(
            caption=fig["caption"],
            fig_id=fig["id"]
        )

        print(f"→ {figure_type}")

        classified.append({
            "id":          fig["id"],
            "caption":     fig["caption"],
            "figure_type": figure_type
        })

        # small delay to avoid overwhelming Ollama
        time.sleep(0.5)

    # print summary
    from collections import Counter
    type_counts = Counter(f["figure_type"] for f in classified)
    print(f"\n[Classifier] Summary: {dict(type_counts)}")

    return classified


def save_classified(classified: list, output_path: str):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classified, f, indent=2)
    print(f"[Classifier] Saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Test directly:
# python shared/figure_classifier.py data/extracted/2306.00800v3_extracted.json
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python shared/figure_classifier.py <extracted_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    figures = data.get("figures", [])
    if not figures:
        print("No figures found in JSON. Run pdf_parser.py first.")
        sys.exit(1)

    classified = classify_figures(figures)

    out_path = json_path.replace("_extracted.json", "_classified.json")
    save_classified(classified, out_path)

    print("\n=== CLASSIFICATION RESULTS ===")
    for fig in classified:
        print(f"  [{fig['figure_type'].upper():12s}] {fig['caption'][:70]}...")