import json, os, time, requests

OLLAMA_URL  = "http://localhost:11434"
MODEL_NAME  = "mistral"
VALID_TYPES = {"architecture", "flowchart", "chart", "conceptual"}

WORD_MAP = {
    "illustration": "conceptual",
    "concept":      "conceptual",
    "diagram":      "architecture",
    "network":      "architecture",
    "plot":         "chart",
    "graph":        "chart",
    "table":        "chart",
    "results":      "chart",
    "algorithm":    "flowchart",
    "procedure":    "flowchart",
}

def _classify_one(caption: str, fig_id: str) -> str:
    prompt = f"""Classify this scientific figure caption into one of four types.

Types:
- architecture: model diagram, neural network, system overview, pipeline, framework, encoder-decoder
- flowchart: step-by-step algorithm, training procedure, decision tree, process flow
- chart: bar chart, line graph, accuracy/loss curves, experimental results, performance numbers
- conceptual: abstract concept illustration, motivation figure, intuition diagram

Caption: "{caption[:280]}"

Respond with exactly one word from: architecture, flowchart, chart, conceptual"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 6}},
            timeout=20
        )
        raw  = resp.json().get("response", "").strip().lower()
        word = raw.split()[0].rstrip(".,;:()") if raw.split() else ""
        if word in VALID_TYPES:  return word
        if word in WORD_MAP:     return WORD_MAP[word]
        for vtype in VALID_TYPES:
            if vtype in raw:     return vtype
        return "conceptual"
    except Exception:
        return "conceptual"

def classify_figures(figures):
    if not figures: return []
    print(f"[Classifier] Classifying {len(figures)} figures...")
    results = []
    for i, fig in enumerate(figures):
        ft = _classify_one(fig["caption"], fig["id"])
        results.append({**fig, "figure_type": ft})
        print(f"  [{i+1}/{len(figures)}] {fig['id']}: → {ft}")
        time.sleep(0.1)
    counts = {}
    for r in results: counts[r["figure_type"]] = counts.get(r["figure_type"], 0) + 1
    print(f"[Classifier] Summary: {counts}")
    return results

def save_classified(classified, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f: json.dump(classified, f, indent=2)
    print(f"[Classifier] Saved: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: print("Usage: python shared/figure_classifier.py <json>"); sys.exit(1)
    with open(sys.argv[1]) as f: data = json.load(f)
    figures = data if isinstance(data, list) else data.get("figures", [])
    results = classify_figures(figures)
    save_classified(results, sys.argv[1].replace("_extracted.json", "_classified.json"))
