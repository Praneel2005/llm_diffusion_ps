"""
shared/prompt_planner.py  —  Layer 2
--------------------------------------
Takes classified figure data and uses Mistral 7B to generate:
  1. A structured JSON layout (entities + bounding boxes + relations)
  2. A clean sd_prompt for SDXL image generation

This implements the DiagrammerGPT methodology locally with open models.
The planner is intentionally SEPARATE from the classifier and parser.

Pipeline position:
  pdf_parser → figure_classifier → prompt_planner → figure_generator
"""

import json
import os
import re
import time
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "mistral"

# Type-specific prompt templates
# Each type gets different instructions because the spatial layout
# rules for an architecture diagram differ from a bar chart
TYPE_PROMPTS = {
    "architecture": """You are planning a scientific architecture diagram.
The diagram shows components connected in a pipeline or network.
Create a layout with 3-6 key components arranged left-to-right or top-to-bottom.
Use arrows to show data/information flow between components.""",

    "flowchart": """You are planning a scientific flowchart.
The diagram shows steps in a process or algorithm.
Create a layout with 4-7 sequential steps arranged top-to-bottom.
Use arrows to show the flow direction between steps.""",

    "chart": """You are planning a scientific result chart.
The diagram shows experimental results or comparisons.
Create a layout with a title, x-axis label, y-axis label, and 2-4 data series.
Use bars or lines to represent the data.""",

    "conceptual": """You are planning a scientific conceptual diagram.
The diagram illustrates an abstract idea or comparison.
Create a layout with 2-5 key concepts arranged to show their relationships.
Use arrows or lines to show connections between concepts."""
}


def _call_mistral(prompt: str, max_retries: int = 3) -> str:
    """Call Mistral via Ollama with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 500,
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"  [Planner] Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return ""


def _extract_json(text: str) -> dict:
    """
    Extract JSON from Mistral's response.
    Mistral sometimes wraps JSON in markdown code blocks — this handles that.
    """
    # try to find JSON block between ```json and ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # try to find raw JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _make_fallback_plan(caption: str, figure_type: str) -> dict:
    """
    If Mistral fails to return valid JSON, use a safe fallback plan.
    This ensures the pipeline never breaks even if Mistral misbehaves.
    """
    return {
        "entities": [
            {"id": "E0", "label": "Input",  "bbox": [10, 40, 20, 20]},
            {"id": "E1", "label": "Process","bbox": [40, 40, 20, 20]},
            {"id": "E2", "label": "Output", "bbox": [70, 40, 20, 20]}
        ],
        "relations": [
            {"from": "E0", "to": "E1", "type": "arrow"},
            {"from": "E1", "to": "E2", "type": "arrow"}
        ],
        "sd_prompt": (
            f"A clean scientific {figure_type} diagram. {caption[:200]}. "
            f"White background, professional research paper style, clear labels."
        ),
        "fallback_used": True
    }


def plan_figure(fig: dict) -> dict:
    """
    Plans the spatial layout for one figure.

    Input:
        {"id": "fig_0", "caption": "...", "figure_type": "architecture"}

    Output:
        {"id": "fig_0", "caption": "...", "figure_type": "...",
         "entities": [...], "relations": [...], "sd_prompt": "..."}
    """
    fig_id      = fig["id"]
    caption     = fig["caption"]
    figure_type = fig.get("figure_type", "conceptual")

    type_instruction = TYPE_PROMPTS.get(figure_type, TYPE_PROMPTS["conceptual"])

    prompt = f"""{type_instruction}

Figure caption: "{caption}"

Generate a spatial layout plan as a JSON object with exactly these fields:
{{
  "entities": [
    {{"id": "E0", "label": "component name", "bbox": [x, y, width, height]}}
  ],
  "relations": [
    {{"from": "E0", "to": "E1", "type": "arrow"}}
  ],
  "sd_prompt": "A detailed image generation prompt describing the visual appearance"
}}

Rules:
- bbox values are integers from 0-100 (percentage of image size)
- x + width must not exceed 100
- y + height must not exceed 100
- entities must not overlap
- sd_prompt must mention: white background, scientific diagram, professional style
- Respond ONLY with the JSON object. No explanation before or after.

JSON:"""

    print(f"  [Planner] Planning layout for {fig_id} ({figure_type})...", flush=True)
    raw_response = _call_mistral(prompt)

    plan = _extract_json(raw_response)

    if not plan or "entities" not in plan or "sd_prompt" not in plan:
        print(f"  [Planner] WARNING: Mistral returned invalid JSON for {fig_id}. Using fallback.")
        plan = _make_fallback_plan(caption, figure_type)

    # merge with original fig data
    result = {
        "id":          fig_id,
        "caption":     caption,
        "figure_type": figure_type,
        "entities":    plan.get("entities", []),
        "relations":   plan.get("relations", []),
        "sd_prompt":   plan.get("sd_prompt", ""),
        "fallback_used": plan.get("fallback_used", False)
    }

    return result


def plan_all_figures(classified_figures: list) -> list:
    """
    Plans layouts for all classified figures.

    Input:  list from figure_classifier
    Output: list with entities, relations, sd_prompt added to each
    """
    if not classified_figures:
        print("[Planner] No figures to plan.")
        return []

    print(f"\n[Planner] Planning layouts for {len(classified_figures)} figures...")

    planned = []
    for i, fig in enumerate(classified_figures):
        result = plan_figure(fig)
        planned.append(result)

        fallback_note = " (fallback)" if result.get("fallback_used") else ""
        print(f"  [{i+1}/{len(classified_figures)}] {result['id']}: "
              f"{len(result['entities'])} entities, "
              f"{len(result['relations'])} relations{fallback_note}")

        time.sleep(0.5)

    fallback_count = sum(1 for p in planned if p.get("fallback_used"))
    print(f"\n[Planner] Done. {len(planned)} figures planned. "
          f"Fallbacks used: {fallback_count}/{len(planned)}")

    return planned


def save_planned(planned: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(planned, f, indent=2, ensure_ascii=False)
    print(f"[Planner] Saved to: {output_path}")


def print_plan_summary(planned: list):
    print("\n=== PLANNING SUMMARY ===")
    for fig in planned:
        print(f"\n[{fig['id']}] Type: {fig['figure_type']}")
        print(f"  Entities ({len(fig['entities'])}):")
        for e in fig['entities']:
            print(f"    {e['id']}: '{e['label']}' at {e['bbox']}")
        print(f"  Relations ({len(fig['relations'])}):")
        for r in fig['relations']:
            print(f"    {r['from']} → {r['to']} ({r['type']})")
        print(f"  SD Prompt: {fig['sd_prompt'][:100]}...")
        if fig.get('fallback_used'):
            print(f"  ⚠ Fallback plan used")


# ─────────────────────────────────────────────────────────────────────
# Test directly:
# python shared/prompt_planner.py data/extracted/2306.00800v3_classified.json
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python shared/prompt_planner.py <classified_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        classified = json.load(f)

    planned = plan_all_figures(classified)

    out_path = json_path.replace("_classified.json", "_planned.json")
    save_planned(planned, out_path)
    print_plan_summary(planned)