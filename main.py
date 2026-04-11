"""
main.py — Layer 2 Pipeline
---------------------------
Full pipeline: PDF → Parse → Classify → Plan → Generate Figures → Video
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.pdf_parser       import parse_pdf, save_extracted
from shared.figure_classifier import classify_figures
from shared.prompt_planner   import plan_all_figures, save_planned
from branch_a.diagram_renderer import run_renderer as run_branch_a
from branch_b.video_generator  import run_branch_b


def run_full_pipeline(pdf_path: str):
    print("=" * 60)
    print("  LLM-GUIDED MULTIMODAL DIFFUSION FRAMEWORK")
    print("  Layer 2 Pipeline")
    print("=" * 60)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs("data/extracted",   exist_ok=True)
    os.makedirs("outputs/figures",  exist_ok=True)
    os.makedirs("outputs/videos",   exist_ok=True)

    # ── STEP 1: Parse PDF ──────────────────────────────────────────
    print("\n>>> STEP 1: Parsing PDF with GROBID...")
    paper_data = parse_pdf(pdf_path)

    extracted_path = f"data/extracted/{base_name}_extracted.json"
    save_extracted(paper_data, extracted_path)

    print(f"    Abstract : {len(paper_data['abstract'])} chars")
    print(f"    Figures  : {len(paper_data['figures'])}")
    print(f"    Tables   : {paper_data['tables_skipped']} skipped")

    if not paper_data["figures"]:
        print("\n[Pipeline] WARNING: No figures found. Check PDF is text-based.")

    # ── STEP 2: Classify figures ───────────────────────────────────
    print("\n>>> STEP 2: Classifying figures with Mistral 7B...")
    classified = classify_figures(paper_data["figures"])

    classified_path = f"data/extracted/{base_name}_classified.json"
    with open(classified_path, "w") as f:
        json.dump(classified, f, indent=2)
    print(f"    Saved: {classified_path}")

    # ── STEP 3: Plan layouts ───────────────────────────────────────
    print("\n>>> STEP 3: Planning spatial layouts with Mistral 7B...")
    planned = plan_all_figures(classified)

    planned_path = f"data/extracted/{base_name}_planned.json"
    save_planned(planned, planned_path)

    # ── STEP 4: Branch A — Generate figures ───────────────────────
    print("\n>>> STEP 4: Branch A — Generating figures with SDXL...")
    figures_dir = f"outputs/figures/{base_name}"
    generated_figures = run_branch_a(planned, figures_dir)
    print(f"    Generated {len(generated_figures)} figures → {figures_dir}/")

    # ── STEP 5: Branch B — Generate video ─────────────────────────
    print("\n>>> STEP 5: Branch B — Generating video abstract...")
    video_path = run_branch_b(
        abstract=paper_data["abstract"],
        introduction=paper_data["introduction"],
        conclusion=paper_data["conclusion"],
        generated_figures=generated_figures,
        output_dir=f"outputs/videos/{base_name}"
    )

    # ── DONE ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Figures : outputs/figures/{base_name}/")
    print(f"  Video   : {video_path}")
    print(f"  Data    : data/extracted/{base_name}_*.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: File not found: {args.pdf}")
        sys.exit(1)

    run_full_pipeline(args.pdf)
