import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.pdf_parser import parse_paper, save_extraction
from branch_a.figure_generator import run_branch_a
from branch_b.video_generator import run_branch_b

def run_full_pipeline(pdf_path: str):
    print("=" * 60 + "\n  LLM-GUIDED MULTIMODAL DIFFUSION FRAMEWORK\n  Layer 1 Demo\n" + "=" * 60)
    
    print("\n>>> STEP 1: Parsing PDF...")
    paper_data = parse_paper(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_extraction(paper_data, f"data/extracted/{base_name}.json")

    print("\n>>> STEP 2: Branch A — Generating scientific figures...")
    generated_figures = run_branch_a(paper_data["figure_captions"], f"outputs/figures/{base_name}")

    print("\n>>> STEP 3: Branch B — Generating video abstract...")
    video_path = run_branch_b(paper_data["abstract"], paper_data["introduction"], paper_data["conclusion"], generated_figures, f"outputs/videos/{base_name}")

    print("\n" + "=" * 60 + "\n  PIPELINE COMPLETE\n" + "=" * 60)
    print(f"\n  Video abstract: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="Path to input research paper PDF")
    args = parser.parse_args()
    run_full_pipeline(args.pdf)
