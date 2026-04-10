"""
shared/vlm_critic.py
---------------------
VLM critic loop using LLaVA-1.5 via Ollama.
Looks at a generated image + original caption, gives structured feedback,
Mistral then replans the prompt, SDXL regenerates.
Max 3 iterations per figure.

Pipeline position:
  prompt_planner → figure_generator → vlm_critic → [replan → regenerate] x3
"""

import base64, json, re, time, requests
from PIL import Image
import io

OLLAMA_URL = "http://localhost:11434"
MAX_ITERATIONS = 3


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for LLaVA."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_llava(image_path: str, caption: str, figure_type: str) -> dict:
    """
    Send image + caption to LLaVA for quality assessment.
    Returns structured feedback dict.
    """
    image_b64 = _image_to_base64(image_path)

    prompt = f"""You are evaluating a generated scientific {figure_type} diagram.
The diagram was generated from this caption:
"{caption[:300]}"

Evaluate the image on these criteria and respond ONLY with a JSON object:
{{
  "quality_score": <integer 1-10>,
  "has_clear_structure": <true/false>,
  "text_readable": <true/false>,
  "matches_caption": <true/false>,
  "issues": ["issue1", "issue2"],
  "improvement": "one specific instruction to improve the next generation"
}}

Be strict. Score 7+ only if the diagram clearly shows the concept from the caption."""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200}
            },
            timeout=60
        )
        raw = resp.json().get("response", "").strip()

        # extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        # fallback if JSON parsing fails
        return {
            "quality_score": 5,
            "has_clear_structure": False,
            "text_readable": False,
            "matches_caption": False,
            "issues": ["Could not parse LLaVA response"],
            "improvement": "Make the diagram cleaner with clearer component boundaries"
        }
    except Exception as e:
        print(f"  [VLM] LLaVA call failed: {e}")
        return {"quality_score": 5, "issues": [], "improvement": ""}


def _replan_prompt(original_plan: dict, feedback: dict, caption: str) -> str:
    """
    Use Mistral to rewrite the SD prompt based on LLaVA feedback.
    Returns improved sd_prompt string.
    """
    issues    = ", ".join(feedback.get("issues", []))
    improve   = feedback.get("improvement", "")
    old_prompt = original_plan.get("sd_prompt", "")

    prompt = f"""You are improving a Stable Diffusion prompt for a scientific diagram.

Original caption: "{caption[:200]}"
Previous SD prompt: "{old_prompt[:300]}"
LLaVA critic issues: {issues}
LLaVA improvement suggestion: {improve}

Write an improved SD prompt that fixes these issues.
The prompt must:
- Be more specific about component placement
- Emphasize white background, clean lines, professional style
- Address the specific issues mentioned
- Be under 200 words

Respond with ONLY the new prompt text, nothing else:"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=30
        )
        return resp.json().get("response", "").strip()
    except Exception:
        return old_prompt


def critic_loop(fig: dict, generate_fn, output_dir: str) -> dict:
    """
    Runs the VLM critic loop for one figure.
    
    Args:
        fig:         planned figure dict (from prompt_planner)
        generate_fn: function(fig, output_dir) → {"path": ..., ...}
        output_dir:  where to save images
    
    Returns:
        best result dict with path to best image
    """
    fig_id      = fig["id"]
    caption     = fig["caption"]
    figure_type = fig.get("figure_type", "architecture")

    print(f"\n[VLM Critic] Starting critic loop for {fig_id}")
    print(f"[VLM Critic] Max iterations: {MAX_ITERATIONS}")

    best_result = None
    best_score  = 0
    current_fig = fig.copy()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  [VLM] Iteration {iteration}/{MAX_ITERATIONS}")

        # Generate image with current prompt
        iter_output_dir = f"{output_dir}/iter_{iteration}"
        result = generate_fn(current_fig, iter_output_dir)

        if not result.get("path"):
            print(f"  [VLM] Generation failed at iteration {iteration}")
            continue

        # Evaluate with LLaVA
        print(f"  [VLM] Evaluating with LLaVA...")
        feedback = _call_llava(result["path"], caption, figure_type)
        score    = feedback.get("quality_score", 0)

        print(f"  [VLM] Score: {score}/10 | Issues: {feedback.get('issues', [])}")

        result["vlm_score"]    = score
        result["vlm_feedback"] = feedback
        result["iteration"]    = iteration

        # Track best
        if score > best_score:
            best_score  = score
            best_result = result.copy()

        # Stop early if quality is good enough
        if score >= 7:
            print(f"  [VLM] Score {score} >= 7. Accepting result.")
            break

        # Replan prompt for next iteration
        if iteration < MAX_ITERATIONS:
            print(f"  [VLM] Score {score} < 7. Replanning prompt...")
            new_prompt = _replan_prompt(current_fig, feedback, caption)
            if new_prompt:
                current_fig = current_fig.copy()
                current_fig["sd_prompt"] = new_prompt
                print(f"  [VLM] New prompt: {new_prompt[:100]}...")
            time.sleep(1)

    # Copy best image to final output location
    if best_result and best_result.get("path"):
        import shutil, os
        final_path = f"{output_dir}/{fig_id}_final.png"
        shutil.copy(best_result["path"], final_path)
        best_result["final_path"] = final_path

        print(f"\n[VLM Critic] Done. Best score: {best_score}/10 "
              f"at iteration {best_result['iteration']}")
        print(f"[VLM Critic] Final image: {final_path}")
    else:
        print(f"[VLM Critic] WARNING: No valid result produced for {fig_id}")

    return best_result or {"id": fig_id, "path": None, "vlm_score": 0}


def run_critic_loop_all(planned_figures: list,
                         generate_fn, output_dir: str) -> list:
    """Run critic loop for all figures."""
    results = []
    for i, fig in enumerate(planned_figures):
        print(f"\n{'='*50}")
        print(f"[VLM] Figure {i+1}/{len(planned_figures)}: {fig['id']}")
        result = critic_loop(fig, generate_fn, output_dir)
        results.append(result)

    scores = [r.get("vlm_score", 0) for r in results if r]
    avg    = sum(scores) / len(scores) if scores else 0
    print(f"\n[VLM] All figures done. Avg score: {avg:.1f}/10")
    return results
