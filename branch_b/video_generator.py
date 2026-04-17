"""
branch_b/video_generator.py  —  Animated Video Abstract Generator
------------------------------------------------------------------
Produces a 70-second video abstract where each scene has its own
distinct visual with progressive animations:

Scene 1 — Problem text fades in line by line
Scene 2 — Prior work diagram with red X crossing it out  
Scene 3 — Method diagram: boxes and arrows appear one by one
Scene 4 — Results bar chart: bars grow from zero
Scene 5 — Full overview with glowing highlight effect

Usage:
    python branch_b/video_generator.py data/extracted/paper_extracted.json
    python branch_b/video_generator.py --pdf data/raw_pdfs/paper.pdf
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, re, time, shutil, subprocess, textwrap, requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OLLAMA_URL = "http://localhost:11434"
OUTPUT_DIR = "outputs/videos"
FPS        = 24

# Scene definitions
SCENES = [
    {"id": "scene_1", "title": "The Problem",  "duration": 15, "type": "text_reveal"},
    {"id": "scene_2", "title": "Prior Work",   "duration": 15, "type": "prior_work"},
    {"id": "scene_3", "title": "Our Method",   "duration": 20, "type": "method_build"},
    {"id": "scene_4", "title": "Results",      "duration": 12, "type": "bar_chart"},
    {"id": "scene_5", "title": "Impact",       "duration": 8,  "type": "full_reveal"},
]

# Color palette
C = {
    "blue":       "#2E6DA4",
    "blue_light": "#D6E8FA",
    "green":      "#2E7D32",
    "green_light":"#D9EDD9",
    "red":        "#C62828",
    "red_light":  "#FFEBEE",
    "amber":      "#E65100",
    "amber_light":"#FFF3E0",
    "gray":       "#546E7A",
    "gray_light": "#ECEFF1",
    "white":      "#FFFFFF",
    "black":      "#1A1A2E",
    "bg":         "#F8F9FA",
}


# ═══════════════════════════════════════════════════════════
# MISTRAL CALLS
# ═══════════════════════════════════════════════════════════

def _call_mistral(prompt: str, max_tokens: int = 500) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "mistral", "prompt": prompt,
                  "stream": False,
                  "options": {"temperature": 0.2, "num_predict": max_tokens}},
            timeout=60
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [Mistral] Call failed: {e}")
        return ""


def _extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def understand_paper(abstract: str, intro: str, conclusion: str) -> dict:
    print("\n[Branch B] Understanding paper...")
    raw = _call_mistral(f"""Extract information from this research paper.

ABSTRACT: {abstract[:500]}
INTRODUCTION: {intro[:300]}
CONCLUSION: {conclusion[:300]}

Respond ONLY with JSON:
{{
  "paper_title": "5-8 word title",
  "problem_lines": ["line 1 of problem (max 10 words)", "line 2 (max 10 words)"],
  "prior_limitation": "what prior work was missing (one sentence)",
  "method_name": "2-4 word method name",
  "components": ["component 1", "component 2", "component 3", "component 4"],
  "results": [{{"label": "Ours", "value": 0.92}}, {{"label": "Baseline", "value": 0.78}}, {{"label": "Prior SOTA", "value": 0.85}}],
  "impact_line": "one sentence on why this matters"
}}""", 500)

    data = _extract_json(raw)
    if not data:
        return {
            "paper_title": "Research Paper Overview",
            "problem_lines": ["Current methods lack accuracy", "and require manual effort"],
            "prior_limitation": "Prior methods did not generalize well.",
            "method_name": "Proposed Method",
            "components": ["Input", "Encoder", "Processor", "Output"],
            "results": [{"label": "Ours", "value": 0.92},
                        {"label": "Baseline", "value": 0.74},
                        {"label": "Prior SOTA", "value": 0.83}],
            "impact_line": "This work enables new possibilities in the field."
        }

    print(f"  Title: {data.get('paper_title','?')}")
    print(f"  Method: {data.get('method_name','?')}")
    return data


def write_script(understanding: dict) -> dict:
    print("\n[Branch B] Writing narration script...")
    raw = _call_mistral(f"""Write narration for a 70-second research paper video.

Paper: {understanding.get('paper_title','')}
Problem: {' '.join(understanding.get('problem_lines',[]))}
Prior issue: {understanding.get('prior_limitation','')}
Method: {understanding.get('method_name','')}
Components: {', '.join(understanding.get('components',[]))}
Impact: {understanding.get('impact_line','')}

Respond ONLY with JSON (word counts matter for timing):
{{
  "scene_1": "35 words about the problem this paper solves",
  "scene_2": "35 words about what prior methods were missing",
  "scene_3": "47 words explaining the proposed method and its components",
  "scene_4": "28 words describing the quantitative results achieved",
  "scene_5": "19 words on the broader impact and future directions"
}}""", 500)

    data = _extract_json(raw)
    if not data:
        u = understanding
        return {
            "scene_1": f"Research today faces a critical challenge. {' '.join(u.get('problem_lines', ['Current methods lack efficiency and accuracy.']))} This paper directly addresses this gap.",
            "scene_2": f"{u.get('prior_limitation', 'Prior methods had significant limitations.')} They failed to generalize and required extensive manual effort, leaving a clear opening for improvement.",
            "scene_3": f"We introduce {u.get('method_name', 'our proposed method')}, a novel framework built on {', '.join(u.get('components', ['key components'])[:3])}. Each component works together to solve the core challenge effectively.",
            "scene_4": f"Our experiments demonstrate state-of-the-art performance. We outperform all baselines by a significant margin across every benchmark tested.",
            "scene_5": f"{u.get('impact_line', 'This work opens new research directions.')} We release our code and models publicly."
        }
    return data


# ═══════════════════════════════════════════════════════════
# SCENE RENDERERS — each returns path to an MP4 clip
# ═══════════════════════════════════════════════════════════

W, H = 1280, 720  # output resolution


def _save_frames_to_mp4(frames: list, output_path: str):
    """Save list of PIL Images as MP4 using ffmpeg."""
    tmp_dir = output_path.replace(".mp4", "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(tmp_dir, f"f{i:04d}.png"))
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
           "-i", os.path.join(tmp_dir, "f%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    shutil.rmtree(tmp_dir)


def _base_frame(bg_color=None) -> Image.Image:
    """Create a blank frame with background."""
    bg = bg_color or C["bg"]
    img = Image.new("RGB", (W, H), bg)
    return img


def _add_header_bar(img: Image.Image, title: str, scene_num: int, total: int = 5):
    """Add top bar with scene title and progress dots."""
    draw = ImageDraw.Draw(img)
    # top bar
    draw.rectangle([0, 0, W, 60], fill=C["black"])
    # scene title
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 18), title.upper(), fill=C["white"], font=font)
    # progress dots
    for i in range(total):
        cx = W - 40 - (total - 1 - i) * 24
        cy = 30
        color = C["white"] if i < scene_num else "#555577"
        draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=color)
    return img


def _add_narration_bar(img: Image.Image, narration: str, progress: float):
    """Add bottom narration bar that reveals text progressively."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, H-80, W, H], fill=C["black"])
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    words     = narration.split()
    n_show    = max(1, int(progress * len(words)))
    show_text = " ".join(words[:n_show])
    wrapped   = textwrap.fill(show_text, width=90)
    lines     = wrapped.split("\n")[:2]
    y         = H - 72
    for line in lines:
        draw.text((W//2, y), line, fill=C["white"], font=font, anchor="mt")
        y += 30
    return img


# ── Scene 1: Text Reveal ────────────────────────────────────

def render_scene1_text_reveal(understanding: dict, narration: str,
                               output_path: str) -> str:
    """Problem statement: lines of text fade/slide in one by one."""
    print("  [Scene 1] Rendering text reveal...")
    n_frames = 15 * FPS
    lines    = understanding.get("problem_lines", ["The problem", "needs solving"])

    # Add extra context lines
    all_lines = [
        understanding.get("paper_title", "Research Overview").upper(),
        "",
        "THE CHALLENGE:",
    ] + lines + [
        "",
        f"This paper introduces: {understanding.get('method_name', 'a new method')}"
    ]

    frames = []
    for i in range(n_frames):
        t     = i / n_frames
        img   = _base_frame(C["white"])
        draw  = ImageDraw.Draw(img)

        # Blue accent bar on left
        draw.rectangle([60, 80, 68, H-80], fill=C["blue"])

        try:
            font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
            font_med   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except Exception:
            font_big = font_med = font_small = ImageFont.load_default()

        # Reveal lines progressively
        lines_per_second = len(all_lines) / 12.0
        n_show = int(t * 12 * lines_per_second) + 1
        n_show = min(n_show, len(all_lines))

        y = 120
        for idx, line in enumerate(all_lines[:n_show]):
            if idx == 0:
                font_use = font_big
                color    = C["blue"]
            elif line in ("THE CHALLENGE:", ""):
                font_use = font_med
                color    = C["gray"]
            elif "This paper introduces" in line:
                font_use = font_med
                color    = C["green"]
            else:
                font_use = font_small
                color    = C["black"]

            # Slide in from left
            slide = min(1.0, (t * 12 - idx * 0.7) * 3)
            slide = max(0.0, slide)
            x     = int(90 + (1 - slide) * (-200))

            if slide > 0 and line:
                draw.text((x, y), line, fill=color, font=font_use)
            y += font_use.size + 12

        img = _add_header_bar(img, "The Problem", 1)
        img = _add_narration_bar(img, narration, t)
        frames.append(img)

    _save_frames_to_mp4(frames, output_path)
    return output_path


# ── Scene 2: Prior Work with Red X ─────────────────────────

def render_scene2_prior_work(understanding: dict, narration: str,
                              output_path: str) -> str:
    """Shows prior approach diagram then crosses it out with red X."""
    print("  [Scene 2] Rendering prior work...")
    n_frames   = 15 * FPS
    limitation = understanding.get("prior_limitation", "Prior methods had limitations.")

    prior_comps = ["Data", "Old Method", "Output"]
    frames = []

    for i in range(n_frames):
        t   = i / n_frames
        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
        fig.patch.set_facecolor("#F8F9FA")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Draw prior method pipeline (appears in first 40%)
        if t > 0.05:
            box_appear = min(1.0, (t - 0.05) / 0.35)
            for idx, comp in enumerate(prior_comps):
                alpha = min(1.0, box_appear * 3 - idx * 0.8)
                alpha = max(0, alpha)
                if alpha > 0:
                    bx = 1.5 + idx * 2.8
                    box = FancyBboxPatch((bx, 2.2), 2.0, 1.2,
                                         boxstyle="round,pad=0.1",
                                         facecolor="#CFD8DC",
                                         edgecolor="#546E7A",
                                         linewidth=2, alpha=alpha)
                    ax.add_patch(box)
                    ax.text(bx + 1.0, 2.8, comp, ha='center', va='center',
                             fontsize=14, fontweight='bold',
                             color="#37474F", alpha=alpha)
                    if idx < len(prior_comps) - 1:
                        ax.annotate("", xy=(bx + 2.2, 2.8),
                                     xytext=(bx + 2.0, 2.8),
                                     arrowprops=dict(arrowstyle="->",
                                                     color="#546E7A", lw=2),
                                     alpha=alpha)

        # Label
        ax.text(5, 5.2, "PRIOR APPROACH", ha='center', fontsize=18,
                 fontweight='bold', color="#546E7A",
                 alpha=min(1.0, max(0, t * 5)))

        # Limitation text appears
        if t > 0.4:
            lim_alpha = min(1.0, (t - 0.4) / 0.2)
            ax.text(5, 1.2, f'"{limitation}"', ha='center',
                     fontsize=11, color="#C62828", style='italic',
                     alpha=lim_alpha, wrap=True)

        # Red X appears over diagram
        if t > 0.65:
            x_alpha = min(1.0, (t - 0.65) / 0.2)
            ax.plot([1.2, 8.8], [1.5, 4.8], color='red',
                     linewidth=12 * x_alpha, alpha=x_alpha * 0.7,
                     solid_capstyle='round')
            ax.plot([8.8, 1.2], [1.5, 4.8], color='red',
                     linewidth=12 * x_alpha, alpha=x_alpha * 0.7,
                     solid_capstyle='round')
            ax.text(5, 3.2, "INSUFFICIENT", ha='center', fontsize=28,
                     fontweight='bold', color='red', alpha=x_alpha,
                     rotation=-15)

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        plt.close()

        img = Image.fromarray(arr[..., :3])
        img = img.resize((W, H), Image.LANCZOS)
        img = _add_header_bar(img, "Prior Work", 2)
        img = _add_narration_bar(img, narration, t)
        frames.append(img)

    _save_frames_to_mp4(frames, output_path)
    return output_path


# ── Scene 3: Method — boxes appear one by one ───────────────

def render_scene3_method_build(understanding: dict, narration: str,
                                output_path: str) -> str:
    """Method pipeline builds up component by component."""
    print("  [Scene 3] Rendering method build...")
    n_frames   = 20 * FPS
    components = understanding.get("components", ["Input", "Encoder", "Process", "Output"])
    method     = understanding.get("method_name", "Our Method")
    n_comp     = len(components)
    frames     = []

    box_colors = [C["blue_light"], C["green_light"], C["amber_light"], C["blue_light"]]
    edge_colors = [C["blue"], C["green"], C["amber"], C["blue"]]

    for i in range(n_frames):
        t   = i / n_frames
        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
        fig.patch.set_facecolor(C["white"])
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Title
        title_alpha = min(1.0, t * 6)
        ax.text(6, 5.5, method, ha='center', fontsize=22,
                 fontweight='bold', color=C["blue"], alpha=title_alpha)

        # Components appear one by one
        comp_spacing = 10.0 / n_comp
        for idx, comp in enumerate(components):
            appear_t = (idx) / n_comp
            comp_alpha = min(1.0, max(0, (t - appear_t * 0.6) / 0.15))

            if comp_alpha > 0:
                cx  = 1.0 + idx * comp_spacing + comp_spacing / 2
                fill = box_colors[idx % len(box_colors)]
                edge = edge_colors[idx % len(edge_colors)]

                # Box slides up from below
                cy_base = 2.5
                cy_slide = cy_base - (1 - comp_alpha) * 2.0

                box = FancyBboxPatch(
                    (cx - comp_spacing * 0.38, cy_slide - 0.6),
                    comp_spacing * 0.76, 1.2,
                    boxstyle="round,pad=0.1",
                    facecolor=fill, edgecolor=edge,
                    linewidth=2.5, alpha=comp_alpha, zorder=3
                )
                ax.add_patch(box)

                wrapped = textwrap.fill(comp, width=10)
                ax.text(cx, cy_slide, wrapped, ha='center', va='center',
                         fontsize=max(9, 13 - n_comp),
                         fontweight='bold', color="#1A1A2E",
                         alpha=comp_alpha, zorder=4)

                # Arrow to next component
                if idx < n_comp - 1:
                    next_appear = ((idx + 1) / n_comp)
                    arrow_alpha = min(1.0, max(0, (t - next_appear * 0.6 + 0.05) / 0.1))
                    if arrow_alpha > 0:
                        ax.annotate("",
                            xy=(cx + comp_spacing * 0.42, cy_base),
                            xytext=(cx + comp_spacing * 0.38, cy_base),
                            arrowprops=dict(arrowstyle="-|>",
                                           color=C["gray"], lw=2.5,
                                           mutation_scale=18),
                            alpha=arrow_alpha, zorder=2)

        # "NEW" badge appears at end
        if t > 0.85:
            badge_alpha = min(1.0, (t - 0.85) / 0.1)
            ax.text(6, 1.2, "✓ NOVEL CONTRIBUTION", ha='center',
                     fontsize=16, fontweight='bold', color=C["green"],
                     alpha=badge_alpha,
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor=C["green_light"],
                               edgecolor=C["green"], alpha=badge_alpha))

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        plt.close()

        img = Image.fromarray(arr[..., :3])
        img = img.resize((W, H), Image.LANCZOS)
        img = _add_header_bar(img, "Our Method", 3)
        img = _add_narration_bar(img, narration, t)
        frames.append(img)

    _save_frames_to_mp4(frames, output_path)
    return output_path


# ── Scene 4: Results — bars grow ────────────────────────────

def render_scene4_results(understanding: dict, narration: str,
                           output_path: str) -> str:
    """Bar chart with bars growing from zero."""
    print("  [Scene 4] Rendering results chart...")
    n_frames = 12 * FPS
    results  = understanding.get("results", [
        {"label": "Ours", "value": 0.92},
        {"label": "Baseline A", "value": 0.74},
        {"label": "Prior SOTA", "value": 0.83},
    ])
    frames = []

    for i in range(n_frames):
        t   = i / n_frames
        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
        fig.patch.set_facecolor(C["white"])
        ax.set_facecolor(C["bg"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(C["gray"])
        ax.spines['bottom'].set_color(C["gray"])
        ax.tick_params(colors=C["gray"])

        n      = len(results)
        labels = [r.get("label", f"M{i}") for i, r in enumerate(results)]
        vals   = [r.get("value", 0.5) for r in results]

        # Easing function for bar growth
        grow_t = min(1.0, t / 0.6)
        ease   = 1 - (1 - grow_t) ** 3  # cubic ease out

        bar_colors = []
        for idx, lbl in enumerate(labels):
            is_ours = idx == 0 or "our" in lbl.lower() or "ours" in lbl.lower()
            bar_colors.append(C["blue"] if is_ours else C["gray_light"])

        animated_vals = [v * ease for v in vals]
        bars = ax.bar(range(n), animated_vals,
                       color=bar_colors,
                       edgecolor=[C["blue"] if c == C["blue"] else C["gray"]
                                   for c in bar_colors],
                       linewidth=2)

        # Value labels on bars
        if ease > 0.1:
            for bar, val, aval in zip(bars, vals, animated_vals):
                if aval > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2,
                             aval + 0.01,
                             f"{val:.3f}",
                             ha='center', va='bottom',
                             fontsize=14, fontweight='bold',
                             color=C["blue"] if bar.get_facecolor()[0] < 0.5
                             else C["gray"])

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Performance", fontsize=13, color=C["gray"])
        ax.set_title("Quantitative Results", fontsize=18,
                      fontweight='bold', color=C["blue"], pad=15)

        # "State of the art" annotation
        if t > 0.7:
            ann_alpha = min(1.0, (t - 0.7) / 0.2)
            best_idx  = vals.index(max(vals))
            ax.annotate("★ Best",
                          xy=(best_idx, vals[best_idx]),
                          xytext=(best_idx + 0.5, vals[best_idx] + 0.08),
                          fontsize=13, color=C["blue"],
                          fontweight='bold',
                          arrowprops=dict(arrowstyle="->",
                                          color=C["blue"], lw=1.5),
                          alpha=ann_alpha)

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        plt.close()

        img = Image.fromarray(arr[..., :3])
        img = img.resize((W, H), Image.LANCZOS)
        img = _add_header_bar(img, "Results", 4)
        img = _add_narration_bar(img, narration, t)
        frames.append(img)

    _save_frames_to_mp4(frames, output_path)
    return output_path


# ── Scene 5: Full reveal with glow ──────────────────────────

def render_scene5_impact(understanding: dict, narration: str,
                          overview_diagram_path: str,
                          output_path: str) -> str:
    """Final scene: overview diagram fades in with glow, impact text."""
    print("  [Scene 5] Rendering impact scene...")
    n_frames = 8 * FPS
    impact   = understanding.get("impact_line", "This work advances the field.")
    title    = understanding.get("paper_title", "Research Overview")
    frames   = []

    # Load the rendered overview diagram
    try:
        diagram = Image.open(overview_diagram_path).convert("RGB")
        diagram = diagram.resize((W - 80, H - 200), Image.LANCZOS)
        has_diagram = True
    except Exception:
        has_diagram = False

    for i in range(n_frames):
        t   = i / n_frames
        img = _base_frame(C["black"])
        draw = ImageDraw.Draw(img)

        # Diagram fades in
        fade = min(1.0, t * 3)
        if has_diagram and fade > 0:
            diagram_faded = diagram.copy()
            enhancer = ImageEnhance.Brightness(diagram_faded)
            diagram_faded = enhancer.enhance(fade)
            img.paste(diagram_faded, (40, 100))

        # Glow effect on top
        if t > 0.3:
            glow_alpha = min(1.0, (t - 0.3) / 0.3)
            draw.rectangle([0, 0, W, 90], fill=(0, 0, 0, 200))

        # Title text
        try:
            font_title  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            font_impact = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except Exception:
            font_title = font_impact = ImageFont.load_default()

        title_alpha = int(min(255, t * 3 * 255))
        draw.text((W//2, 45), title.upper(), fill=(255, 255, 255),
                   font=font_title, anchor="mm")

        # Impact text at bottom
        if t > 0.5:
            impact_alpha = int(min(255, (t - 0.5) * 4 * 255))
            draw.rectangle([0, H-110, W, H], fill=(0, 0, 30))
            wrapped = textwrap.fill(impact, width=70)
            draw.text((W//2, H-75), wrapped, fill=(255, 220, 100),
                       font=font_impact, anchor="mm")
            draw.text((W//2, H-35), "Code & models available publicly",
                       fill=(150, 200, 255), font=font_impact, anchor="mm")

        img = _add_header_bar(img, "Impact", 5)
        frames.append(img)

    _save_frames_to_mp4(frames, output_path)
    return output_path


# ═══════════════════════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════════════════════

def generate_audio(text: str, scene_id: str, out_dir: str) -> str:
    audio_path = os.path.join(out_dir, f"{scene_id}.wav")
    duration   = {"scene_1": 15, "scene_2": 15, "scene_3": 20,
                  "scene_4": 12, "scene_5": 8}.get(scene_id, 15)
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC",
                   progress_bar=False)
        tts.tts_to_file(text=text, file_path=audio_path)
        print(f"  [TTS] Audio: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"  [TTS] Failed ({type(e).__name__}). Using silence.")
        cmd = ["ffmpeg", "-y", "-f", "lavfi",
               "-i", "anullsrc=r=22050:cl=mono",
               "-t", str(duration), audio_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path


def combine_av(video: str, audio: str, out: str) -> str:
    cmd = ["ffmpeg", "-y", "-i", video, "-i", audio,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-shortest", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def stitch_final(clips: list, out_path: str) -> str:
    txt = out_path.replace(".mp4", "_list.txt")
    with open(txt, 'w') as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", txt, "-c", "copy", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(txt)
    return out_path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def render_overview_diagram(understanding: dict, out_path: str) -> str:
    """Render clean overview diagram for Scene 5 and reference."""
    from branch_a.diagram_renderer import render_figure
    components = understanding.get("components", ["Input", "Process", "Output"])
    entities   = []
    relations  = []
    n          = len(components)
    for i, comp in enumerate(components):
        x = int(5 + i * (88 / n))
        entities.append({"id": f"E{i}", "label": comp,
                          "bbox": [x, 35, int(80/n)-4, 22],
                          "type": "process" if 0 < i < n-1
                                  else ("input" if i == 0 else "output")})
        if i > 0:
            relations.append({"from": f"E{i-1}", "to": f"E{i}", "label": ""})

    plan = {
        "id":          "overview",
        "figure_type": "architecture",
        "caption":     understanding.get("paper_title", ""),
        "entities":    entities,
        "relations":   relations,
        "sd_prompt":   ""
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return render_figure(plan, os.path.dirname(out_path))


def run_branch_b(abstract: str, introduction: str, conclusion: str,
                  paper_name: str = "paper",
                  output_dir: str = OUTPUT_DIR) -> str:

    print("\n" + "="*60)
    print("  BRANCH B — ANIMATED VIDEO ABSTRACT")
    print("="*60)

    paper_dir = os.path.join(output_dir, paper_name)
    os.makedirs(paper_dir, exist_ok=True)

    # 1. Understand paper
    understanding = understand_paper(abstract, introduction, conclusion)
    with open(os.path.join(paper_dir, "understanding.json"), 'w') as f:
        json.dump(understanding, f, indent=2)

    # 2. Write script
    script = write_script(understanding)
    with open(os.path.join(paper_dir, "script.json"), 'w') as f:
        json.dump(script, f, indent=2)

    # 3. Render overview diagram (used in Scene 5)
    diagram_path = os.path.join(paper_dir, "overview.png")
    try:
        render_overview_diagram(understanding, diagram_path)
    except Exception as e:
        print(f"  [Diagram] Render failed ({e}), Scene 5 will use text only.")
        diagram_path = None

    print("\n[Branch B] Generating 5 animated scenes...")

    scene_renderers = {
        "scene_1": lambda narr, out: render_scene1_text_reveal(understanding, narr, out),
        "scene_2": lambda narr, out: render_scene2_prior_work(understanding, narr, out),
        "scene_3": lambda narr, out: render_scene3_method_build(understanding, narr, out),
        "scene_4": lambda narr, out: render_scene4_results(understanding, narr, out),
        "scene_5": lambda narr, out: render_scene5_impact(
                       understanding, narr, diagram_path or "", out),
    }

    final_clips = []
    for scene in SCENES:
        sid      = scene["id"]
        narration = script.get(sid, "")
        print(f"\n  ── {sid}: {scene['title']} ──")

        # Render animated scene
        raw_clip = os.path.join(paper_dir, f"{sid}_video.mp4")
        scene_renderers[sid](narration, raw_clip)

        # Generate audio
        audio_path = generate_audio(narration, sid, paper_dir)

        # Combine audio + video
        final_clip = os.path.join(paper_dir, f"{sid}_final.mp4")
        combine_av(raw_clip, audio_path, final_clip)
        final_clips.append(final_clip)

    # Stitch all scenes
    final_path = os.path.join(paper_dir, f"{paper_name}_video_abstract.mp4")
    stitch_final(final_clips, final_path)

    print("\n" + "="*60)
    print("  VIDEO ABSTRACT COMPLETE")
    print(f"  Output: {final_path}")
    print("="*60)
    return final_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted_json", nargs="?")
    parser.add_argument("--pdf",        type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.pdf:
        from shared.pdf_parser import parse_paper
        data     = parse_paper(args.pdf)
        abstract = data.get("abstract", "")
        intro    = data.get("introduction", "")
        conc     = data.get("conclusion", "")
        name     = os.path.basename(args.pdf).replace(".pdf", "")
    elif args.extracted_json:
        with open(args.extracted_json) as f:
            data = json.load(f)
        abstract = data.get("abstract", "")
        intro    = data.get("introduction", "")
        conc     = data.get("conclusion", "")
        name     = os.path.basename(args.extracted_json).replace("_extracted.json","")
    else:
        # Demo
        abstract = "We present a novel LLM-guided multimodal diffusion framework for automated scientific figure and video abstract generation from research papers."
        intro    = "Research papers require significant effort to understand. Prior methods failed to automate visual content generation. We address this with an end-to-end pipeline."
        conc     = "Our system generates scientific figures and video abstracts automatically. Results show significant improvement over baselines. Future work will expand to more domains."
        name     = "demo"

    run_branch_b(abstract, intro, conc, name, args.output_dir)