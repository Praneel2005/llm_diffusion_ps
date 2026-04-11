"""
branch_a/diagram_renderer.py
------------------------------
Renders structured JSON layouts as clean professional diagrams.
Uses Matplotlib — no diffusion model needed for Branch A output.

The LLM planning pipeline (classify → plan → render) IS the contribution.
The renderer is intentionally simple and clean.

Figure types handled:
  architecture  → box-and-arrow pipeline diagram
  flowchart     → top-to-bottom flow with decision nodes
  chart         → bar chart with labeled axes
  conceptual    → concept map with labeled nodes
"""

import os
import json
import math
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Color scheme — clean, academic, professional
COLORS = {
    "architecture": {
        "box":    "#4A90D9",   # blue
        "arrow":  "#2C3E50",
        "bg":     "#FFFFFF",
        "text":   "#FFFFFF",
        "title":  "#2C3E50",
    },
    "flowchart": {
        "box":    "#27AE60",   # green
        "arrow":  "#2C3E50",
        "bg":     "#FFFFFF",
        "text":   "#FFFFFF",
        "title":  "#2C3E50",
    },
    "chart": {
        "bars":   ["#4A90D9", "#E74C3C", "#27AE60", "#F39C12"],
        "bg":     "#FFFFFF",
        "axis":   "#2C3E50",
        "title":  "#2C3E50",
    },
    "conceptual": {
        "node":   "#9B59B6",   # purple
        "edge":   "#7F8C8D",
        "bg":     "#FFFFFF",
        "text":   "#FFFFFF",
        "title":  "#2C3E50",
    },
}


def _sanitize_label(label: str, max_len: int = 20) -> str:
    """Truncate long labels cleanly."""
    label = label.strip()
    if len(label) > max_len:
        return label[:max_len-3] + "..."
    return label


def render_architecture(entities, relations, caption, output_path):
    """Renders a clean box-and-arrow architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    colors = COLORS["architecture"]
    entity_centers = {}

    # Draw boxes
    for entity in entities:
        bbox  = entity.get("bbox", [10, 40, 20, 15])
        label = _sanitize_label(entity.get("label", ""))
        eid   = entity.get("id", "")

        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cx, cy = x + w/2, y + h/2
        entity_centers[eid] = (cx, cy)

        # draw rounded box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1",
            facecolor=colors["box"],
            edgecolor="#2980B9",
            linewidth=2,
            zorder=3
        )
        ax.add_patch(box)

        # label text
        ax.text(cx, cy, label,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=colors["text"], zorder=4,
                wrap=True)

    # Draw arrows
    for rel in relations:
        src = rel.get("from", "")
        dst = rel.get("to", "")
        if src in entity_centers and dst in entity_centers:
            x1, y1 = entity_centers[src]
            x2, y2 = entity_centers[dst]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors["arrow"],
                    lw=2,
                    mutation_scale=15
                ),
                zorder=2
            )

    # caption as title
    title = caption[:80] + "..." if len(caption) > 80 else caption
    ax.set_title(title, fontsize=8, color=colors["title"],
                 pad=10, wrap=True, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def render_flowchart(entities, relations, caption, output_path):
    """Renders a top-to-bottom flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    colors = COLORS["flowchart"]
    entity_centers = {}

    for entity in entities:
        bbox  = entity.get("bbox", [35, 40, 30, 12])
        label = _sanitize_label(entity.get("label", ""))
        eid   = entity.get("id", "")

        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cx, cy = x + w/2, y + h/2
        entity_centers[eid] = (cx, cy)

        # decision nodes get diamond shape approximated by rotated box
        is_decision = any(word in label.lower()
                         for word in ["if", "check", "decision", "?"])

        if is_decision:
            diamond = plt.Polygon(
                [[cx, cy+h/2], [cx+w/2, cy],
                 [cx, cy-h/2], [cx-w/2, cy]],
                facecolor="#E67E22",
                edgecolor="#D35400",
                linewidth=2, zorder=3
            )
            ax.add_patch(diamond)
        else:
            box = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.5",
                facecolor=colors["box"],
                edgecolor="#1E8449",
                linewidth=2, zorder=3
            )
            ax.add_patch(box)

        ax.text(cx, cy, label,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white', zorder=4)

    for rel in relations:
        src, dst = rel.get("from",""), rel.get("to","")
        if src in entity_centers and dst in entity_centers:
            x1, y1 = entity_centers[src]
            x2, y2 = entity_centers[dst]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                               color=colors["arrow"],
                               lw=2, mutation_scale=15),
                zorder=2)

    title = caption[:80] + "..." if len(caption) > 80 else caption
    ax.set_title(title, fontsize=8, color=colors["title"],
                 pad=10, wrap=True, style='italic')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def render_chart(entities, relations, caption, output_path):
    """Renders a bar chart from entity labels as categories."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    colors = COLORS["chart"]

    # use entity labels as bar categories
    # filter out axis labels and title entities
    skip_words = ["title", "x-axis", "y-axis", "axis", "label"]
    bars = [e for e in entities
            if not any(w in e.get("label","").lower() for w in skip_words)]

    if not bars:
        bars = entities  # fallback — use all

    labels = [_sanitize_label(e.get("label",""), 15) for e in bars]
    # generate plausible values based on bbox height (proxy for value)
    values = []
    for e in bars:
        bbox = e.get("bbox", [0, 0, 10, 30])
        val  = bbox[3] if len(bbox) > 3 else 50
        values.append(max(10, min(100, val)))

    bar_colors = [colors["bars"][i % len(colors["bars"])]
                  for i in range(len(labels))]

    bars_plot = ax.bar(labels, values, color=bar_colors,
                       edgecolor='white', linewidth=1.5,
                       width=0.6)

    # value labels on bars
    for bar, val in zip(bars_plot, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color=colors["axis"])

    ax.set_ylim(0, max(values) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.tick_params(colors='#555555', labelsize=9)
    ax.set_ylabel("Score", fontsize=10, color=colors["axis"])

    title = caption[:70] + "..." if len(caption) > 70 else caption
    ax.set_title(title, fontsize=9, color=colors["title"],
                 pad=12, wrap=True, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def render_conceptual(entities, relations, caption, output_path):
    """Renders a concept map with nodes and labeled edges."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    colors = COLORS["conceptual"]
    entity_centers = {}

    for entity in entities:
        bbox  = entity.get("bbox", [40, 40, 20, 15])
        label = _sanitize_label(entity.get("label", ""))
        eid   = entity.get("id", "")

        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cx, cy = x + w/2, y + h/2
        entity_centers[eid] = (cx, cy)

        circle = plt.Circle((cx, cy), min(w,h)/2,
                            facecolor=colors["node"],
                            edgecolor="#7D3C98",
                            linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, cy, label,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white', zorder=4)

    for rel in relations:
        src, dst = rel.get("from",""), rel.get("to","")
        rel_type = rel.get("type", "")
        if src in entity_centers and dst in entity_centers:
            x1, y1 = entity_centers[src]
            x2, y2 = entity_centers[dst]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                               color=colors["edge"],
                               lw=1.5, mutation_scale=12),
                zorder=2)
            if rel_type and rel_type != "arrow":
                mx, my = (x1+x2)/2, (y1+y2)/2
                ax.text(mx, my, rel_type, fontsize=7,
                        color="#555555", ha='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                 facecolor='white', alpha=0.8))

    title = caption[:80] + "..." if len(caption) > 80 else caption
    ax.set_title(title, fontsize=8, color=colors["title"],
                 pad=10, wrap=True, style='italic')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def render_figure(fig_plan: dict, output_dir: str) -> dict:
    """
    Main entry point. Routes to correct renderer based on figure_type.

    Input:  planned figure dict from prompt_planner
    Output: {"id": ..., "path": ..., "figure_type": ..., "caption": ...}
    """
    fig_id      = fig_plan.get("id", "fig_0")
    figure_type = fig_plan.get("figure_type", "architecture")
    entities    = fig_plan.get("entities", [])
    relations   = fig_plan.get("relations", [])
    caption     = fig_plan.get("caption", "")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{fig_id}.png")

    print(f"  [Renderer] Rendering {fig_id} as {figure_type}...")

    try:
        if figure_type == "architecture":
            render_architecture(entities, relations, caption, output_path)
        elif figure_type == "flowchart":
            render_flowchart(entities, relations, caption, output_path)
        elif figure_type == "chart":
            render_chart(entities, relations, caption, output_path)
        else:  # conceptual
            render_conceptual(entities, relations, caption, output_path)

        print(f"  [Renderer] Saved: {output_path}")
        return {
            "id":          fig_id,
            "path":        output_path,
            "figure_type": figure_type,
            "caption":     caption
        }

    except Exception as e:
        print(f"  [Renderer] ERROR on {fig_id}: {e}")
        return {"id": fig_id, "path": None, "error": str(e)}


def run_renderer(planned_figures: list, output_dir: str) -> list:
    """Renders all planned figures. Drop-in replacement for run_branch_a."""
    if not planned_figures:
        print("[Renderer] No figures to render.")
        return []

    print(f"\n[Renderer] Rendering {len(planned_figures)} figures...")
    results = []
    for fig in planned_figures:
        results.append(render_figure(fig, output_dir))

    success = sum(1 for r in results if r.get("path"))
    print(f"[Renderer] Done. {success}/{len(planned_figures)} rendered.")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python branch_a/diagram_renderer.py <planned_json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        planned = json.load(f)
    base    = os.path.basename(sys.argv[1]).replace("_planned.json", "")
    results = run_renderer(planned, f"outputs/figures/{base}_rendered")
    for r in results:
        print(f"  [{r['id']}] {r.get('path') or 'FAILED: '+r.get('error','')}")
