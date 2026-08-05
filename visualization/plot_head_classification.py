"""Visualize AoH head classification heatmap (Retrieval vs Streaming)."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_head_classification(output_dir: str, save_path: str | None = None):
    out = Path(output_dir)
    head_labels = np.load(out / "head_classification.npy")
    num_layers, G = head_labels.shape

    # Try to read metadata for title
    title_extra = ""
    results_path = out / "results.json"
    if results_path.exists():
        import json
        with open(results_path) as f:
            meta = json.load(f)
        model = meta.get("model", "")
        agg = meta.get("agg", "")
        sparsity = meta.get("sparsity", "")
        title_extra = f"\n({model}, agg={agg}, sparsity={sparsity})"

    _, ax = plt.subplots(figsize=(max(3, G * 1.2), max(6, num_layers * 0.35)))

    cmap = plt.cm.colors.ListedColormap(["#4ECDC4", "#FF6B6B"])
    ax.imshow(head_labels, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xlabel("KV Head Group", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"Head Classification{title_extra}", fontsize=13, fontweight="bold")
    ax.set_xticks(range(G))
    ax.set_xticklabels([f"G{i}" for i in range(G)])
    ax.set_yticks(range(0, num_layers, 2))
    ax.set_yticklabels(range(0, num_layers, 2))

    for l in range(num_layers):
        for g in range(G):
            label = "R" if head_labels[l, g] == 1.0 else "S"
            ax.text(g, l, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    legend_patches = [
        mpatches.Patch(color="#FF6B6B", label="Retrieval (Full Attn)"),
        mpatches.Patch(color="#4ECDC4", label="Streaming (SWA)"),
    ]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = str(out / "head_classification_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot head classification heatmap")
    parser.add_argument("output_dir", help="AoH output directory")
    parser.add_argument("--save", default=None, help="Output image path")
    args = parser.parse_args()
    plot_head_classification(args.output_dir, args.save)
