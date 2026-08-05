"""Visualize AoH per-KV-group effective rank heatmap."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_effective_rank(output_dir: str, save_path: str | None = None):
    out = Path(output_dir)
    eff_rank_kv = np.load(out / "eff_rank_kv.npy")
    num_layers, G = eff_rank_kv.shape

    # Try to read metadata for title
    title_extra = ""
    results_path = out / "results.json"
    if results_path.exists():
        import json
        with open(results_path) as f:
            meta = json.load(f)
        model = meta.get("model", "")
        agg = meta.get("agg", "")
        title_extra = f"\n({model}, agg={agg})"

    _, ax = plt.subplots(figsize=(max(3, G * 1.2), max(6, num_layers * 0.35)))

    im = ax.imshow(eff_rank_kv, aspect="auto", cmap="YlOrRd_r", interpolation="nearest")

    ax.set_xlabel("KV Head Group", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"Effective Rank (per KV Group){title_extra}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(G))
    ax.set_xticklabels([f"G{i}" for i in range(G)])
    ax.set_yticks(range(0, num_layers, 2))
    ax.set_yticklabels(range(0, num_layers, 2))

    for l in range(num_layers):
        for g in range(G):
            val = eff_rank_kv[l, g]
            ax.text(g, l, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if val < 90 else "black")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Effective Rank", fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = str(out / "eff_rank_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot effective rank heatmap")
    parser.add_argument("output_dir", help="AoH output directory")
    parser.add_argument("--save", default=None, help="Output image path")
    args = parser.parse_args()
    plot_effective_rank(args.output_dir, args.save)
