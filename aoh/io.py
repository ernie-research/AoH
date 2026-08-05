"""Save and load native AoH result files."""

import json
from pathlib import Path

import numpy as np


def save_results(
    output_dir: str,
    eff_ranks_q: np.ndarray,
    eff_ranks_kv: np.ndarray,
    head_labels: np.ndarray,
    metadata: dict,
) -> None:
    """Save effective ranks, classifications, and run metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "eff_rank.npy", eff_ranks_q)
    np.save(out / "eff_rank_kv.npy", eff_ranks_kv)
    np.save(out / "head_classification.npy", head_labels)

    retrieval_per_layer = head_labels.mean(axis=1)
    results = {
        **metadata,
        "version": "AoH-v1",
        "num_layers": int(head_labels.shape[0]),
        "num_kv_heads": int(head_labels.shape[1]),
        "overall_retrieval_ratio": float(head_labels.mean()),
        "overall_streaming_ratio": float(1.0 - head_labels.mean()),
        "retrieval_ratio_per_layer": retrieval_per_layer.tolist(),
        "streaming_ratio_per_layer": (1.0 - retrieval_per_layer).tolist(),
    }
    with (out / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def load_results(output_dir: str) -> dict:
    """Load results created by :func:`save_results`."""
    out = Path(output_dir)
    required_paths = {
        "eff_ranks_q": out / "eff_rank.npy",
        "eff_ranks_kv": out / "eff_rank_kv.npy",
        "head_labels": out / "head_classification.npy",
        "metadata": out / "results.json",
    }
    missing = [path.name for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing AoH result files in {out}: {', '.join(sorted(missing))}"
        )

    with required_paths["metadata"].open(encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        "eff_ranks_q": np.load(required_paths["eff_ranks_q"]),
        "eff_ranks_kv": np.load(required_paths["eff_ranks_kv"]),
        "head_labels": np.load(required_paths["head_labels"]),
        "metadata": metadata,
    }
