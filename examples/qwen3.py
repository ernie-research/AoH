"""AoH usage example: Qwen3-8B and Qwen3-32B head classification.

Model specs
-----------
Qwen3-8B  : GQA, H=32 Q heads, G=8 KV heads, GRP=4, 36 layers, head_dim=128
Qwen3-32B : GQA, H=64 Q heads, G=8 KV heads, GRP=8, 64 layers, head_dim=128

Key differences vs Qwen2.5:
  - Larger group ratio (GRP=4/8 vs 7)
  - Qwen3 introduces QK-Norm (q_norm / k_norm RMSNorm per head);
    AoH weight_extract reads raw W_Q / W_K before norm, which is correct
    for the effective-rank analysis of the projection matrices.

Usage:
    python examples/qwen3.py
"""

import sys
from pathlib import Path

# Allow running the example directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoh import AoHClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


# ─── Qwen3-8B ─────────────────────────────────────────────────────────────────

def run_qwen3_8b():
    MODEL_PATH = "Qwen/Qwen3-8B"

    # GQA mean aggregation (recommended)
    cls = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",
        binary=False,
        agg="mean",       # GRP=4, mean over 4 Q-heads per KV group
    )
    # sparsity=0.5 → 4/8 KV groups per layer are Streaming
    labels = cls.classify(sparsity=0.5)
    cls.save(OUTPUT_ROOT / "aoh_qwen3_8b_mean")

    # min aggregation (conservative: any retrieval Q-head → whole group is retrieval)
    cls_min = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",
        agg="min",
    )
    cls_min.classify(sparsity=0.5)
    cls_min.save(OUTPUT_ROOT / "aoh_qwen3_8b_min")

    # Binary acceleration (~64x faster, sign() approximation)
    cls_bin = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",
        binary=True,
        agg="mean",
    )
    cls_bin.classify(sparsity=0.5)
    cls_bin.save(OUTPUT_ROOT / "aoh_qwen3_8b_binary")

    print("\nQwen3-8B AoH done.")
    print(f"  labels shape : {labels.shape}  (36 layers × 8 KV groups)")
    print(f"  retrieval    : {labels.mean():.2%}")


# ─── Qwen3-32B ────────────────────────────────────────────────────────────────

def run_qwen3_32b():
    MODEL_PATH = "Qwen/Qwen3-32B"

    cls = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",
        binary=False,
        agg="mean",       # GRP=8, mean over 8 Q-heads per KV group
    )
    labels = cls.classify(sparsity=0.5)
    cls.save(OUTPUT_ROOT / "aoh_qwen3_32b_mean")

    print("\nQwen3-32B AoH done.")
    print(f"  labels shape : {labels.shape}  (64 layers × 8 KV groups)")
    print(f"  retrieval    : {labels.mean():.2%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "32b", "all"], default="8b")
    args = parser.parse_args()

    if args.model in ("8b", "all"):
        run_qwen3_8b()
    if args.model in ("32b", "all"):
        run_qwen3_32b()
