"""AoH usage example: Qwen2.5-7B head classification.

Usage:
    python examples/qwen25_7b.py
"""

import sys
from pathlib import Path

# Allow running the example directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoh import AoHClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def main():
    MODEL_PATH = "Qwen/Qwen2.5-7B"

    # ── GQA + mean 聚合（推荐）──────────────────────────
    cls = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",       # 纯权重计算，单卡即可
        binary=False,          # True 则用 sign() 近似，快 ~64x
        agg="mean",            # GQA 聚合策略: "mean"、"min" 或 "max"
    )

    # 分类: sparsity=0.5 → 每层 50% KV group 为 Streaming
    cls.classify(sparsity=0.5)
    # Classification shape: (28 layers, 4 KV groups).

    # 保存 AoH 原生结果
    cls.save(OUTPUT_ROOT / "aoh_qwen25_7b_mean")
    # 产出:
    #   eff_rank.npy              ← (28, 28) 每个 Q head 的有效秩
    #   eff_rank_kv.npy           ← (28, 4)  聚合后 KV group 有效秩
    #   head_classification.npy   ← (28, 4)  二值分类
    #   results.json              ← 元信息

    # ── GQA + min 聚合（保守策略）──────────────────────
    cls_min = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",
        agg="min",             # 只要 group 内有一个 Q head 是 retrieval → 整组 retrieval
    )
    cls_min.classify(sparsity=0.5)
    cls_min.save(OUTPUT_ROOT / "aoh_qwen25_7b_min")

    # ── Binary 加速模式 ───────────────────────────────
    cls_bin = AoHClassifier(
        model_path=MODEL_PATH,
        device="cuda:0",
        binary=True,           # sign(W) 近似，BinaryAttention 定理保证谱等价
        agg="mean",
    )
    cls_bin.classify(sparsity=0.5)
    cls_bin.save(OUTPUT_ROOT / "aoh_qwen25_7b_binary")

    # ── 加载已有结果（不需要模型）───────────────────────
    # loaded = AoHClassifier.from_saved("outputs/aoh_qwen25_7b_mean")
    # print(f"\nLoaded: {loaded.head_labels.shape}")
    # print(f"Retrieval ratio: {loaded.head_labels.mean():.2%}")


if __name__ == "__main__":
    main()
