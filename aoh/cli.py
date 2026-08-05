"""Command-line interface for AoH head classification.

Usage::

    python -m aoh \\
        --model_path Qwen/Qwen2.5-7B \\
        --sparsity 0.5 \\
        --agg mean \\
        --output_dir outputs/aoh_qwen25_7b

    # Binary acceleration + min aggregation
    python -m aoh \\
        --model_path Qwen/Qwen2.5-7B \\
        --sparsity 0.5 --agg min --binary \\
        --output_dir outputs/aoh_binary

    # MHA model (agg is auto-ignored)
    python -m aoh \\
        --model_path meta-llama/Llama-2-7b-hf \\
        --sparsity 0.5 \\
        --output_dir outputs/aoh_llama2
"""

import argparse

import torch

from .classifier import AoHClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="AoH: Training-free attention head classification via effective rank",
    )
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        help="Checkpoint loading dtype (default: preserve stored dtype)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device: 'cpu', 'cuda:0', etc.")

    # Classification
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Fraction of KV heads classified as Streaming per layer (0.0-1.0)")
    parser.add_argument("--agg", type=str, default="mean",
                        choices=["mean", "min", "max"],
                        help="GQA aggregation strategy (ignored for MHA models)")
    parser.add_argument("--binary", action="store_true",
                        help="Use binary (sign) approximation for speedup")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")

    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    classifier = AoHClassifier(
        model_path=args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device=args.device,
        binary=args.binary,
        agg=args.agg,
    )

    head_labels = classifier.classify(sparsity=args.sparsity)
    classifier.save(args.output_dir)

    # Summary
    info = classifier._info
    print(f"\n{'=' * 60}")
    print(f"AoH Classification Complete")
    print(f"{'=' * 60}")
    print(f"Model:     {args.model_path}")
    print(f"Mode:      {'GQA' if info['is_gqa'] else 'MHA'} "
          f"(H={info['H']}, G={info['G']}, GRP={info['GRP']})")
    if info["is_gqa"]:
        print(f"Aggregate: {args.agg}")
    print(f"Binary:    {args.binary}")
    print(f"Sparsity:  {args.sparsity:.0%}")
    print(f"Retrieval: {head_labels.mean():.2%}")
    print(f"Output:    {args.output_dir}")
    print(f"{'=' * 60}")

    # Per-layer breakdown
    print(f"\n  {'Layer':>5}  {'Retrieval':>10}  {'Streaming':>10}")
    for l in range(info["num_layers"]):
        r = head_labels[l].mean()
        print(f"  {l:5d}  {r:10.2%}  {1 - r:10.2%}")


if __name__ == "__main__":
    main()
