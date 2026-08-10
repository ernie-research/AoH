#!/usr/bin/env python3
"""Run the paper-style efficiency benchmark for all three AoH models."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def add_common(command: list[str], args) -> list[str]:
    command += [
        "--mode", args.mode,
        "--sps", args.sparsities,
        "--gpus", args.gpus,
        "--per_gpu_gb", str(args.per_gpu_gb),
        "--chunk_size", str(args.chunk_size),
        "--prefill_lens", args.context_lengths,
        "--decode_lens", args.context_lengths,
        "--n_reps", str(args.repetitions),
        "--n_decode_reps", str(args.decode_repetitions),
        "--n_decode_steps", str(args.decode_steps),
        "--sink_size", str(args.sink_size),
        "--recent_size", str(args.recent_size),
        "--out_dir", str(args.output_dir),
    ]
    if args.force:
        command.append("--force")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--qwen3-head-root", type=Path, required=True)
    parser.add_argument("--llama31-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--llama31-head-root", type=Path, required=True)
    parser.add_argument("--qwen25-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--qwen25-head-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "full", "aoh", "compare"], default="all")
    parser.add_argument("--sparsities", default="0.25,0.5,0.75")
    parser.add_argument("--context-lengths", default="4096,8192,16384,32768,65536,131072,262144")
    parser.add_argument("--gpus", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--per-gpu-gb", type=int, default=72)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--decode-repetitions", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--sink-size", type=int, default=128)
    parser.add_argument("--recent-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/efficiency/outputs"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runs = [
        [sys.executable, str(here / "benchmark_qwen.py"), "--model", "Qwen3-8B", "--model-path", args.qwen3_model, "--head-root", str(args.qwen3_head_root)],
        [sys.executable, str(here / "benchmark_llama31.py"), "--model-path", args.llama31_model, "--head-root", str(args.llama31_head_root)],
        [sys.executable, str(here / "benchmark_qwen.py"), "--model", "Qwen2.5-7B", "--model-path", args.qwen25_model, "--head-root", str(args.qwen25_head_root)],
    ]
    for command in runs:
        subprocess.run(add_common(command, args), check=True)

    subprocess.run(
        [sys.executable, str(here / "collect_results.py"), args.output_dir],
        check=True,
    )


if __name__ == "__main__":
    main()
