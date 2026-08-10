#!/usr/bin/env python3
"""Run AoH LongBench evaluation for the three paper models."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNS = (
    ("Qwen3-8B", "qwen3_model", "qwen3_head"),
    ("Llama-3.1-8B-Instruct", "llama31_model", "llama31_head"),
    ("Qwen2.5-7B", "qwen25_model", "qwen25_head"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3-head", type=Path, required=True)
    parser.add_argument("--qwen3-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--llama31-head", type=Path, required=True)
    parser.add_argument("--llama31-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--qwen25-head", type=Path, required=True)
    parser.add_argument("--qwen25-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results/outputs"))
    parser.add_argument("--gpus", default="cuda:0")
    parser.add_argument("--per-gpu-gb", type=int, default=72)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("run_longbench.py")
    for model, model_attr, head_attr in RUNS:
        command = [
            sys.executable,
            str(script),
            "--model", model,
            "--model-path", getattr(args, model_attr),
            "--head-path", str(getattr(args, head_attr)),
            "--tasks", args.tasks,
            "--output-dir", str(args.output_dir),
            "--gpus", args.gpus,
            "--per-gpu-gb", str(args.per_gpu_gb),
        ]
        if args.max_samples is not None:
            command += ["--max-samples", str(args.max_samples)]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
