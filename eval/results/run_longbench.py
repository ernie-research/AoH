#!/usr/bin/env python3
"""Generate AoH predictions for the 21-task LongBench main-result table."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = Path(__file__).resolve().parent / "config"
MODEL_CONFIG = {
    "Qwen3-8B": {
        "path": "Qwen/Qwen3-8B",
        "runtime": "eval.efficiency.benchmark_qwen",
        "max_length": 32000,
    },
    "Llama-3.1-8B-Instruct": {
        "path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "runtime": "eval.efficiency.benchmark_llama31",
        "max_length": 32000,
    },
    "Qwen2.5-7B": {
        "path": "Qwen/Qwen2.5-7B",
        "runtime": "eval.efficiency.benchmark_qwen",
        "max_length": 32000,
    },
}
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIG, required=True)
    parser.add_argument("--model-path", help="Hugging Face model ID or local model directory")
    parser.add_argument("--head-path", type=Path, required=True)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or 'all'")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results/outputs"))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--sink-size", type=int, default=128)
    parser.add_argument("--recent-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--protocol",
        choices=["aoh", "eager"],
        default="aoh",
        help="aoh (default): FlashAttention-2 AoH runtime. All but the last "
        "--decoding-simulation-length prompt tokens are prefilled in one flash_attn_func forward, "
        "streaming heads are then trimmed to sink+recent KV and the remaining prompt tokens are "
        "decoded one by one. eager: reference eager kernels with chunked prefill (--chunk-size); "
        "streaming heads keep only sink+recent KV during the whole prefill",
    )
    parser.add_argument("--decoding-simulation-length", type=int, default=50)
    parser.add_argument("--gpus", default="cuda:0")
    parser.add_argument("--per-gpu-gb", type=int, default=72)
    parser.add_argument("--max-samples", type=int, help="Debug subset per task")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def select_tasks(raw: str, all_tasks: list[str]) -> list[str]:
    if raw == "all":
        return all_tasks
    selected = [task.strip() for task in raw.split(",") if task.strip()]
    unknown = sorted(set(selected) - set(all_tasks))
    if unknown:
        raise ValueError("Unknown LongBench tasks: " + ", ".join(unknown))
    return selected


def build_prompt(tokenizer, text: str, model_name: str, task: str) -> str:
    if task in NO_CHAT_TASKS or "Instruct" not in model_name:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def truncate_middle(tokenizer, prompt: str, max_length: int) -> str:
    token_ids = tokenizer(prompt, truncation=False).input_ids
    if len(token_ids) <= max_length:
        return prompt
    half = max_length // 2
    return tokenizer.decode(token_ids[:half], skip_special_tokens=True) + tokenizer.decode(
        token_ids[-half:], skip_special_tokens=True
    )


def eos_ids(tokenizer, model) -> set[int]:
    values = [tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]
    output = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            output.update(int(item) for item in value)
        elif value is not None:
            output.add(int(value))
    return output


def flash_prefill(model, make_cache, input_ids, sim_len):
    """FlashAttention-2 AoH prefill: prompt[:-sim_len] in one forward (flash_attn_func over all
    heads), the streaming heads' KV is trimmed to sink + recent, and the last sim_len prompt tokens
    are decoded one at a time (attention_utils.aoh_flash_attention). The prefill / decoding-
    simulation schedule matches the DuoAttention LongBench evaluation, so the two methods are
    compared under the same protocol. Decoding continues on the same cache."""
    split = max(input_ids.shape[1] - sim_len, 1)
    cache = make_cache("flash")
    # The patched attention ignores the HF mask; under "flash_attention_2" HF skips building the
    # dense (split x split) 4D causal mask that eager would allocate.
    attn_impl = model.config._attn_implementation
    model.config._attn_implementation = "flash_attention_2"
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids[:, :split], past_key_values=cache, use_cache=True,
                           logits_to_keep=1).logits
            for t in range(split, input_ids.shape[1]):
                logits = model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits
    finally:
        model.config._attn_implementation = attn_impl
    return cache, logits


def greedy_generate(model, runtime, make_cache, input_ids, chunk_size, max_new_tokens, stop_ids, sim_len=None):
    if sim_len is None:
        cache, logits = runtime.aoh_prefill(model, make_cache, input_ids, chunk_size)
    else:
        cache, logits = flash_prefill(model, make_cache, input_ids, sim_len)
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = []
    with torch.no_grad(), runtime.math_sdpa_only():
        for _ in range(max_new_tokens):
            token_id = int(token.item())
            generated.append(token_id)
            if token_id in stop_ids:
                break
            output = model(input_ids=token, past_key_values=cache, use_cache=True)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    return generated


def run_task(
    task: str,
    model,
    tokenizer,
    runtime,
    make_cache,
    args,
    prompt_format: str,
    max_generation: int,
    max_length: int,
    device: str,
    output_path: Path,
) -> None:
    dataset = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for sample in tqdm(dataset, desc=task):
            prompt = prompt_format.format(**sample)
            prompt = truncate_middle(tokenizer, prompt, max_length)
            prompt = build_prompt(tokenizer, prompt, args.model, task)
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
            generated = greedy_generate(
                model,
                runtime,
                make_cache,
                input_ids,
                min(args.chunk_size, input_ids.shape[1]),
                max_generation,
                eos_ids(tokenizer, model),
                args.decoding_simulation_length if args.protocol == "aoh" else None,
            )
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            record = {
                "pred": prediction,
                "answers": sample["answers"],
                "all_classes": sample["all_classes"],
                "length": sample["length"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_json(CONFIG_DIR / "dataset2prompt.json")
    max_generations = load_json(CONFIG_DIR / "dataset2maxlen.json")
    tasks = select_tasks(args.tasks, list(prompts))
    config = MODEL_CONFIG[args.model]
    model_path = args.model_path or config["path"]
    max_length = args.max_length or config["max_length"]

    if not args.head_path.is_file():
        raise FileNotFoundError(f"AoH head map not found: {args.head_path}")
    labels = np.load(args.head_path)
    runtime = importlib.import_module(config["runtime"])
    if args.model.startswith("Llama"):
        model = runtime.load_model(model_path, args.gpus, args.per_gpu_gb, eager=True)
    else:
        model = runtime.load_model(
            model_path,
            args.gpus,
            args.per_gpu_gb,
            eager=True,
            max_position_embeddings=max_length + max(max_generations.values()),
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    make_cache, _, _ = runtime.patch_aoh_model(
        model, labels, args.sink_size, args.recent_size
    )
    device = runtime.primary_device(args.gpus)
    output_dir = args.output_dir / args.model / args.head_path.parent.name
    if args.protocol == "eager":
        output_dir = output_dir.with_name(output_dir.name + "_eager")

    for task in tasks:
        output_path = output_dir / f"{task}.jsonl"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path}")
            continue
        run_task(
            task,
            model,
            tokenizer,
            runtime,
            make_cache,
            args,
            prompts[task],
            max_generations[task],
            max_length,
            device,
            output_path,
        )
    print(f"Predictions written to {output_dir}")


if __name__ == "__main__":
    main()
