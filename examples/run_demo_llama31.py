#!/usr/bin/env python3
"""Terminal demo comparing Full Attention and AoH on Llama-3.1-8B-Instruct."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_ROOT / "experiments" / "block2_efficiency"
sys.path.insert(0, str(BENCH_DIR))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import bench_llama31_aoh_optimized as bench
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This demo requires experiments/block2_efficiency/"
        "bench_llama31_aoh_optimized.py, but that file is not included in this "
        "repository archive. Add it under the project root before running the demo."
    ) from exc

DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_HEAD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "llama31_8b_aoh_mean_sp0.5"
    / "head_classification.npy"
)


DEFAULT_PROMPT = """Read the entire technical report and answer the question using evidence from the document.

Question:
What was the root cause of the latency spike in Project Aurora?

Answer briefly and cite the relevant section."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sp",
        required=True,
        dest="sparsity",
        type=float,
        choices=(0.0, 0.5),
        help="0.0 uses Full Attention; 0.5 uses AoH.",
    )
    parser.add_argument(
        "--input-length",
        required=True,
        dest="demo_input_tokens",
        type=int,
        help="Synthetic report length in tokens.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Question appended to the synthetic technical report.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--head-path",
        type=Path,
        default=DEFAULT_HEAD_PATH,
        help="AoH classification file (relative paths are resolved from project root).",
    )
    parser.set_defaults(
        document_file=None,
        display_delay_ms=0.0,
        gpus="cuda:0",
        per_gpu_gb=120,
        max_new_tokens=100,
        chunk_size=1024,
        sink_size=128,
        recent_size=256,
        raw_prompt=False,
    )
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def cuda_memory_gib(kind: str) -> float:
    total = 0
    for device_idx in range(torch.cuda.device_count()):
        if kind == "allocated":
            total += torch.cuda.memory_allocated(device_idx)
        elif kind == "peak":
            total += torch.cuda.max_memory_allocated(device_idx)
        else:
            raise ValueError(f"Unknown memory kind: {kind}")
    return total / 1024**3


def compose_prompt(
    tokenizer,
    prompt: str,
    document_file: Path | None,
    demo_input_tokens: int,
) -> str:
    if document_file is not None and demo_input_tokens > 0:
        raise ValueError("Use either --document-file or --demo-input-tokens, not both.")
    if document_file is not None:
        if not document_file.is_file():
            raise FileNotFoundError(f"Document file does not exist: {document_file}")
        document = document_file.read_text(encoding="utf-8")
        return f"<technical_report>\n{document}\n</technical_report>\n\n{prompt}"
    if demo_input_tokens <= 0:
        return prompt
    if demo_input_tokens < 1024:
        raise ValueError("--demo-input-tokens must be at least 1024.")

    report_header = (
        "<technical_report>\n"
        "Project Aurora Production Latency Incident - Technical Report\n"
        "This synthetic report is used only for the AoH terminal demonstration.\n\n"
    )
    root_cause = (
        "\nSection 4.3 Root Cause Analysis\n"
        "The latency spike was caused by an expired service-discovery cache that "
        "continued routing 38 percent of requests to a decommissioned database shard. "
        "Those requests repeatedly retried, exhausted the connection pool, and created "
        "a queue backlog across the Aurora API tier. The retry storm, rather than CPU "
        "saturation or network bandwidth, was the direct root cause.\n\n"
    )
    report_footer = "\nSection 7 Conclusion\nEnd of technical report.\n</technical_report>\n\n"
    filler = "".join(
        f"Appendix {index:06d}: archive={index * 7919 % 1000003:06d}; "
        f"sample={index * 104729 % 1000003:06d}.\n"
        for index in range(max(256, demo_input_tokens // 6))
    )

    fixed_text = report_header + root_cause + report_footer + prompt
    fixed_tokens = len(tokenizer.encode(fixed_text, add_special_tokens=False))
    padding_count = max(0, demo_input_tokens - fixed_tokens)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False, verbose=False)
    repeated_ids = (filler_ids * (padding_count // len(filler_ids) + 1))[:padding_count]
    filler_text = tokenizer.decode(repeated_ids, skip_special_tokens=True)
    return report_header + root_cause + filler_text + report_footer + prompt


def encode_prompt(tokenizer, prompt: str, raw_prompt: bool) -> torch.Tensor:
    if raw_prompt:
        return tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return encoded.input_ids if hasattr(encoded, "input_ids") else encoded


def prefill_with_progress(
    model,
    input_ids: torch.Tensor,
    mode: str,
    make_cache,
    chunk_size: int,
):
    total_tokens = input_ids.shape[1]
    cache = make_cache() if mode == "aoh" else None
    last_logits = None
    progress = tqdm(
        range(0, total_tokens, chunk_size),
        dynamic_ncols=True,
        leave=True,
        unit="chunk",
    )

    synchronize()
    started = time.perf_counter()
    with torch.no_grad(), bench.math_sdpa_only():
        for start in progress:
            chunk = input_ids[:, start : start + chunk_size]
            position_ids = torch.arange(
                start,
                start + chunk.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
            output = model(
                input_ids=chunk,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            if mode == "full":
                cache = output.past_key_values
            last_logits = output.logits
            completed = min(start + chunk_size, total_tokens)
            progress.set_description(
                f"Pre-filling ({completed // 1000}K/{total_tokens // 1000}K, "
                f"Mem: {cuda_memory_gib('peak'):.1f} GiB)"
            )
    synchronize()
    elapsed = time.perf_counter() - started
    progress.close()
    return cache, last_logits, elapsed


def decode_with_progress(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cache,
    last_logits: torch.Tensor,
    max_new_tokens: int,
    display_delay_ms: float,
):
    current = last_logits[:, -1:, :].argmax(dim=-1)
    generated = [int(current.item())]
    eos_token_ids = tokenizer.eos_token_id
    eos_token_ids = set(
        eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]
    )
    total_latency_ms = 0.0
    measured_tokens = 0

    first_piece = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(first_piece.replace("\n", " "), end="", flush=True)
    with torch.no_grad(), bench.math_sdpa_only():
        for step in range(1, max_new_tokens):
            if generated[-1] in eos_token_ids:
                break
            position_ids = torch.tensor(
                [[input_ids.shape[1] + step - 1]],
                dtype=torch.long,
                device=input_ids.device,
            )
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = model(
                input_ids=current,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            end_event.record()
            synchronize()
            total_latency_ms += start_event.elapsed_time(end_event)
            measured_tokens += 1

            cache = output.past_key_values
            current = output.logits[:, -1:, :].argmax(dim=-1)
            token_id = int(current.item())
            generated.append(token_id)
            piece = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(piece.replace("\n", " "), end="", flush=True)
            if display_delay_ms > 0:
                time.sleep(display_delay_ms / 1000)

    print()
    return total_latency_ms, measured_tokens


def main() -> None:
    args = parse_args()
    mode = "full" if args.sparsity == 0.0 else "aoh"
    if not args.head_path.is_absolute():
        args.head_path = PROJECT_ROOT / args.head_path

    if mode == "aoh" and not args.head_path.is_file():
        raise FileNotFoundError(f"AoH head classification does not exist: {args.head_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison.")
    if args.display_delay_ms < 0:
        raise ValueError("--display-delay-ms cannot be negative.")

    title = "Full Attention" if mode == "full" else "AoH"
    print("=" * 76)
    print(f"{title} | Llama-3.1-8B-Instruct (128K) | sparsity={args.sparsity}")
    if mode == "aoh":
        print(f"Using AoH with {args.sparsity} sparsity.")
        print(f"Streaming cache: sink={args.sink_size}, recent={args.recent_size}")
    else:
        print("Using Full Attention.")
    print("=" * 76, flush=True)

    bench.MODEL_PATH = args.model_id
    model = bench.load_model(
        args.gpus,
        args.per_gpu_gb,
        eager=mode == "aoh",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    prompt = compose_prompt(
        tokenizer,
        args.prompt,
        args.document_file,
        args.demo_input_tokens,
    )

    make_cache = None
    if mode == "aoh":
        labels = np.load(args.head_path)
        make_cache, _, _ = bench.patch_aoh_model(
            model,
            labels,
            args.sink_size,
            args.recent_size,
        )

    synchronize()
    input_ids = encode_prompt(tokenizer, prompt, args.raw_prompt).to(
        next(model.parameters()).device
    )
    if input_ids.shape[1] > 131000:
        raise ValueError(
            f"Input has {input_ids.shape[1]} tokens; limit is 131000 tokens."
        )

    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)

    print("\n" + "=" * 76)
    print("STAGE 1/2: PREFILL")
    print("=" * 76, flush=True)
    cache, last_logits, prefill_seconds = prefill_with_progress(
        model,
        input_ids,
        mode,
        make_cache,
        args.chunk_size,
    )
    prefill_peak_vram_gib = cuda_memory_gib("peak")
    print("\nPREFILL SUMMARY")
    print(f"Memory     : {prefill_peak_vram_gib:.2f} GiB")
    print(f"Total Time : {prefill_seconds:.2f}s")

    print("\n" + "=" * 76)
    print("STAGE 2/2: DECODE")
    print("=" * 76, flush=True)
    decode_latency_ms, measured_tokens = decode_with_progress(
        model,
        tokenizer,
        input_ids,
        cache,
        last_logits,
        args.max_new_tokens,
        args.display_delay_ms,
    )
    decode_ms_per_token = decode_latency_ms / max(measured_tokens, 1)
    peak_vram_gib = cuda_memory_gib("peak")

    print(f"\n\nPer-token decoding latency: {decode_ms_per_token:.2f} ms")
    print(f"Peak memory: {peak_vram_gib:.2f} GiB")


if __name__ == "__main__":
    main()
