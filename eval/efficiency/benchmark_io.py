"""Progress-file helpers for resumable efficiency sweeps."""
from __future__ import annotations

import json
from pathlib import Path


def load_progress(path: Path, config: dict, force: bool) -> dict:
    if force or not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if data.get("config") != config:
        raise ValueError(
            f"Existing benchmark configuration differs in {path}; "
            "use --force or choose another --out_dir"
        )
    return data


def save_progress(
    path: Path,
    config: dict,
    prefill: list[dict],
    decode: list[dict],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            {"config": config, "prefill": prefill, "decode": decode},
            file,
            indent=2,
        )
    temporary.replace(path)


def benchmark_config(args, method: str, head_path: Path | None = None) -> dict:
    config = {
        "model_path": str(args.model_path),
        "method": method,
        "gpus": args.gpus,
        "per_gpu_gb": args.per_gpu_gb,
        "chunk_size": args.chunk_size,
        "prefill_lens": args.prefill_lens,
        "decode_lens": args.decode_lens,
        "prefill_repetitions": args.n_reps,
        "decode_repetitions": args.n_decode_reps,
        "decode_steps": args.n_decode_steps,
    }
    if head_path is not None:
        config.update(
            {
                "head_path": str(head_path.resolve()),
                "sink_size": args.sink_size,
                "recent_size": args.recent_size,
            }
        )
    return config
