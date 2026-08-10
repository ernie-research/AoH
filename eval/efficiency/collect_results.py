#!/usr/bin/env python3
"""Merge per-model efficiency comparisons into paper-friendly CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODEL_DIRS = ["Qwen3-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B"]


def collect(root: Path, phase: str) -> list[dict]:
    rows = []
    for model in MODEL_DIRS:
        path = root / model / "comparison.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        for row in data.get(phase, []):
            rows.append({"model": model, "phase": phase, **row})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    for phase in ("prefill", "decode"):
        rows = collect(args.output_dir, phase)
        path = args.output_dir / f"all_models_{phase}.csv"
        write_csv(path, rows)
        print(f"{phase}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
