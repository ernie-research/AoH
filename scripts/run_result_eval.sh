#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
HEAD_ROOT="${HEAD_ROOT:-$REPO_ROOT/outputs/head_maps}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval/results/outputs}"

QWEN3_HEAD="${QWEN3_HEAD:-${1:-$HEAD_ROOT/Qwen3-8B/aoh_mean_sp0.5/head_classification.npy}}"
LLAMA31_HEAD="${LLAMA31_HEAD:-${2:-$HEAD_ROOT/Llama-3.1-8B-Instruct/aoh_mean_sp0.5/head_classification.npy}}"
QWEN25_HEAD="${QWEN25_HEAD:-${3:-$HEAD_ROOT/Qwen2.5-7B/aoh_mean_sp0.5/head_classification.npy}}"

QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3-8B}"
LLAMA31_MODEL="${LLAMA31_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN25_MODEL="${QWEN25_MODEL:-Qwen/Qwen2.5-7B}"
TASKS="${TASKS:-all}"
GPUS="${GPUS:-cuda:0,cuda:1,cuda:2,cuda:3}"
PER_GPU_GB="${PER_GPU_GB:-72}"

for path in "$QWEN3_HEAD" "$LLAMA31_HEAD" "$QWEN25_HEAD"; do
    if [[ ! -f "$path" ]]; then
        printf 'Missing head classification file: %s\n' "$path" >&2
        printf 'Pass three paths as arguments or set QWEN3_HEAD, LLAMA31_HEAD, and QWEN25_HEAD.\n' >&2
        exit 1
    fi
done

command=(
    "$PYTHON_BIN" -m eval.results.run_all
    --qwen3-head "$QWEN3_HEAD"
    --qwen3-model "$QWEN3_MODEL"
    --llama31-head "$LLAMA31_HEAD"
    --llama31-model "$LLAMA31_MODEL"
    --qwen25-head "$QWEN25_HEAD"
    --qwen25-model "$QWEN25_MODEL"
    --tasks "$TASKS"
    --output-dir "$OUTPUT_DIR"
    --gpus "$GPUS"
    --per-gpu-gb "$PER_GPU_GB"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    command+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    command+=(--overwrite)
fi

"${command[@]}"

models=("Qwen3-8B" "Llama-3.1-8B-Instruct" "Qwen2.5-7B")
heads=("$QWEN3_HEAD" "$LLAMA31_HEAD" "$QWEN25_HEAD")
for index in "${!models[@]}"; do
    head_name="$(basename "$(dirname "${heads[$index]}")")"
    prediction_dir="$OUTPUT_DIR/${models[$index]}/$head_name"
    "$PYTHON_BIN" -m eval.results.evaluate "$prediction_dir"
done

printf 'LongBench results written to %s\n' "$OUTPUT_DIR"
