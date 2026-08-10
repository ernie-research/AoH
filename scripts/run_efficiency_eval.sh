#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
HEAD_ROOT="${HEAD_ROOT:-$REPO_ROOT/outputs/head_maps}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval/efficiency/outputs}"

QWEN3_HEAD_ROOT="${QWEN3_HEAD_ROOT:-${1:-$HEAD_ROOT/Qwen3-8B}}"
LLAMA31_HEAD_ROOT="${LLAMA31_HEAD_ROOT:-${2:-$HEAD_ROOT/Llama-3.1-8B-Instruct}}"
QWEN25_HEAD_ROOT="${QWEN25_HEAD_ROOT:-${3:-$HEAD_ROOT/Qwen2.5-7B}}"

QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3-8B}"
LLAMA31_MODEL="${LLAMA31_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN25_MODEL="${QWEN25_MODEL:-Qwen/Qwen2.5-7B}"
MODE="${MODE:-all}"
SPARSITIES="${SPARSITIES:-0.25,0.5,0.75}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-4096,8192,16384,32768,65536,131072,262144}"
GPUS="${GPUS:-cuda:0,cuda:1,cuda:2,cuda:3}"
PER_GPU_GB="${PER_GPU_GB:-72}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
REPETITIONS="${REPETITIONS:-1}"
DECODE_REPETITIONS="${DECODE_REPETITIONS:-1}"
DECODE_STEPS="${DECODE_STEPS:-20}"
SINK_SIZE="${SINK_SIZE:-128}"
RECENT_SIZE="${RECENT_SIZE:-256}"

if [[ "$MODE" == "all" || "$MODE" == "aoh" ]]; then
    for root in "$QWEN3_HEAD_ROOT" "$LLAMA31_HEAD_ROOT" "$QWEN25_HEAD_ROOT"; do
        if [[ ! -d "$root" ]]; then
            printf 'Missing head-map root: %s\n' "$root" >&2
            printf 'Pass three roots as arguments or set QWEN3_HEAD_ROOT, LLAMA31_HEAD_ROOT, and QWEN25_HEAD_ROOT.\n' >&2
            exit 1
        fi
    done

    IFS=',' read -ra sparsity_list <<< "$SPARSITIES"
    for root in "$QWEN3_HEAD_ROOT" "$LLAMA31_HEAD_ROOT" "$QWEN25_HEAD_ROOT"; do
        for sparsity in "${sparsity_list[@]}"; do
            head_file="$root/aoh_mean_sp${sparsity}/head_classification.npy"
            if [[ ! -f "$head_file" ]]; then
                printf 'Missing head classification file: %s\n' "$head_file" >&2
                exit 1
            fi
        done
    done
fi

command=(
    "$PYTHON_BIN" -m eval.efficiency.run_all
    --qwen3-model "$QWEN3_MODEL"
    --qwen3-head-root "$QWEN3_HEAD_ROOT"
    --llama31-model "$LLAMA31_MODEL"
    --llama31-head-root "$LLAMA31_HEAD_ROOT"
    --qwen25-model "$QWEN25_MODEL"
    --qwen25-head-root "$QWEN25_HEAD_ROOT"
    --mode "$MODE"
    --sparsities "$SPARSITIES"
    --context-lengths "$CONTEXT_LENGTHS"
    --gpus "$GPUS"
    --per-gpu-gb "$PER_GPU_GB"
    --chunk-size "$CHUNK_SIZE"
    --repetitions "$REPETITIONS"
    --decode-repetitions "$DECODE_REPETITIONS"
    --decode-steps "$DECODE_STEPS"
    --sink-size "$SINK_SIZE"
    --recent-size "$RECENT_SIZE"
    --output-dir "$OUTPUT_DIR"
)

if [[ "${FORCE:-0}" == "1" ]]; then
    command+=(--force)
fi

"${command[@]}"

printf 'Efficiency results written to %s\n' "$OUTPUT_DIR"
