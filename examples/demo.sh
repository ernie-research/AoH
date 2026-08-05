#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

## Full Attention
CUDA_VISIBLE_DEVICES=0 \
python "$PROJECT_ROOT/examples/run_demo_llama31.py" \
--sp 0.0 \
--input-length 128000 \
--prompt $'Read the entire technical report and answer the question using evidence from the document.\n\nQuestion:\nWhat was the root cause of the latency spike in Project Aurora?\n\nAnswer briefly and cite the relevant section.'


## AoH Attention
CUDA_VISIBLE_DEVICES=1 \
python "$PROJECT_ROOT/examples/run_demo_llama31.py" \
--sp 0.5 \
--input-length 128000 \
--prompt $'Read the entire technical report and answer the question using evidence from the document.\n\nQuestion:\nWhat was the root cause of the latency spike in Project Aurora?\n\nAnswer briefly and cite the relevant section.'
