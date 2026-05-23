#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:-/root/aslora-code/outputs/Llama-2-7b-hf-dual_scalar_gate/final}"

mkdir -p runs
uv run aslora eval \
    --config configs/dual_scalar_gate.yaml \
    --adapter "$ADAPTER" \
    2>&1 | tee "runs/eval_dual_scalar_gate.log"
