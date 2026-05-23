#!/usr/bin/env bash
set -euo pipefail

uv run aslora eval \
  --config configs/dual_scalar_gate.yaml \
  --adapter twanghcmut/Llama-2-7b-hf-aslora-dual_scalar_gate \
  2>&1 | tee runs/arc_easy_dual_scalar_gate.log
