#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
uv run aslora train --config configs/dual_scalar_gate.yaml 2>&1 | tee "runs/dual_scalar_gate.log"
