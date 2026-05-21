#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
uv run aslora train --config configs/dense_only.yaml 2>&1 | tee "runs/dense_only.log"
