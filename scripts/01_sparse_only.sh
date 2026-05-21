#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
uv run aslora train --config configs/sparse_only.yaml 2>&1 | tee "runs/sparse_only.log"
