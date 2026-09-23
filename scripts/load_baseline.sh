#!/usr/bin/env bash
# Load the UVM baseline: parallel faults off, async copy-map off, merge dispatch off.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/backLib"
exec bash ./perf_baseline.sh "$@"
