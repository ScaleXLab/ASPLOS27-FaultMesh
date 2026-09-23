#!/usr/bin/env bash
# Load the confirmed FaultMesh BackLib:
#   parallel fault processing, async copy-map on, merge dispatch off.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export BUILD_KERNEL="${BUILD_KERNEL:-1}"
cd "${ROOT}/backLib"
exec bash ./perf_ours.sh "$@"
