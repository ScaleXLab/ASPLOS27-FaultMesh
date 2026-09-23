#!/usr/bin/env bash
# Compile the GPGPU microbenchmarks against frontLib/frontend_prefault_common.cuh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/usr/local/cuda/bin:${PATH}"
ARCH="${ARCH:-sm_80}"
cd "${ROOT}/microbenchmark"
bash ./compile.sh
echo "Binaries are in ${ROOT}/microbenchmark/<app>/"
