#!/usr/bin/env bash
# Compile the GPGPU applications against frontLib/frontend_prefault_common.cuh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/env/nvidia_550_common.sh"
use_repo_cuda
ARCH="${ARCH:-sm_80}"
cd "${ROOT}/application"
bash ./compile.sh
echo "Binaries are in ${ROOT}/application/<app>/"
