#!/usr/bin/env bash
# Download the 550.54.14 driver (libcuda / NVML / nvidia-smi / GSP firmware)
# and the matching CUDA 12.4 toolkit. Nothing is installed into /usr.
#
#   bash scripts/download_nvidia_550.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/nvidia_550_common.sh"

log "driver: ${DRIVER_URL}"
download_file "${DRIVER_URL}" "${DRIVER_RUN}"
stage_driver_userspace

log "CUDA toolkit: ${CUDA_URL}"
download_file "${CUDA_URL}" "${CUDA_RUN}"
install_cuda_toolkit

log "550 userspace: ${USERSPACE_DIR}"
log "CUDA toolkit:  ${CUDA_PREFIX}"
"${USERSPACE_DIR}/bin/nvidia-smi" --version || true
"${CUDA_PREFIX}/bin/nvcc" --version
