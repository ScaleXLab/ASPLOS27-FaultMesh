#!/usr/bin/env bash
# Download the 550.54.14 driver (libcuda / NVML / nvidia-smi / GSP firmware)
# and the matching CUDA 12.4 toolkit. Nothing is installed into /usr.
#
#   bash scripts/download_nvidia_550.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/nvidia_550_common.sh"

if host_has_550_cuda_lib; then
  log "libcuda ${NVIDIA_550_VERSION} is already installed. Nothing to download."
  exit 0
fi

log "driver: ${DRIVER_URL}"
download_file "${DRIVER_URL}" "${DRIVER_RUN}"
stage_driver_userspace

log "CUDA toolkit: ${CUDA_URL}"
download_file "${CUDA_URL}" "${CUDA_RUN}"
install_cuda_toolkit

log "550 userspace: ${USERSPACE_DIR}"
if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
  log "CUDA toolkit:  ${CUDA_PREFIX}"
  "${CUDA_PREFIX}/bin/nvcc" --version
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  log "CUDA toolkit:  /usr/local/cuda"
  /usr/local/cuda/bin/nvcc --version
else
  die "nvcc was not installed"
fi
if [[ -x "${USERSPACE_DIR}/bin/nvidia-smi" ]]; then
  "${USERSPACE_DIR}/bin/nvidia-smi" --version || true
fi
