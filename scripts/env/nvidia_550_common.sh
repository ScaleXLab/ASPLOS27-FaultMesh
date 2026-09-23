# Shared paths for the 550.54.14 driver and CUDA 12.4 userspace.
# Source this file; do not execute it.

NVIDIA_550_VERSION="550.54.14"
CUDA_TOOLKIT_VERSION="12.4.0"

DRIVER_URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_550_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_550_VERSION}.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_TOOLKIT_VERSION}/local_installers/cuda_${CUDA_TOOLKIT_VERSION}_${NVIDIA_550_VERSION}_linux.run"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
DOWNLOAD_DIR="${THIRD_PARTY}/download"
DRIVER_RUN="${DOWNLOAD_DIR}/NVIDIA-Linux-x86_64-${NVIDIA_550_VERSION}.run"
CUDA_RUN="${DOWNLOAD_DIR}/cuda_${CUDA_TOOLKIT_VERSION}_${NVIDIA_550_VERSION}_linux.run"
DRIVER_EXTRACT="${THIRD_PARTY}/nvidia-${NVIDIA_550_VERSION}-extract"
USERSPACE_DIR="${THIRD_PARTY}/nvidia-${NVIDIA_550_VERSION}"
CUDA_PREFIX="${REPO_ROOT}/cuda-toolkit"
SNAPSHOT_DIR="${THIRD_PARTY}/host-snapshot"

# 64-bit driver libraries a CUDA process actually loads.
USERSPACE_LIBS=(
  "libcuda.so.${NVIDIA_550_VERSION}"
  "libnvidia-ml.so.${NVIDIA_550_VERSION}"
  "libnvidia-ptxjitcompiler.so.${NVIDIA_550_VERSION}"
  "libcudadebugger.so.${NVIDIA_550_VERSION}"
)

LIB_DIRS=(
  "/usr/lib/x86_64-linux-gnu"
  "/lib/x86_64-linux-gnu"
)

# Symlinks the switch retargets at the extracted 550 libraries.
LIB_LINKS=(
  "libcuda.so.1"
  "libcuda.so"
  "libnvidia-ml.so.1"
  "libnvidia-ml.so"
  "libnvidia-ptxjitcompiler.so.1"
  "libnvidia-ptxjitcompiler.so"
)

log() { printf '[nvidia-550] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

download_file() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "${dest}")"
  if [[ -f "${dest}" ]]; then
    log "already downloaded: ${dest}"
    return 0
  fi
  log "download ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "${dest}.partial" "${url}"
  else
    curl -fL --retry 3 -o "${dest}.partial" "${url}"
  fi
  mv "${dest}.partial" "${dest}"
}

stage_driver_userspace() {
  [[ -x "${DRIVER_RUN}" || -f "${DRIVER_RUN}" ]] || die "missing ${DRIVER_RUN}"
  if [[ ! -f "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" ]]; then
    log "extract driver runfile"
    rm -rf "${DRIVER_EXTRACT}"
    sh "${DRIVER_RUN}" --extract-only --target "${DRIVER_EXTRACT}"
    mkdir -p "${USERSPACE_DIR}/lib" "${USERSPACE_DIR}/bin" "${USERSPACE_DIR}/firmware"
    local lib
    for lib in "${USERSPACE_LIBS[@]}"; do
      [[ -f "${DRIVER_EXTRACT}/${lib}" ]] || die "runfile has no ${lib}"
      cp -a "${DRIVER_EXTRACT}/${lib}" "${USERSPACE_DIR}/lib/${lib}"
    done
    cp -a "${DRIVER_EXTRACT}/nvidia-smi" "${USERSPACE_DIR}/bin/nvidia-smi"
    chmod +x "${USERSPACE_DIR}/bin/nvidia-smi"
    cp -a "${DRIVER_EXTRACT}/firmware/gsp_ga10x.bin" "${USERSPACE_DIR}/firmware/"
    cp -a "${DRIVER_EXTRACT}/firmware/gsp_tu10x.bin" "${USERSPACE_DIR}/firmware/"
    ln -sfn "libcuda.so.${NVIDIA_550_VERSION}" "${USERSPACE_DIR}/lib/libcuda.so.1"
    ln -sfn "libcuda.so.1" "${USERSPACE_DIR}/lib/libcuda.so"
    ln -sfn "libnvidia-ml.so.${NVIDIA_550_VERSION}" "${USERSPACE_DIR}/lib/libnvidia-ml.so.1"
    ln -sfn "libnvidia-ml.so.1" "${USERSPACE_DIR}/lib/libnvidia-ml.so"
    ln -sfn "libnvidia-ptxjitcompiler.so.${NVIDIA_550_VERSION}" "${USERSPACE_DIR}/lib/libnvidia-ptxjitcompiler.so.1"
    ln -sfn "libnvidia-ptxjitcompiler.so.1" "${USERSPACE_DIR}/lib/libnvidia-ptxjitcompiler.so"
    rm -rf "${DRIVER_EXTRACT}"
  fi
}

use_repo_cuda() {
  export PATH="${CUDA_PREFIX}/bin:${USERSPACE_DIR}/bin:${PATH}"
  export LD_LIBRARY_PATH="${USERSPACE_DIR}/lib:${CUDA_PREFIX}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

install_cuda_toolkit() {
  if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
    return 0
  fi
  [[ -f "${CUDA_RUN}" ]] || die "missing ${CUDA_RUN}; run scripts/env/download_nvidia_550.sh"
  log "install CUDA ${CUDA_TOOLKIT_VERSION} toolkit into ${CUDA_PREFIX}"
  sh "${CUDA_RUN}" --silent --toolkit --toolkitpath="${CUDA_PREFIX}" --override
  [[ -x "${CUDA_PREFIX}/bin/nvcc" ]] || die "CUDA toolkit install did not produce ${CUDA_PREFIX}/bin/nvcc"
}

install_gsp_firmware() {
  local dest="/lib/firmware/nvidia/${NVIDIA_550_VERSION}"
  mkdir -p "${dest}"
  cp -a "${USERSPACE_DIR}/firmware/gsp_ga10x.bin" "${dest}/"
  cp -a "${USERSPACE_DIR}/firmware/gsp_tu10x.bin" "${dest}/"
}

unique_lib_dirs() {
  local dir real seen=""
  for dir in "${LIB_DIRS[@]}"; do
    [[ -d "${dir}" ]] || continue
    real="$(readlink -f "${dir}")"
    case " ${seen} " in
      *" ${real} "*) continue ;;
    esac
    seen="${seen} ${real}"
    printf '%s\n' "${dir}"
  done
}
