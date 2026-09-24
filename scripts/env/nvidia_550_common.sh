# Shared paths for the 550.54.14 driver and CUDA 12.4 userspace.
# Source this file; do not execute it.

NVIDIA_550_VERSION="550.54.14"
CUDA_TOOLKIT_VERSION="12.4.0"

DRIVER_URL="https://download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_550_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_550_VERSION}.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_TOOLKIT_VERSION}/local_installers/cuda_${CUDA_TOOLKIT_VERSION}_${NVIDIA_550_VERSION}_linux.run"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
# The two .run installers stay in the repository root. Unpacked libraries and
# the toolkit go in cuda-toolkit/.
DOWNLOAD_DIR="${REPO_ROOT}"
DRIVER_RUN="${DOWNLOAD_DIR}/NVIDIA-Linux-x86_64-${NVIDIA_550_VERSION}.run"
CUDA_RUN="${DOWNLOAD_DIR}/cuda_${CUDA_TOOLKIT_VERSION}_${NVIDIA_550_VERSION}_linux.run"
USERSPACE_DIR="${REPO_ROOT}/cuda-toolkit"
CUDA_PREFIX="${REPO_ROOT}/cuda-toolkit"
SNAPSHOT_DIR="${THIRD_PARTY}/host-snapshot"
CONDA_ENV_DIR="${REPO_ROOT}/.conda/faultmesh-550"

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

# True when the CUDA library already loaded by this machine is 550.54.14.
host_has_550_cuda_lib() {
  local path real
  local candidates=(
    /usr/lib/x86_64-linux-gnu/libcuda.so.1
    /lib/x86_64-linux-gnu/libcuda.so.1
    /usr/lib64/libcuda.so.1
    "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_550_VERSION}"
    "/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_550_VERSION}"
  )
  for path in "${candidates[@]}"; do
    [[ -e "${path}" ]] || continue
    real="$(readlink -f "${path}" 2>/dev/null || echo "${path}")"
    [[ "${real}" == *"libcuda.so.${NVIDIA_550_VERSION}" ]] && return 0
  done
  ldconfig -p 2>/dev/null | grep -q "libcuda.so.1 => .*libcuda.so.${NVIDIA_550_VERSION}"
}

download_file() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "${dest}")"
  if [[ -f "${dest}" ]]; then
    log "already downloaded: ${dest}"
    return 0
  fi
  log "download ${url}"
  local ua='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  if command -v wget >/dev/null 2>&1; then
    wget -c --user-agent="${ua}" --referer='https://www.nvidia.com/' -O "${dest}.partial" "${url}"
  else
    curl -fL --retry 3 -A "${ua}" -e 'https://www.nvidia.com/' -o "${dest}.partial" "${url}"
  fi
  mv "${dest}.partial" "${dest}"
}

stage_driver_userspace() {
  [[ -x "${DRIVER_RUN}" || -f "${DRIVER_RUN}" ]] || die "missing ${DRIVER_RUN}"
  if [[ ! -f "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" ]]; then
    log "extract driver runfile, then keep only libcuda, nvidia-smi, and GSP firmware"
    mkdir -p "${USERSPACE_DIR}"
    local extract="${USERSPACE_DIR}/.extract"
    rm -rf "${extract}"
    sh "${DRIVER_RUN}" --extract-only --target "${extract}"
    mkdir -p "${USERSPACE_DIR}/lib" "${USERSPACE_DIR}/bin" "${USERSPACE_DIR}/firmware"
    local lib
    for lib in "${USERSPACE_LIBS[@]}"; do
      [[ -f "${extract}/${lib}" ]] || die "runfile has no ${lib}"
      cp -a "${extract}/${lib}" "${USERSPACE_DIR}/lib/${lib}"
    done
    cp -a "${extract}/nvidia-smi" "${USERSPACE_DIR}/bin/nvidia-smi"
    chmod +x "${USERSPACE_DIR}/bin/nvidia-smi"
    cp -a "${extract}/firmware/gsp_ga10x.bin" "${USERSPACE_DIR}/firmware/"
    cp -a "${extract}/firmware/gsp_tu10x.bin" "${USERSPACE_DIR}/firmware/"
    ln -sfn "libcuda.so.${NVIDIA_550_VERSION}" "${USERSPACE_DIR}/lib/libcuda.so.1"
    ln -sfn "libcuda.so.1" "${USERSPACE_DIR}/lib/libcuda.so"
    ln -sfn "libnvidia-ml.so.${NVIDIA_550_VERSION}" "${USERSPACE_DIR}/lib/libnvidia-ml.so.1"
    ln -sfn "libnvidia-ml.so.1" "${USERSPACE_DIR}/lib/libnvidia-ml.so"
    ln -sfn "libnvidia-ptxjitcompiler.so.${NVIDIA_550_VERSION}" "${USERSPACE_DIR}/lib/libnvidia-ptxjitcompiler.so.1"
    ln -sfn "libnvidia-ptxjitcompiler.so.1" "${USERSPACE_DIR}/lib/libnvidia-ptxjitcompiler.so"
    rm -rf "${extract}"
  fi
}

use_repo_cuda() {
  local nvcc_bin="" lib_path=""
  if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
    nvcc_bin="${CUDA_PREFIX}/bin"
    [[ -d "${CUDA_PREFIX}/lib64" ]] && lib_path="${CUDA_PREFIX}/lib64"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin="/usr/local/cuda/bin"
  fi
  if [[ -n "${nvcc_bin}" ]]; then
    export PATH="${nvcc_bin}:${USERSPACE_DIR}/bin:${PATH}"
  else
    export PATH="${USERSPACE_DIR}/bin:${PATH}"
  fi
  if [[ -e "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" ]]; then
    lib_path="${USERSPACE_DIR}/lib${lib_path:+:${lib_path}}"
  fi
  if [[ -d "${CONDA_ENV_DIR}/lib" ]]; then
    lib_path="${CONDA_ENV_DIR}/lib${lib_path:+:${lib_path}}"
    export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
  fi
  if [[ -n "${lib_path}" ]]; then
    export LD_LIBRARY_PATH="${lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

require_nvcc() {
  use_repo_cuda
  if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]] && /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -q 'release 12.4'; then
    return 0
  fi
  die "CUDA 12.4 nvcc was not found. Run: bash scripts/env/download_nvidia_550.sh"
}

# Userspace 550 libraries live in a conda prefix inside this repository.
# The kernel module switch still changes the machine; these libraries do not.
find_conda() {
  local candidate home_dir
  home_dir="${HOME}"
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    home_dir="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
  fi
  for candidate in \
    "$(command -v conda 2>/dev/null || true)" \
    "${home_dir}/miniconda3/bin/conda" \
    "${home_dir}/anaconda3/bin/conda" \
    /opt/conda/bin/conda; do
    [[ -n "${candidate}" && -x "${candidate}" ]] && { printf '%s\n' "${candidate}"; return 0; }
  done
  return 1
}

install_miniconda() {
  local dest="${HOME}/miniconda3"
  local installer="${TMPDIR:-/tmp}/Miniconda3-latest-Linux-x86_64.sh"
  log "conda was not found. Installing Miniconda to ${dest}"
  curl -fL --retry 3 -o "${installer}" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "${installer}" -b -p "${dest}"
  "${dest}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  "${dest}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  "${dest}/bin/conda" init bash >/dev/null
  printf '%s\n' "${dest}/bin/conda"
}

setup_conda_userspace() {
  local libcuda="${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}"
  [[ -e "${libcuda}" ]] || die "missing ${libcuda}. Run: bash scripts/env/download_nvidia_550.sh"
  local conda_bin=""
  conda_bin="$(find_conda || true)"
  if [[ -z "${conda_bin}" ]]; then
    conda_bin="$(install_miniconda)"
  fi
  log "using conda: ${conda_bin}"
  if [[ ! -f "${CONDA_ENV_DIR}/conda-meta/history" ]]; then
    "${conda_bin}" create -y --prefix "${CONDA_ENV_DIR}"
  fi
  mkdir -p "${CONDA_ENV_DIR}/lib" "${CONDA_ENV_DIR}/bin" \
    "${CONDA_ENV_DIR}/etc/conda/activate.d" "${CONDA_ENV_DIR}/etc/conda/deactivate.d"
  local name
  for name in libcuda libnvidia-ml libnvidia-ptxjitcompiler libcudadebugger; do
    cp -a "${USERSPACE_DIR}/lib/${name}.so.${NVIDIA_550_VERSION}" "${CONDA_ENV_DIR}/lib/"
    ln -sfn "${name}.so.${NVIDIA_550_VERSION}" "${CONDA_ENV_DIR}/lib/${name}.so.1"
    ln -sfn "${name}.so.1" "${CONDA_ENV_DIR}/lib/${name}.so"
  done
  cp -a "${USERSPACE_DIR}/bin/nvidia-smi" "${CONDA_ENV_DIR}/bin/nvidia-smi"
  chmod +x "${CONDA_ENV_DIR}/bin/nvidia-smi"
  cat > "${CONDA_ENV_DIR}/etc/conda/activate.d/faultmesh-550.sh" <<EOF
export PATH="${CONDA_ENV_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
  cat > "${CONDA_ENV_DIR}/etc/conda/deactivate.d/faultmesh-550.sh" <<EOF
export PATH="\${PATH//${CONDA_ENV_DIR}\/bin:/}"
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH//${CONDA_ENV_DIR}\/lib:/}"
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH//${CONDA_ENV_DIR}\/lib/}"
EOF
  log "conda userspace: ${CONDA_ENV_DIR}"
}

# A previous toolkit install into the repository can stop halfway and leave
# symlinks behind. The NVIDIA installer aborts when it tries to create them again.
clean_incomplete_toolkit() {
  [[ -x "${CUDA_PREFIX}/bin/nvcc" ]] && return 0
  local item
  for item in targets include lib64 nvvm gds compute-sanitizer extras share doc DOCS pkgconfig src version.json EULA.txt; do
    rm -rf "${CUDA_PREFIX}/${item}"
  done
  rm -rf "${CUDA_PREFIX}"/nsight-compute-* "${CUDA_PREFIX}"/nsight-systems-*
  if [[ -d "${CUDA_PREFIX}/bin" ]]; then
    find "${CUDA_PREFIX}/bin" -mindepth 1 ! -name 'nvidia-smi' -exec rm -rf {} +
  fi
}

install_cuda_toolkit() {
  if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]] && /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -q 'release 12.4'; then
    log "CUDA 12.4 nvcc is already available at /usr/local/cuda/bin/nvcc"
    return 0
  fi
  [[ -f "${CUDA_RUN}" ]] || die "missing ${CUDA_RUN}; run scripts/env/download_nvidia_550.sh"
  clean_incomplete_toolkit
  log "install CUDA ${CUDA_TOOLKIT_VERSION} toolkit into ${CUDA_PREFIX}"
  if ! sh "${CUDA_RUN}" --silent --toolkit --toolkitpath="${CUDA_PREFIX}" --override; then
    echo "CUDA installer failed. Last errors from /tmp/cuda-installer.log:" >&2
    grep -E '\[ERROR\]' /tmp/cuda-installer.log 2>/dev/null | tail -20 >&2 || true
    die "CUDA toolkit install failed"
  fi
  [[ -x "${CUDA_PREFIX}/bin/nvcc" ]] || die "CUDA toolkit install did not produce ${CUDA_PREFIX}/bin/nvcc"
}

# Point the system CUDA library at the 550.54.14 files in this repository.
# ldconfig applies immediately, so the current shell's next process sees it.
bind_550_userspace() {
  [[ "$(id -u)" -eq 0 ]] || die "bind_550_userspace must run as root"
  local libcuda="${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}"
  [[ -e "${libcuda}" ]] || die "missing ${libcuda}. Run: bash scripts/env/download_nvidia_550.sh"
  local dir
  while read -r dir; do
    ln -sfn "${libcuda}" "${dir}/libcuda.so.1"
    ln -sfn "libcuda.so.1" "${dir}/libcuda.so"
    ln -sfn "${USERSPACE_DIR}/lib/libnvidia-ml.so.${NVIDIA_550_VERSION}" "${dir}/libnvidia-ml.so.1"
    ln -sfn "libnvidia-ml.so.1" "${dir}/libnvidia-ml.so"
    ln -sfn "${USERSPACE_DIR}/lib/libnvidia-ptxjitcompiler.so.${NVIDIA_550_VERSION}" "${dir}/libnvidia-ptxjitcompiler.so.1"
    ln -sfn "libnvidia-ptxjitcompiler.so.1" "${dir}/libnvidia-ptxjitcompiler.so"
  done < <(unique_lib_dirs)
  if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
    ln -sfn "${CUDA_PREFIX}/bin/nvcc" /usr/local/bin/nvcc
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    ln -sfn /usr/local/cuda/bin/nvcc /usr/local/bin/nvcc
  fi
  if [[ -x "${USERSPACE_DIR}/bin/nvidia-smi" ]]; then
    ln -sfn "${USERSPACE_DIR}/bin/nvidia-smi" /usr/local/bin/nvidia-smi
  fi
  ldconfig
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
