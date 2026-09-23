#!/usr/bin/env bash
# Save the host NVIDIA driver and CUDA libraries, then switch to
# FaultMesh's 550.54.14 kernel modules plus the matching userspace.
#
#   sudo bash scripts/switch_to_faultmesh.sh
#   sudo BUILD_KERNEL=0 bash scripts/switch_to_faultmesh.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/nvidia_550_common.sh"

[[ "$(id -u)" -eq 0 ]] || die "run as root: sudo bash $0"
[[ -f "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" ]] || bash "${REPO_ROOT}/scripts/download_nvidia_550.sh"
[[ -x "${CUDA_PREFIX}/bin/nvcc" ]] || install_cuda_toolkit

kernel="$(uname -r)"
module_root="/lib/modules/${kernel}"

snapshot_host() {
  [[ -f "${SNAPSHOT_DIR}/manifest.env" ]] && return 0
  log "record the current driver and CUDA libraries in ${SNAPSHOT_DIR}"
  mkdir -p "${SNAPSHOT_DIR}/kos"
  {
    echo "KERNEL_UNAME=${kernel}"
    echo "NVIDIA_VERSION=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 || true)"
    if [[ -L /usr/local/cuda ]]; then
      echo "CUDA_LINK_TARGET=$(readlink /usr/local/cuda)"
    elif [[ -e /usr/local/cuda ]]; then
      echo "CUDA_LINK_TARGET=NOT_A_LINK"
    else
      echo "CUDA_LINK_TARGET=MISSING"
    fi
    echo "NVIDIA_SMI=$(command -v nvidia-smi || true)"
  } > "${SNAPSHOT_DIR}/manifest.env"

  : > "${SNAPSHOT_DIR}/liblinks.txt"
  local dir link
  while read -r dir; do
    for link in "${LIB_LINKS[@]}"; do
      if [[ -L "${dir}/${link}" ]]; then
        printf '%s %s\n' "${dir}/${link}" "$(readlink "${dir}/${link}")" >> "${SNAPSHOT_DIR}/liblinks.txt"
      elif [[ -e "${dir}/${link}" ]]; then
        printf '%s FILE\n' "${dir}/${link}" >> "${SNAPSHOT_DIR}/liblinks.txt"
        cp -a "${dir}/${link}" "${SNAPSHOT_DIR}/$(basename "${dir}")-${link}"
      fi
    done
  done < <(unique_lib_dirs)

  : > "${SNAPSHOT_DIR}/ko-before.txt"
  if [[ -d "${module_root}" ]]; then
    while read -r ko; do
      local rel="${ko#${module_root}/}"
      mkdir -p "${SNAPSHOT_DIR}/kos/$(dirname "${rel}")"
      cp -a "${ko}" "${SNAPSHOT_DIR}/kos/${rel}"
      printf '%s\n' "${ko}" >> "${SNAPSHOT_DIR}/ko-before.txt"
    done < <(find "${module_root}" \( -name 'nvidia.ko' -o -name 'nvidia-*.ko' \) | sort -u)
  fi
}

install_userspace() {
  local dir link
  : > "${SNAPSHOT_DIR}/liblinks-created.txt"
  while read -r dir; do
    for link in "${LIB_LINKS[@]}"; do
      if [[ ! -e "${dir}/${link}" && ! -L "${dir}/${link}" ]]; then
        printf '%s\n' "${dir}/${link}" >> "${SNAPSHOT_DIR}/liblinks-created.txt"
      fi
    done
    ln -sfn "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" "${dir}/libcuda.so.1"
    ln -sfn "libcuda.so.1" "${dir}/libcuda.so"
    ln -sfn "${USERSPACE_DIR}/lib/libnvidia-ml.so.${NVIDIA_550_VERSION}" "${dir}/libnvidia-ml.so.1"
    ln -sfn "libnvidia-ml.so.1" "${dir}/libnvidia-ml.so"
    ln -sfn "${USERSPACE_DIR}/lib/libnvidia-ptxjitcompiler.so.${NVIDIA_550_VERSION}" "${dir}/libnvidia-ptxjitcompiler.so.1"
    ln -sfn "libnvidia-ptxjitcompiler.so.1" "${dir}/libnvidia-ptxjitcompiler.so"
  done < <(unique_lib_dirs)
  ldconfig || true
  install -m 0755 "${USERSPACE_DIR}/bin/nvidia-smi" /usr/local/bin/nvidia-smi
  if [[ ! -e /usr/local/cuda && ! -L /usr/local/cuda ]]; then
    echo "CUDA_LINK_CREATED=1" >> "${SNAPSHOT_DIR}/manifest.env"
  fi
  ln -sfn "${CUDA_PREFIX}" /usr/local/cuda
  cat > /etc/profile.d/faultmesh-550.sh <<EOF
export PATH=/usr/local/cuda/bin:\${PATH}
export LD_LIBRARY_PATH=${USERSPACE_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOF
}

restore_kos_after_install() {
  # modules_install may drop new nvidia*.ko files beside the distro ones.
  # Remember every nvidia*.ko path that was not in the original snapshot.
  : > "${SNAPSHOT_DIR}/ko-added.txt"
  [[ -d "${module_root}" ]] || return 0
  while read -r ko; do
    if ! grep -qx "${ko}" "${SNAPSHOT_DIR}/ko-before.txt"; then
      printf '%s\n' "${ko}" >> "${SNAPSHOT_DIR}/ko-added.txt"
    fi
  done < <(find "${module_root}" \( -name 'nvidia.ko' -o -name 'nvidia-*.ko' \) | sort -u)
}

snapshot_host
install_gsp_firmware
log "build and load FaultMesh kernel modules"
BUILD_KERNEL="${BUILD_KERNEL:-1}" bash "${REPO_ROOT}/scripts/load_faultmesh.sh"
restore_kos_after_install
install_userspace
touch "${SNAPSHOT_DIR}/ACTIVE"
log "FaultMesh 550.54.14 is loaded. nvidia-smi:"
nvidia-smi || true
log "Restore the previous stack with: sudo bash scripts/restore_host_nvidia.sh"
