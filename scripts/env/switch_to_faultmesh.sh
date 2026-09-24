#!/usr/bin/env bash
# Save the host NVIDIA driver and CUDA libraries, then switch to
# FaultMesh's 550.54.14 kernel modules plus the matching userspace.
#
#   sudo bash scripts/switch_to_faultmesh.sh
#   sudo BUILD_KERNEL=0 bash scripts/switch_to_faultmesh.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/nvidia_550_common.sh"

[[ "$(id -u)" -eq 0 ]] || die "run as root: sudo bash $0"
if host_has_550_cuda_lib; then
  log "libcuda ${NVIDIA_550_VERSION} is already installed. Skip the driver and CUDA toolkit download."
else
  [[ -f "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" ]] || bash "${REPO_ROOT}/scripts/env/download_nvidia_550.sh"
  [[ -x "${CUDA_PREFIX}/bin/nvcc" ]] || install_cuda_toolkit
fi

kernel="$(uname -r)"
module_root="/lib/modules/${kernel}"

snapshot_host() {
  [[ -f "${SNAPSHOT_DIR}/manifest.env" ]] && return 0
  log "record the current driver and CUDA libraries in ${SNAPSHOT_DIR}"
  mkdir -p "${SNAPSHOT_DIR}/kos"
  cat /proc/driver/nvidia/version > "${SNAPSHOT_DIR}/nvidia-version.txt" 2>/dev/null || true
  {
    printf 'KERNEL_UNAME=%q\n' "${kernel}"
    if [[ -L /usr/local/cuda ]]; then
      printf 'CUDA_LINK_TARGET=%q\n' "$(readlink /usr/local/cuda)"
    elif [[ -e /usr/local/cuda ]]; then
      printf 'CUDA_LINK_TARGET=%q\n' "NOT_A_LINK"
    else
      printf 'CUDA_LINK_TARGET=%q\n' "MISSING"
    fi
    printf 'NVIDIA_SMI=%q\n' "$(command -v nvidia-smi || true)"
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

# Keep 550 libcuda and NVML in the conda prefix. Do not retarget system libraries.
install_userspace() {
  rm -f /etc/profile.d/faultmesh-550.sh /etc/ld.so.conf.d/faultmesh-550.conf
  local owner="${SUDO_USER:-}"
  if [[ -n "${owner}" && "${owner}" != "root" ]]; then
    sudo -u "${owner}" HOME="$(getent passwd "${owner}" | cut -d: -f6)" \
      bash -c "source '${REPO_ROOT}/scripts/env/nvidia_550_common.sh' && setup_conda_userspace"
    chown -R "${owner}:${owner}" "${CONDA_ENV_DIR}"
  else
    setup_conda_userspace
  fi
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
# Prepare userspace before unloading the current driver. A missing conda
# install must not leave the machine on a half-switched kernel.
install_userspace
log "build and load FaultMesh kernel modules"
BUILD_KERNEL="${BUILD_KERNEL:-1}" bash "${REPO_ROOT}/scripts/env/load_faultmesh.sh"
restore_kos_after_install
touch "${SNAPSHOT_DIR}/ACTIVE"
use_repo_cuda
log "FaultMesh 550.54.14 is loaded. nvidia-smi:"
nvidia-smi || true
log "Restore the previous stack with: sudo bash scripts/env/restore_host_nvidia.sh"
