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

# Put nvcc, nvidia-smi, and libcuda where the current shell already looks.
# /etc/profile.d is read only by a new login shell, so it cannot update this one.
install_userspace() {
  rm -f /etc/profile.d/faultmesh-550.sh
  local nvcc_bin="" smi_bin="" dir link
  if [[ -x "${CUDA_PREFIX}/bin/nvcc" ]]; then
    nvcc_bin="${CUDA_PREFIX}/bin/nvcc"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin="/usr/local/cuda/bin/nvcc"
  fi
  if [[ -x "${USERSPACE_DIR}/bin/nvidia-smi" ]]; then
    smi_bin="${USERSPACE_DIR}/bin/nvidia-smi"
  fi

  : > "${SNAPSHOT_DIR}/bin-links.txt"
  if [[ -n "${nvcc_bin}" ]]; then
    if [[ -e /usr/local/bin/nvcc || -L /usr/local/bin/nvcc ]]; then
      printf 'nvcc %s\n' "$(readlink -f /usr/local/bin/nvcc)" >> "${SNAPSHOT_DIR}/bin-links.txt"
    else
      printf 'nvcc MISSING\n' >> "${SNAPSHOT_DIR}/bin-links.txt"
    fi
    ln -sfn "${nvcc_bin}" /usr/local/bin/nvcc
  fi
  if [[ -n "${smi_bin}" ]]; then
    if [[ -e /usr/local/bin/nvidia-smi || -L /usr/local/bin/nvidia-smi ]]; then
      printf 'nvidia-smi %s\n' "$(readlink -f /usr/local/bin/nvidia-smi)" >> "${SNAPSHOT_DIR}/bin-links.txt"
    else
      printf 'nvidia-smi MISSING\n' >> "${SNAPSHOT_DIR}/bin-links.txt"
    fi
    ln -sfn "${smi_bin}" /usr/local/bin/nvidia-smi
  fi

  if [[ -e "${USERSPACE_DIR}/lib/libcuda.so.${NVIDIA_550_VERSION}" ]]; then
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
  fi

  local ld_conf="/etc/ld.so.conf.d/faultmesh-550.conf"
  : > "${ld_conf}"
  [[ -d "${USERSPACE_DIR}/lib" ]] && printf '%s\n' "${USERSPACE_DIR}/lib" >> "${ld_conf}"
  [[ -d "${CUDA_PREFIX}/lib64" ]] && printf '%s\n' "${CUDA_PREFIX}/lib64" >> "${ld_conf}"
  ldconfig
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
BUILD_KERNEL="${BUILD_KERNEL:-1}" bash "${REPO_ROOT}/scripts/env/load_faultmesh.sh"
restore_kos_after_install
install_userspace
touch "${SNAPSHOT_DIR}/ACTIVE"
use_repo_cuda
log "FaultMesh 550.54.14 is loaded. nvidia-smi:"
nvidia-smi || true
log "Restore the previous stack with: sudo bash scripts/env/restore_host_nvidia.sh"
