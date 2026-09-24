#!/usr/bin/env bash
# Put back the driver, libcuda, and /usr/local/cuda recorded by
# scripts/switch_to_faultmesh.sh.
#
#   sudo bash scripts/restore_host_nvidia.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/nvidia_550_common.sh"

[[ "$(id -u)" -eq 0 ]] || die "run as root: sudo bash $0"
[[ -f "${SNAPSHOT_DIR}/manifest.env" ]] || die "no snapshot in ${SNAPSHOT_DIR}; nothing to restore"
# shellcheck disable=SC1091
source "${SNAPSHOT_DIR}/manifest.env"

kernel="$(uname -r)"
[[ "${kernel}" == "${KERNEL_UNAME}" ]] || die "snapshot is for kernel ${KERNEL_UNAME}, this machine is ${kernel}"
module_root="/lib/modules/${kernel}"

log "unload NVIDIA modules"
for mod in nvidia_peermem nvidia_drm nvidia_modeset nvidia_uvm libnvm nvidia; do
  if lsmod | awk -v name="${mod}" '$1 == name { found = 1 } END { exit !found }'; then
    rmmod "${mod}" || modprobe -r "${mod}"
  fi
done
if lsmod | awk '$1 == "nvidia" { found = 1 } END { exit !found }'; then
  die "nvidia is still loaded; stop GPU processes and retry"
fi

if [[ -f "${SNAPSHOT_DIR}/ko-added.txt" ]]; then
  while read -r ko; do
    [[ -n "${ko}" && -f "${ko}" ]] || continue
    log "remove module installed by the switch: ${ko}"
    rm -f "${ko}"
  done < "${SNAPSHOT_DIR}/ko-added.txt"
fi

if [[ -d "${SNAPSHOT_DIR}/kos" ]]; then
  while read -r saved; do
    rel="${saved#${SNAPSHOT_DIR}/kos/}"
    dest="${module_root}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    log "restore ${dest}"
    cp -a "${saved}" "${dest}"
  done < <(find "${SNAPSHOT_DIR}/kos" -type f -name '*.ko' | sort)
fi
depmod -a "${kernel}" || true

log "load the previous kernel modules"
modprobe nvidia
modprobe nvidia-uvm || true
modprobe nvidia-modeset || true
modprobe nvidia-drm || true
# nvidia_peermem takes no parameters. Kernels without the InfiniBand
# peer-memory interface return -EINVAL from its init. Do not load it.

if [[ -f "${SNAPSHOT_DIR}/liblinks.txt" ]]; then
  while read -r path target; do
    [[ -n "${path}" ]] || continue
    if [[ "${target}" == "FILE" ]]; then
      base="$(basename "$(dirname "${path}")")-$(basename "${path}")"
      [[ -f "${SNAPSHOT_DIR}/${base}" ]] || continue
      log "restore file ${path}"
      cp -a "${SNAPSHOT_DIR}/${base}" "${path}"
    else
      log "restore symlink ${path} -> ${target}"
      ln -sfn "${target}" "${path}"
    fi
  done < "${SNAPSHOT_DIR}/liblinks.txt"
fi
ldconfig || true

if [[ -f "${SNAPSHOT_DIR}/liblinks-created.txt" ]]; then
  while read -r path; do
    [[ -n "${path}" ]] || continue
    log "remove link created by the switch: ${path}"
    rm -f "${path}"
  done < "${SNAPSHOT_DIR}/liblinks-created.txt"
fi

rm -f /etc/profile.d/faultmesh-550.sh /etc/ld.so.conf.d/faultmesh-550.conf
if [[ -f "${SNAPSHOT_DIR}/bin-links.txt" ]]; then
  while read -r name target; do
    [[ -n "${name}" ]] || continue
    if [[ "${target}" == "MISSING" ]]; then
      rm -f "/usr/local/bin/${name}"
    elif [[ -n "${target}" ]]; then
      ln -sfn "${target}" "/usr/local/bin/${name}"
    fi
  done < "${SNAPSHOT_DIR}/bin-links.txt"
else
  rm -f /usr/local/bin/nvidia-smi
fi
ldconfig || true
if [[ "${CUDA_LINK_TARGET}" == "MISSING" ]]; then
  rm -f /usr/local/cuda
elif [[ "${CUDA_LINK_TARGET}" != "NOT_A_LINK" ]]; then
  ln -sfn "${CUDA_LINK_TARGET}" /usr/local/cuda
fi

rm -f "${SNAPSHOT_DIR}/ACTIVE"
log "previous NVIDIA stack restored"
nvidia-smi || true
cat /proc/driver/nvidia/version 2>/dev/null || true
