#!/bin/bash
sudo systemctl stop gdm 2>/dev/null || true
set -euo pipefail

# 参数化，便于批量脚本复用
BUILD_KERNEL="${BUILD_KERNEL:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-0}"
ASYNC_COPY_MAP_ENABLE="${ASYNC_COPY_MAP_ENABLE:-0}"
# Legacy compatibility knobs kept for shared scripts. The HPCA-style dual-slot
# pipeline ignores workers/depth and always runs with one per-GPU consumer.
ASYNC_COPY_MAP_WORKERS="${ASYNC_COPY_MAP_WORKERS:-1}"
ASYNC_COPY_MAP_QUEUE_DEPTH="${ASYNC_COPY_MAP_QUEUE_DEPTH:-2}"
ASYNC_COPY_MAP_PREFETCH_SAFE="${ASYNC_COPY_MAP_PREFETCH_SAFE:-1}"

set_param_if_exists() {
  local name="$1"
  local value="$2"
  local path="/sys/module/nvidia_uvm/parameters/${name}"
  if [ -e "${path}" ]; then
    echo "${value}" | sudo tee "${path}" > /dev/null
  fi
}

# Keep baseline reload behavior identical to perf_ours.sh.  libnvm holds a
# reference on the nvidia core module on this host, so it must be removed
# before the final core-module unload.
unload_nvidia_stack() {
  local mod
  for mod in nvidia_peermem nvidia_drm nvidia_modeset nvidia_uvm libnvm nvidia; do
    if lsmod | awk -v name="${mod}" '$1 == name { found = 1 } END { exit !found }'; then
      sudo modprobe -r "${mod}"
    fi
  done
  if lsmod | awk '$1 == "nvidia" { found = 1 } END { exit !found }'; then
    echo "ERROR: nvidia remains loaded after unloading the stack." >&2
    echo "Stop processes using /dev/nvidia* (or any external GPU client) and retry." >&2
    exit 1
  fi
}

# Build before unloading. A failed make must not leave the GPU without a driver.
# make -C does not depend on the caller's working directory.
BACKLIB_DIR="$(dirname "$(readlink -f "$0")")"
if [ "${BUILD_KERNEL}" = "1" ]; then
  make -C "${BACKLIB_DIR}" modules -j"$(nproc)"
fi

unload_nvidia_stack

if [ "${BUILD_KERNEL}" = "1" ]; then
  make -C "${BACKLIB_DIR}" modules_install -j"$(nproc)"
fi

# Resolve the core driver's ecc dependency via modprobe.  A bare insmod
# cannot load it and fails after a clean unload with "Unknown symbol".
sudo modprobe nvidia
sudo modprobe nvidia-uvm \
  uvm_parallel_fault_processing=0 \
  uvm_kthread_workers="${PARALLEL_WORKERS}" \
  uvm_batched_ipi_unmap=0 \
  uvm_merge_dispatch=0 \
  uvm_parallel_async_copy_map_enable="${ASYNC_COPY_MAP_ENABLE}" \
  uvm_parallel_async_copy_map_workers="${ASYNC_COPY_MAP_WORKERS}" \
  uvm_parallel_async_copy_map_queue_depth="${ASYNC_COPY_MAP_QUEUE_DEPTH}" \
  uvm_parallel_async_copy_map_prefetch_safe="${ASYNC_COPY_MAP_PREFETCH_SAFE}" \
  uvm_perf_fault_prev_fetch_predictor_enable=0 \
  uvm_perf_fault_prev_fetch_mode=2 \
  uvm_perf_fault_prev_batch_predictor_enable=0 \
  uvm_perf_fault_pred_timing_enable=0 \
  uvm_perf_fault_stale_detail_enable=0 \
  uvm_perf_fault_replay_stale_update_put_enable=0 \
  uvm_perf_fault_fetch_predictor_boost_enable=0 \
  uvm_perf_fault_pred_skip_enable=0 \
  uvm_perf_fault_replay_force_update_put=0
sudo modprobe nvidia-modeset
sudo modprobe nvidia-drm
sudo modprobe nvidia-peermem
sudo dmesg -C
set_param_if_exists uvm_fpd_profile_enable 0
set_param_if_exists uvm_merge_dispatch 0
set_param_if_exists uvm_parallel_async_copy_map_enable "${ASYNC_COPY_MAP_ENABLE}"
set_param_if_exists uvm_parallel_async_copy_map_workers "${ASYNC_COPY_MAP_WORKERS}"
set_param_if_exists uvm_parallel_async_copy_map_queue_depth "${ASYNC_COPY_MAP_QUEUE_DEPTH}"
set_param_if_exists uvm_parallel_async_copy_map_prefetch_safe "${ASYNC_COPY_MAP_PREFETCH_SAFE}"
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_prev_fetch_predictor_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_prev_fetch_predictor_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_prev_fetch_mode ]; then
  echo 2 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_prev_fetch_mode > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_prev_batch_predictor_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_prev_batch_predictor_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_pred_timing_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_pred_timing_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_stale_detail_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_stale_detail_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_stale_update_put_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_stale_update_put_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_fetch_predictor_boost_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_fetch_predictor_boost_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_pred_skip_enable ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_pred_skip_enable > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_force_update_put ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_force_update_put > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_merge_segmented_dispatch ]; then
  echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_merge_segmented_dispatch > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_merge_shared_push_max_faults ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_merge_shared_push_max_faults > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_merge_map_allow_2m_split_shared ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_merge_map_allow_2m_split_shared > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_merge_pipeline_overlap ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_merge_pipeline_overlap > /dev/null
fi
if [ -e /sys/module/nvidia_uvm/parameters/uvm_merge_continuous_workers ]; then
  echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_merge_continuous_workers > /dev/null
fi
echo "Reload finished!"
