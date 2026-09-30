#!/usr/bin/env bash
# nvidia_uvm cannot be unloaded while another program holds /dev/nvidia*.
# Stop those processes so the driver switch can continue.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo bash "$0" "$@"
fi

systemctl stop gdm gdm3 nvidia-persistenced 2>/dev/null || true

mapfile -t pids < <(fuser /dev/nvidia* /dev/nvidia-uvm /dev/nvidiactl 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+$/ { print }' | sort -u)
if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No other process is using the GPU."
  exit 0
fi

echo "Stopping processes that hold the GPU:"
ps -o pid,user,cmd -p "$(IFS=,; echo "${pids[*]}")" || true
kill -TERM "${pids[@]}" 2>/dev/null || true
sleep 2
mapfile -t left < <(fuser /dev/nvidia* /dev/nvidia-uvm /dev/nvidiactl 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+$/ { print }' | sort -u)
if [[ "${#left[@]}" -gt 0 ]]; then
  echo "Still using the GPU, sending SIGKILL:"
  ps -o pid,user,cmd -p "$(IFS=,; echo "${left[*]}")" || true
  kill -KILL "${left[@]}" 2>/dev/null || true
fi
