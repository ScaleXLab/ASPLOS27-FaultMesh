#!/usr/bin/env bash
# Direct FrontLib + FaultMesh BackLib versus the UVM baseline.
# Problem sizes are the sources' historical ~4 GiB defaults.
#
#   bash scripts/experiment/run_baseline_vs_faultmesh.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/env/nvidia_550_common.sh"
require_nvcc
log "550 userspace: ${CONDA_ENV_DIR}"
BENCH="${ROOT}/application"
ARCH="${ARCH:-sm_80}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/results/baseline_vs_faultmesh_${STAMP}"
SKIP="-DSKIP_CPU_VERIFY"

log() { printf '[experiment] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

mkdir -p "${OUT}/logs" "${OUT}/bin"
echo "benchmark,design,wall_s,gpu_s,rc" > "${OUT}/results.csv"

ensure_bfs_graph() {
  local graph="${BENCH}/bfs/graph6M.txt"
  if [[ -f "${graph}" ]]; then
    return 0
  fi
  log "generate ${graph} (6291456 nodes)"
  make -C "${BENCH}/bfs/inputgen" graphgen
  (
    cd "${BENCH}/bfs/inputgen"
    ./graphgen 6291456 6M
  )
  mv "${BENCH}/bfs/inputgen/graph6M.txt" "${graph}"
}

compile_one() {
  local name="$1"
  log "compile ${name}"
  case "${name}" in
    2DCONV|ATAX|BICG|GEMM|GESUMMV|MVT)
      make -C "${BENCH}/${name}" CUDA_ARCH="${ARCH}" DEF="${SKIP}"
      cp -a "${BENCH}/${name}/run" "${OUT}/bin/${name}"
      ;;
    hellinger)
      nvcc -O3 -arch="${ARCH}" ${SKIP} "${BENCH}/hellinger/main.cu" -o "${OUT}/bin/hellinger"
      ;;
    nw)
      nvcc -O3 -arch="${ARCH}" ${SKIP} "${BENCH}/nw/needle.cu" -o "${OUT}/bin/nw"
      ;;
    XSBench)
      make -C "${BENCH}/XSBench" clean >/dev/null 2>&1 || true
      make -C "${BENCH}/XSBench" SM_VERSION=80 ARCH="${ARCH}"
      cp -a "${BENCH}/XSBench/XSBench" "${OUT}/bin/XSBench"
      ;;
    bfs)
      nvcc -O3 -arch="${ARCH}" "${BENCH}/bfs/main.cu" -o "${OUT}/bin/bfs"
      ensure_bfs_graph
      ln -sfn "${BENCH}/bfs/graph6M.txt" "${OUT}/bin/graph6M.txt"
      ;;
  esac
}

for app in 2DCONV ATAX BICG GEMM GESUMMV hellinger MVT nw XSBench bfs; do
  compile_one "${app}"
done

# Baseline and FaultMesh are the same backLib binary with different module
# parameters. Rebuild only when that binary is missing, for another kernel,
# not 550.54.14, or older than the UVM sources. BUILD_KERNEL=1 forces a rebuild.
need_kernel_build() {
  [[ "${BUILD_KERNEL:-auto}" == "1" ]] && return 0
  [[ "${BUILD_KERNEL:-auto}" == "0" ]] && return 1
  local ko vermagic version
  ko="$(modinfo -n nvidia-uvm 2>/dev/null || true)"
  [[ -n "${ko}" && -f "${ko}" ]] || return 0
  vermagic="$(modinfo -F vermagic "${ko}" 2>/dev/null || true)"
  version="$(modinfo -F version "${ko}" 2>/dev/null || true)"
  [[ "${vermagic}" == "$(uname -r)"* && "${version}" == "550.54.14" ]] || return 0
  if find "${ROOT}/backLib/kernel-open/nvidia-uvm" -name '*.c' -newer "${ko}" -print -quit | grep -q .; then
    return 0
  fi
  return 1
}

load_design() {
  local design="$1" build=0
  # Baseline and FaultMesh are two loads of backLib. Baseline must be loaded
  # first through perf_baseline.sh, which unloads whatever driver is present.
  if [[ "${design}" == "baseline" ]] && need_kernel_build; then
    build=1
    log "backLib module is missing or older than the source; compiling once"
  elif [[ "${design}" != "faultmesh" ]]; then
    log "reload ${design} parameters without compiling"
  fi
  log "switch driver to ${design}"
  if [[ "${design}" == "baseline" ]]; then
    sudo bash -c "source '${ROOT}/scripts/env/nvidia_550_common.sh' && bind_550_userspace && install_gsp_firmware"
    sudo BUILD_KERNEL="${build}" bash "${ROOT}/backLib/perf_baseline.sh"
  else
    sudo BUILD_KERNEL=0 bash "${ROOT}/backLib/perf_ours.sh"
  fi
  local ver
  ver="$(cat /proc/driver/nvidia/version 2>/dev/null || true)"
  printf '%s\n' "${ver}" | grep -q '550\.54\.14' || die "loaded kernel is not 550.54.14 after ${design}"
}

bench_cmd() {
  local app="$1"
  case "${app}" in
    2DCONV|ATAX|BICG|GEMM|GESUMMV|MVT) echo "${OUT}/bin/${app}" ;;
    hellinger) echo "${OUT}/bin/hellinger" ;;
    nw) echo "${OUT}/bin/nw 23120 23120" ;;
    XSBench) echo "${OUT}/bin/XSBench" ;;
    bfs) echo "${OUT}/bin/bfs" ;;
  esac
}

parse_gpu() {
  local logf="$1" gpu
  gpu="$(grep -oP 'GPU Runtime: ?\K[0-9.]+' "${logf}" 2>/dev/null | head -1 || true)"
  if [[ -z "${gpu}" ]]; then
    gpu="$(grep -oP ',(plain|baseline|prefault|frontend),\K[0-9.]+' "${logf}" 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "${gpu}"
}

run_app() {
  local design="$1" app="$2" variant="$3"
  local cmd logfile rc=0 t0 wall gpu i
  local rc_all=0 sum_wall=0 n_gpu=0 gpu_list=""
  cmd="$(bench_cmd "${app}")"
  log "run ${app} ${design}"
  for i in 1 2 3; do
    logfile="${OUT}/logs/${app}_${design}_${i}.log"
    rc=0
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
    t0=${SECONDS}
    if [[ "${app}" == "bfs" ]]; then
      (cd "${OUT}/bin" && timeout --kill-after=30 1800 env BENCH_VARIANT="${variant}" SKIP_NAIVE_GEMM=1 ${cmd}) \
        > "${logfile}" 2>&1 || rc=$?
    else
      timeout --kill-after=30 1800 env BENCH_VARIANT="${variant}" SKIP_NAIVE_GEMM=1 \
        ${cmd} > "${logfile}" 2>&1 || rc=$?
    fi
    wall=$((SECONDS - t0))
    gpu="$(parse_gpu "${logfile}")"
    sum_wall=$((sum_wall + wall))
    [[ "${rc}" -eq 0 ]] || rc_all="${rc}"
    if [[ -n "${gpu}" ]]; then
      gpu_list="${gpu_list} ${gpu}"
      n_gpu=$((n_gpu + 1))
    fi
  done
  wall=$(((sum_wall + 1) / 3))
  gpu=""
  if [[ "${n_gpu}" -gt 0 ]]; then
    gpu="$(python3 -c "import sys; v=list(map(float, sys.argv[1:])); print(f'{sum(v)/len(v):.6f}')" ${gpu_list})"
  fi
  echo "${app},${design},${wall},${gpu},${rc_all}" >> "${OUT}/results.csv"
  log "  rc=${rc_all} wall=${wall}s gpu=${gpu:-NA}s"
}

run_design() {
  local design="$1" variant="$2"
  load_design "${design}"
  local app
  for app in 2DCONV ATAX BICG GEMM GESUMMV hellinger MVT nw XSBench bfs; do
    run_app "${design}" "${app}" "${variant}"
  done
}

run_design baseline baseline
run_design faultmesh frontend

python3 - "${OUT}/results.csv" "${OUT}/summary.md" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1])))
base, ours = {}, {}
for row in rows:
    app, design = row["benchmark"], row["design"]
    try:
        value = float(row["gpu_s"]) if row["gpu_s"] else None
    except ValueError:
        value = None
    (base if design == "baseline" else ours)[app] = value
lines = [
    "# UVM vs FaultMesh",
    "",
    "GPU seconds, mean of 3 runs. Speedup is UVM baseline / FaultMesh.",
    "",
    "| App | UVM baseline | FaultMesh | Speedup |",
    "|---|---:|---:|---:|",
]
print(lines[0])
print(f"{'App':12} {'UVM':>10} {'FaultMesh':>10} {'Speedup':>8}")
for app in ["2DCONV","ATAX","BICG","GEMM","GESUMMV","hellinger","MVT","nw","XSBench","bfs"]:
    b, f = base.get(app), ours.get(app)
    if b and f:
        speed = f"{b/f:.2f}x"
        bs, fs = f"{b:.3f}", f"{f:.3f}"
    else:
        speed, bs, fs = "-", "-" if b is None else f"{b:.3f}", "-" if f is None else f"{f:.3f}"
    lines.append(f"| {app} | {bs} | {fs} | {speed} |")
    print(f"{app:12} {bs:>10} {fs:>10} {speed:>8}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
print(f"\nWrote {sys.argv[2]}")
PY
log "results: ${OUT}"
