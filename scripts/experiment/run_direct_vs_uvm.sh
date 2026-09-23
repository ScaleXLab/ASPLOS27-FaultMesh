#!/usr/bin/env bash
# Direct FrontLib + FaultMesh BackLib versus the UVM baseline.
# Problem sizes are the sources' historical ~4 GiB defaults.
#
#   bash scripts/experiment/run_direct_vs_uvm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="${ROOT}/microbenchmark"
export PATH="/usr/local/cuda/bin:${PATH}"
ARCH="${ARCH:-sm_80}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/results/direct_vs_uvm_${STAMP}"
SKIP="-DSKIP_CPU_VERIFY"

log() { printf '[experiment] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x /usr/local/cuda/bin/nvcc ]] || die "nvcc not found. Run: sudo bash scripts/env/switch_to_faultmesh.sh"
ver="$(cat /proc/driver/nvidia/version 2>/dev/null || true)"
printf '%s\n' "${ver}" | grep -q '550\.54\.14' || die "kernel driver is not 550.54.14. Run: sudo bash scripts/env/switch_to_faultmesh.sh"

mkdir -p "${OUT}/logs" "${OUT}/bin"
echo "benchmark,design,wall_s,gpu_s,rc" > "${OUT}/results.csv"

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
      if [[ -f "${BENCH}/bfs/graph6M.txt" ]]; then
        ln -sfn "${BENCH}/bfs/graph6M.txt" "${OUT}/bin/graph6M.txt"
      fi
      ;;
  esac
}

for app in 2DCONV ATAX BICG GEMM GESUMMV hellinger MVT nw XSBench bfs; do
  compile_one "${app}"
done

load_design() {
  local design="$1"
  log "load ${design}"
  if [[ "${design}" == "baseline" ]]; then
    sudo BUILD_KERNEL=0 bash "${ROOT}/scripts/env/load_baseline.sh"
  else
    sudo BUILD_KERNEL=0 bash "${ROOT}/scripts/env/load_faultmesh.sh"
  fi
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
  if [[ "${app}" == "bfs" && ! -f "${OUT}/bin/graph6M.txt" ]]; then
    log "skip bfs: microbenchmark/bfs/graph6M.txt is not in the repository"
    echo "bfs,${design},,,skipped_no_graph" >> "${OUT}/results.csv"
    return 0
  fi
  local cmd logfile rc=0 t0 wall gpu
  cmd="$(bench_cmd "${app}")"
  logfile="${OUT}/logs/${app}_${design}.log"
  log "run ${app} ${design}"
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
  echo "${app},${design},${wall},${gpu},${rc}" >> "${OUT}/results.csv"
  log "  rc=${rc} wall=${wall}s gpu=${gpu:-NA}s"
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
    "# Direct FrontLib + FaultMesh vs UVM baseline",
    "",
    "GPU seconds. Speedup is UVM baseline / FaultMesh.",
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
