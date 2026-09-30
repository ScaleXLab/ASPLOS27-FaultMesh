#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/env/nvidia_550_common.sh"
require_nvcc

BASE="${1:-/mnt/nvme0n1/sift100m/base.100M.fbin}"
RUNS="${2:-1}"
QUERY="$(dirname "${BASE}")/query.fbin"
if [[ $# -ge 1 ]]; then
  GRAPH="${BASE%.fbin}.graph.ibin"
else
  GRAPH="/mnt/nvme0n1/sift100m/cagra_100M.graph.ibin"
fi
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
SRC="${ROOT}/application/cagra"
OUT="${ROOT}/results/cagra"
BIN="${OUT}/bin/cagra_micro"

log() { printf '[cagra] %s\n' "$*"; }

[[ "$(id -u)" -eq 0 ]] || die "run with sudo"
command -v numactl > /dev/null || die "numactl is missing (apt install numactl)"
for f in "${BASE}" "${QUERY}" "${GRAPH}"; do
  [[ -f "${f}" ]] || die "${f} is missing"
done

rm -rf "${OUT}"
mkdir -p "${OUT}/logs" "${OUT}/bin"
CSV="${OUT}/results.csv"
echo "design,run,build_s,search_s,build_checksum,search_checksum" > "${CSV}"

log "compile ${SRC}/cagra_micro.cu"
nvcc -O3 -std=c++17 -arch=sm_80 -I"${ROOT}/frontLib" "${SRC}/cagra_micro.cu" -o "${BIN}"

need_kernel_build() {
  [[ "${BUILD_KERNEL:-auto}" == "1" ]] && return 0
  [[ "${BUILD_KERNEL:-auto}" == "0" ]] && return 1
  local ko vermagic version
  ko="$(modinfo -n nvidia-uvm 2>/dev/null || true)"
  [[ -n "${ko}" && -f "${ko}" ]] || return 0
  vermagic="$(modinfo -F vermagic "${ko}" 2>/dev/null || true)"
  version="$(modinfo -F version "${ko}" 2>/dev/null || true)"
  [[ "${vermagic}" == "$(uname -r)"* && "${version}" == "550.54.14" ]] || return 0
  find "${ROOT}/backLib/kernel-open/nvidia-uvm" -name '*.c' -newer "${ko}" -print -quit | grep -q .
}

load_design() {
  local design="$1" build=0
  if [[ "${design}" == "baseline" ]]; then
    need_kernel_build && build=1
    bind_550_userspace
    install_gsp_firmware
    BUILD_KERNEL="${build}" bash "${ROOT}/backLib/perf_baseline.sh" > "${OUT}/logs/load_${design}.log" 2>&1
  else
    BUILD_KERNEL=0 bash "${ROOT}/backLib/perf_ours.sh" > "${OUT}/logs/load_${design}.log" 2>&1
  fi
  grep -q '550\.54\.14' /proc/driver/nvidia/version || die "loaded kernel is not 550.54.14 after ${design}"
  log "driver: ${design}"
}

run_one() {
  local design="$1" run="$2" rc=0 xid0 xid
  local logf="${OUT}/logs/${design}_r${run}.log"
  local frontlib=0
  [[ "${design}" == "faultmesh" ]] && frontlib=1
  sync; echo 3 > /proc/sys/vm/drop_caches
  xid0="$(dmesg | grep -c Xid || true)"
  log "run ${design} ${run}/${RUNS}"
  timeout --kill-after=30 "${RUN_TIMEOUT}" numactl --interleave=all "${BIN}" \
    --base "${BASE}" --query "${QUERY}" --graph "${GRAPH}" \
    --frontlib "${frontlib}" > "${logf}" 2>&1 || rc=$?
  xid=$(( $(dmesg | grep -c Xid || true) - xid0 ))
  local b s bc sc
  b="$(awk '/^Build phase:/ {sub("s","",$3); print $3}' "${logf}")"
  s="$(awk '/^Search phase:/ {sub("s","",$3); print $3}' "${logf}")"
  bc="$(grep -o 'Build phase.*checksum=[0-9]*' "${logf}" | sed 's/.*checksum=//' || true)"
  sc="$(grep -o 'Search phase.*checksum=[0-9]*' "${logf}" | sed 's/.*checksum=//' || true)"
  echo "${design},${run},${b},${s},${bc},${sc}" >> "${CSV}"
  log "  build ${b:-FAILED}s, search ${s:-FAILED}s"
  [[ "${rc}" -eq 0 ]] || log "  WARNING: cagra_micro exited with ${rc}, see ${logf}"
  [[ "${xid}" -eq 0 ]] || log "  WARNING: ${xid} Xid errors in dmesg"
}

for design in baseline faultmesh; do
  load_design "${design}"
  for run in $(seq 1 "${RUNS}"); do
    run_one "${design}" "${run}"
  done
done

python3 - "${CSV}" <<'PY'
import csv, sys
csv_path = sys.argv[1]
rows = list(csv.DictReader(open(csv_path)))
def mean(design, col):
    v = [float(r[col]) for r in rows if r["design"] == design and r[col]]
    return sum(v) / len(v) if v else None
checks = {(r["build_checksum"], r["search_checksum"]) for r in rows if r["build_checksum"]}
lines = [
    "# CAGRA (SIFT): UVM baseline vs FaultMesh",
    "",
    "Kernel seconds. Speedup is baseline / FaultMesh.",
    "",
    "| Phase | UVM baseline | FaultMesh | Speedup |",
    "|---|---:|---:|---:|",
]
for name, col in (("build", "build_s"), ("search", "search_s")):
    b, f = mean("baseline", col), mean("faultmesh", col)
    if b and f:
        lines.append(f"| {name} | {b:.3f} | {f:.3f} | {b / f:.2f}x |")
    else:
        lines.append(f"| {name} | {'-' if b is None else f'{b:.3f}'} | {'-' if f is None else f'{f:.3f}'} | - |")
lines += ["", "Results identical across designs: " + ("yes" if len(checks) == 1 else "NO")]
print("\n".join(lines))
PY
