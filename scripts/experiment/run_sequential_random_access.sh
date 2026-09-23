#!/usr/bin/env bash
# Sequential and random access at 512, 1024, 2048, and 4096 threads.
#
#   bash scripts/experiment/run_sequential_random_access.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/env/nvidia_550_common.sh"
require_nvcc
SRC="${ROOT}/microbenchmark/sequential_random_access.cu"
ARCH="${ARCH:-sm_80}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/results/sequential_random_access_${STAMP}"

log() { printf '[sequential-random] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

mkdir -p "${OUT}"

log "compile"
nvcc -arch="${ARCH}" -O3 -DTEST_NUM_PAGES=32768 -DDUP_FACTOR=4 \
  "${SRC}" -o "${OUT}/sequential_random_access"

log "run"
set +e
(
  cd "${OUT}"
  PF_SWEEP=1 PF_SCHED_SMOKE_ONLY=1 CUDA_MODULE_LOADING=EAGER ./sequential_random_access
) >"${OUT}/run.log"
rc=$?
set -e
if [[ "${rc}" -ne 0 ]]; then
  log "benchmark failed rc=${rc}"
  tail -n 40 "${OUT}/run.log" >&2
  exit "${rc}"
fi

python3 - "${OUT}/run.log" "${OUT}/summary.md" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="ignore").splitlines()
section = None
rows = {"sequential": {}, "random": {}}
for line in text:
    if line.startswith("========== SWEEP: seq_dup"):
        section = "sequential"
        continue
    if line.startswith("========== SWEEP: rand_dup"):
        section = "random"
        continue
    if section is None or not line[:1].isdigit():
        continue
    parts = line.split(",")
    threads = int(parts[0])
    if threads not in (512, 1024, 2048, 4096):
        continue
    plain_ms = float(parts[2] if section == "sequential" else parts[3])
    time_ms = float(parts[6] if section == "sequential" else parts[7])
    rows[section][threads] = (plain_ms, time_ms, plain_ms / time_ms)

lines = [
    "# Sequential and random access",
    "",
    "Times are milliseconds. Speedup is plain access / this run.",
    "",
    "| Pattern | Threads | Plain (ms) | Time (ms) | Speedup |",
    "|---|---:|---:|---:|---:|",
]
for pattern in ("sequential", "random"):
    for threads in (512, 1024, 2048, 4096):
        plain_ms, time_ms, speedup = rows[pattern][threads]
        lines.append(
            f"| {pattern} | {threads} | {plain_ms:.2f} | {time_ms:.2f} | {speedup:.2f}x |"
        )
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
log "results: ${OUT}"
