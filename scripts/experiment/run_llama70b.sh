#!/usr/bin/env bash
# Llama-3.1-70B inference (Q4_K_M, 42.5 GB of weights) on a 40 GB A100:
# UVM baseline versus FaultMesh.
#
#   sudo bash scripts/experiment/run_llama70b.sh [runs]
#
# GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 makes llama.cpp allocate the weights with
# cudaMallocManaged. They do not fit in GPU memory, so UVM migrates pages on
# demand and evicts older layers during every token. The reported time is
# end-to-end inference (LLAMA_PP prompt tokens, then LLAMA_TG generated
# tokens); model loading is excluded.
#
# The script builds application/llama/ when llama-bench is missing and downloads the model
# when it is missing. LLAMA_MODEL selects an existing GGUF file and HF_ENDPOINT
# a Hugging Face mirror. Results go to results/llama70b/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/env/nvidia_550_common.sh"
require_nvcc

RUNS="${1:-1}"
MODEL_NAME="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
MODEL_SIZE=42520398400
MODEL="${LLAMA_MODEL:-${THIRD_PARTY}/models/${MODEL_NAME}}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_URL="${HF_ENDPOINT}/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/${MODEL_NAME}"
PP="${LLAMA_PP:-512}"
TG="${LLAMA_TG:-16}"
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"
LLAMA_SRC="${ROOT}/application/llama"
BENCH="${LLAMA_SRC}/build/bin/llama-bench"
OUT="${ROOT}/results/llama70b"

log() { printf '[llama] %s\n' "$*"; }

[[ "$(id -u)" -eq 0 ]] || die "run with sudo"

build_llama() {
  if [[ -x "${BENCH}" ]] && ! find "${LLAMA_SRC}/ggml/src/ggml-cuda" -newer "${BENCH}" -print -quit | grep -q .; then
    return 0
  fi
  log "build llama-bench"
  cmake -S "${LLAMA_SRC}" -B "${LLAMA_SRC}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF > "${OUT}/logs/build_llama.log" 2>&1
  cmake --build "${LLAMA_SRC}/build" --target llama-bench -j"$(nproc)" >> "${OUT}/logs/build_llama.log" 2>&1 \
    || die "llama.cpp build failed, see ${OUT}/logs/build_llama.log"
}

fetch_model() {
  if [[ -f "${MODEL}" && ! -f "${MODEL}.done" && "$(stat -L -c %s "${MODEL}")" == "${MODEL_SIZE}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${MODEL}")"
  log "download ${MODEL_URL} (42.5 GB)"
  python3 "${ROOT}/scripts/env/parallel_download.py" "${MODEL_URL}" "${MODEL}" "${MODEL_SIZE}" 8
  rm -f "${MODEL}.done"
}

# Same rule as run_baseline_vs_faultmesh.sh: rebuild backLib only when the
# installed module is missing, stale, or not 550.54.14.
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
  local tag="${design}_r${run}"
  local env_extra=()
  [[ "${design}" == "faultmesh" ]] \
    && env_extra=(REALAPP_PREFETCH=1 REALAPP_PF_DECODE=1 REALAPP_PF_COLD_PASSES=1000000)
  sync; echo 3 > /proc/sys/vm/drop_caches
  xid0="$(dmesg | grep -c Xid || true)"
  log "run ${design} ${run}/${RUNS}"
  timeout --kill-after=30 "${RUN_TIMEOUT}" env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "${env_extra[@]}" \
    "${BENCH}" -m "${MODEL}" -p "${PP}" -n "${TG}" -ngl 99 -r 1 --no-warmup -o json \
    > "${OUT}/logs/${tag}.json" 2> "${OUT}/logs/${tag}.log" || rc=$?
  xid=$(( $(dmesg | grep -c Xid || true) - xid0 ))
  python3 - "${OUT}/logs/${tag}.json" "${design}" "${run}" >> "${CSV}" <<'PY'
import json, sys
path, design, run = sys.argv[1:]
total = 0.0
try:
    tests = json.load(open(path))
    total = sum(t["avg_ns"] for t in tests) / 1e9 if len(tests) == 2 else 0.0
except (OSError, ValueError, KeyError):
    pass
print(f"{design},{run},{total:.3f}" if total else f"{design},{run},")
PY
  local total
  total="$(tail -1 "${CSV}" | cut -d, -f3)"
  log "  end-to-end ${total:-FAILED}s"
  [[ "${rc}" -eq 0 ]] || log "  WARNING: llama-bench exited with ${rc}, see ${OUT}/logs/${tag}.log"
  [[ "${xid}" -eq 0 ]] || log "  WARNING: ${xid} Xid errors in dmesg"
}

rm -rf "${OUT}"
mkdir -p "${OUT}/logs"
CSV="${OUT}/results.csv"
echo "design,run,end_to_end_s" > "${CSV}"

build_llama
fetch_model
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
def mean(design):
    v = [float(r["end_to_end_s"]) for r in rows if r["design"] == design and r["end_to_end_s"]]
    return sum(v) / len(v) if v else None
b, f = mean("baseline"), mean("faultmesh")
lines = [
    "# Llama-3.1-70B: UVM baseline vs FaultMesh",
    "",
    "End-to-end inference seconds. Speedup is baseline / FaultMesh.",
    "",
    "| UVM baseline | FaultMesh | Speedup |",
    "|---:|---:|---:|",
    f"| {b:.3f} | {f:.3f} | {b / f:.2f}x |" if b and f else
    f"| {'-' if b is None else f'{b:.3f}'} | {'-' if f is None else f'{f:.3f}'} | - |",
]
print("\n".join(lines))
PY
