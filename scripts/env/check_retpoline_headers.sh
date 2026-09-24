#!/usr/bin/env bash
# Kernels through 6.8.0-94-generic define CONFIG_RETPOLINE.
# Later Ubuntu 6.8 headers rename it to CONFIG_MITIGATION_RETPOLINE.
# 550.54.14 then injects a raw indirect jump and the build stops in nv.o.
# If that rename is present, define the old name from the new one.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL="${KERNEL_UNAME:-$(uname -r)}"
CONFIG=""
for candidate in \
  "/usr/src/linux-headers-${KERNEL}/.config" \
  "/lib/modules/${KERNEL}/build/.config" \
  "/boot/config-${KERNEL}"; do
  if [[ -f "${candidate}" ]]; then
    CONFIG="${candidate}"
    break
  fi
done

[[ -n "${CONFIG}" ]] || {
  echo "ERROR: no kernel config for ${KERNEL}" >&2
  exit 1
}

has_old=0
has_new=0
grep -q '^CONFIG_RETPOLINE=y' "${CONFIG}" && has_old=1
grep -q '^CONFIG_MITIGATION_RETPOLINE=y' "${CONFIG}" && has_new=1

echo "kernel ${KERNEL}: config ${CONFIG}"
echo "CONFIG_RETPOLINE=${has_old} CONFIG_MITIGATION_RETPOLINE=${has_new}"

if [[ "${has_old}" -eq 1 || "${has_new}" -eq 0 ]]; then
  echo "retpoline name matches 550.54.14. No patch."
  exit 0
fi

echo "headers renamed CONFIG_RETPOLINE. Adding the two-line compatibility define."
patch_one() {
  local file="$1"
  if grep -q 'CONFIG_MITIGATION_RETPOLINE' "${file}"; then
    echo "already patched: ${file}"
    return 0
  fi
  python3 - "${file}" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = "#if !defined(CONFIG_RETPOLINE)\n"
insert = (
    "#if defined(CONFIG_MITIGATION_RETPOLINE) && !defined(CONFIG_RETPOLINE)\n"
    "#define CONFIG_RETPOLINE CONFIG_MITIGATION_RETPOLINE\n"
    "#endif\n"
)
if needle not in text:
    sys.exit(f"missing retpoline check in {path}")
path.write_text(text.replace(needle, insert + needle, 1))
PY
  echo "patched: ${file}"
}

patch_one "${ROOT}/backLib/kernel-open/nvidia/nv.c"
patch_one "${ROOT}/backLib/kernel-open/nvidia-modeset/nvidia-modeset-linux.c"
