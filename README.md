# FaultMesh

GPU page-fault handling split into a device-side FrontLib and a UVM BackLib.

**FrontLib** (`frontLib/frontend_prefault_common.cuh`) intercepts managed-memory loads. The miss path is direct: one `atomicCAS` winner per 4 KiB page touches that page. The hit path keeps `last_page` and `last_blk` in registers and reads a 2 MiB block counter before the per-page status table.

**BackLib** (`backLib/`) is the NVIDIA open kernel module tree. The FaultMesh profile is `uvm_merge_dispatch=0` and `uvm_parallel_async_copy_map_enable=1`, loaded by `scripts/load_faultmesh.sh`.

## Hardware and software

* NVIDIA GPU, compute capability 8.0 (tested on A100-SXM4-40GB).
* Linux kernel headers matching `uname -r`.
* The FaultMesh kernel modules are the open driver **550.54.14**. `libcuda` and `nvidia-smi` have to be that same version. A newer driver already installed on the machine (for example 560) will not run these modules.
* CUDA toolkit **12.4.0**, which is the toolkit released with driver 550.54.14.

## Layout

| Path | Contents |
|---|---|
| `frontLib/` | Device header used by the benchmarks |
| `backLib/` | Open GPU kernel modules, including `perf_ours.sh` and `perf_baseline.sh` |
| `microbenchmark/` | GPGPU applications (2DCONV, ATAX, BICG, GEMM, GESUMMV, MVT, hellinger, nw, XSBench, bfs) |
| `scripts/` | Load the driver and compile the applications |

## Setup

Download the 550.54.14 driver and the CUDA 12.4 toolkit into `third_party/` (not installed yet):

```bash
bash scripts/download_nvidia_550.sh
```

That fetches:

* `NVIDIA-Linux-x86_64-550.54.14.run` — driver userspace (`libcuda.so.550.54.14`, NVML, `nvidia-smi`) and GSP firmware
* `cuda_12.4.0_550.54.14_linux.run` — CUDA 12.4 toolkit, installed under `third_party/cuda-12.4`

If the machine already has another NVIDIA driver, switch to FaultMesh and remember the previous stack:

```bash
sudo bash scripts/switch_to_faultmesh.sh
```

The script saves the current kernel modules, `libcuda` / `libnvidia-ml` symlinks, and the `/usr/local/cuda` link under `third_party/host-snapshot/`. It then installs the 550 firmware, builds `backLib/`, loads it with async copy-map on and merge dispatch off, and points `libcuda` and `/usr/local/cuda` at the 550 libraries.

After the experiments, put the previous driver and CUDA libraries back:

```bash
sudo bash scripts/restore_host_nvidia.sh
```

Open a new shell after either switch so `PATH` picks up `/etc/profile.d/faultmesh-550.sh`.

Load the UVM baseline instead of FaultMesh, without changing userspace:

```bash
sudo bash scripts/load_baseline.sh
```

`BUILD_KERNEL=0` skips rebuilding the modules when `backLib/` is already installed.

Compile the applications:

```bash
bash scripts/compile_microbenchmarks.sh
```

Run one application with the FrontLib kernel:

```bash
cd microbenchmark/BICG
BENCH_VARIANT=frontend ./run
```

`BENCH_VARIANT=baseline` runs the plain kernel on the same binary. The sources default to the historical ~4 GiB problem sizes.
