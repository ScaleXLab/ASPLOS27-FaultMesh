# FaultMesh

GPU page-fault handling split into a device-side FrontLib and a UVM BackLib.

**FrontLib** (`frontLib/frontend_prefault_common.cuh`) intercepts managed-memory loads. The miss path is direct: one `atomicCAS` winner per 4 KiB page touches that page. The hit path keeps `last_page` and `last_blk` in registers and reads a 2 MiB block counter before the per-page status table.

**BackLib** (`backLib/`) is the NVIDIA open kernel module tree. The FaultMesh profile is `uvm_merge_dispatch=0` and `uvm_parallel_async_copy_map_enable=1`, loaded by `scripts/load_faultmesh.sh`.

## Hardware and software

* NVIDIA GPU, compute capability 8.0 (tested on A100-SXM4-40GB).
* CUDA 12.4 and the open kernel driver 550.54.14.
* Linux kernel headers matching `uname -r`.
* Run the load scripts as root.

## Layout

| Path | Contents |
|---|---|
| `frontLib/` | Device header used by the benchmarks |
| `backLib/` | Open GPU kernel modules, including `perf_ours.sh` and `perf_baseline.sh` |
| `microbenchmark/` | GPGPU applications (2DCONV, ATAX, BICG, GEMM, GESUMMV, MVT, hellinger, nw, XSBench, bfs) |
| `scripts/` | Load the driver and compile the applications |

## Setup

Build and load FaultMesh:

```bash
sudo bash scripts/load_faultmesh.sh
```

Load the UVM baseline instead:

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
