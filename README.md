# FaultMesh

GPU page-fault handling split into a device-side FrontLib and a UVM BackLib.

**FrontLib** (`frontLib/frontend_prefault_common.cuh`) intercepts managed-memory loads. The miss path is direct: one `atomicCAS` winner per 4 KiB page touches that page. The hit path keeps `last_page` and `last_blk` in registers and reads a 2 MiB block counter before the per-page status table.

**BackLib** (`backLib/`) is the NVIDIA open kernel module tree. The FaultMesh profile is `uvm_merge_dispatch=0` and `uvm_parallel_async_copy_map_enable=1`.

Tested on an A100-SXM4-40GB. The kernel modules are open driver 550.54.14, and `libcuda` must be the same version. The matching toolkit is CUDA 12.4.0.

## Layout

| Path | Contents |
|---|---|
| `frontLib/` | Device header used by the benchmarks |
| `backLib/` | Open GPU kernel modules |
| `microbenchmark/` | 2DCONV, ATAX, BICG, GEMM, GESUMMV, MVT, hellinger, nw, XSBench, bfs |
| `scripts/build/` | Download 550, switch the machine to it, and restore the previous driver |
| `scripts/experiment/` | Run the direct FrontLib + FaultMesh versus UVM baseline comparison |

From the repository root:

```bash
bash scripts/build/download_nvidia_550.sh
sudo bash scripts/build/switch_to_faultmesh.sh
bash scripts/experiment/run_direct_vs_uvm.sh
```

`download_nvidia_550.sh` fetches the 550.54.14 driver (libcuda, NVML, nvidia-smi, GSP firmware) and the CUDA 12.4.0 toolkit into `third_party/`. It does not replace the driver yet. If CUDA 12.4 is already installed on the machine, that toolkit is reused instead of writing a second copy.

`switch_to_faultmesh.sh` records the current kernel modules, `libcuda` / `libnvidia-ml` links, and `/usr/local/cuda` under `third_party/host-snapshot/`. It then loads FaultMesh and points the CUDA libraries at 550.54.14.

`run_direct_vs_uvm.sh` compiles the ~4 GiB applications, runs each one on the UVM baseline and on direct FrontLib + FaultMesh, and prints the speedup table. Results are written to `results/direct_vs_uvm_<date>/`.

bfs reads `microbenchmark/bfs/graph6M.txt` (Rodinia, 6291456 nodes). The other applications do not need an external dataset. If that graph is absent, bfs is marked skipped and the rest of the table is still produced.

After the experiments, restore the previous driver and CUDA libraries:

```bash
sudo bash scripts/build/restore_host_nvidia.sh
```

Open a new shell after switching or restoring so `PATH` picks up `/etc/profile.d/faultmesh-550.sh`.
