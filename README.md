# FaultMesh: Characterizing and Taming GPU UVM Page Faults

Qi Chen, Guanyi Chen, Jian Zhang

Artifact evaluation for ASPLOS'27.

Note: All commands in this guide should be run as root (`sudo -i` or `sudo su`).

## Hardware Requirements

- NVIDIA Ampere GPU. The artifact was evaluated on an A100-SXM4-40GB (`sm_80`).

## Software Requirements

- Linux with kernel headers for the running kernel. Evaluated on Ubuntu 22.04 and Linux 6.8.0.
- `gcc`, `make`, `python3`, and `wget` or `curl`.
- Open kernel modules 550.54.14, provided in `backLib/`. This is the kernel driver.
- Userspace libraries from the same 550.54.14 driver release: `libcuda`, NVML, `nvidia-smi`, and the GSP firmware.
- CUDA toolkit 12.4.0 (`nvcc` and the toolkit libraries), installed into `cuda-toolkit/` in this repository. `/usr/local/cuda` is not changed.


## Layout

| Path | Contents |
|---|---|
| `frontLib/` | FaultMesh-FrontLib Design, used by the applications |
| `backLib/` | FaultMesh-BackLib Design, included in Open GPU kernel modules |
| `application/` | 2DCONV, ATAX, BICG, GEMM, GESUMMV, MVT, hellinger, nw, XSBench, bfs |
| `cuda-toolkit/` | CUDA 12.4.0 toolkit downloaded by the setup script |
| `scripts/env/` | Download 550 userspace and CUDA 12.4 into this repository, switch the kernel modules, and restore them |
| `scripts/experiment/` | Application comparison |

From the repository root:

```bash
bash scripts/env/download_nvidia_550.sh
bash scripts/env/switch_to_faultmesh.sh
bash scripts/experiment/run_baseline_vs_faultmesh.sh
```

`switch_to_faultmesh.sh` records the current kernel modules, then builds and loads the `backLib/` kernel modules. The CUDA toolkit stays in `cuda-toolkit/`. The 550.54.14 `libcuda`, NVML, and `nvidia-smi` stay in this repository and are selected with `PATH` and `LD_LIBRARY_PATH`. `/usr/local/cuda` is left unchanged.

Open a new shell after switching so `PATH` picks up `/etc/profile.d/faultmesh-550.sh`.

## Restore the original driver

After the experiments, put back the kernel modules recorded before the switch:

```bash
sudo bash scripts/env/restore_host_nvidia.sh
```

Open a new shell after restoring so the previous `PATH` and library links are picked up.
