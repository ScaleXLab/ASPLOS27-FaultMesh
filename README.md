# FaultMesh: Characterizing and Taming GPU UVM Page Faults

Qi Chen, Guanyi Chen, Jian Zhang

Artifact evaluation for ASPLOS'27.

Note: All commands in this guide should be run as root (`sudo -i` or `sudo su`).

## Hardware Requirements

- NVIDIA Ampere GPU. The artifact was evaluated on an A100-SXM4-40GB (`sm_80`).

## Software Requirements

- GCC >= 5.4.0 with C++11 and POSIX threads support.
- CUDA 12.4 with Nvidia open-source driver 550.54.14 (see below).
- GNU Make, Python 3, and wget or curl.
- Linux kernel headers for the running kernel. Evaluated on Ubuntu 22.04 and Linux 6.8.0.


## Layout

| Path | Contents |
|---|---|
| `frontLib/` | FaultMesh-FrontLib Design, used by the applications |
| `backLib/` | FaultMesh-BackLib Design, included in Open GPU kernel modules |
| `application/` | 2DCONV, ATAX, BICG, GEMM, GESUMMV, MVT, hellinger, nw, XSBench, bfs |
| `scripts/env/` | Download the 550 userspace and CUDA 12.4, switch the kernel modules, and restore them |
| `scripts/experiment/` | Application comparison |

## 🚀 One-Click Reproduction

Reproduce our main results with just a few commands.

From the repository root:

```bash
bash scripts/env/download_nvidia_550.sh
bash scripts/env/switch_to_faultmesh.sh
bash scripts/experiment/run_baseline_vs_faultmesh.sh
```

`download_nvidia_550.sh` downloads and installs CUDA 12.4 Lib.

`switch_to_faultmesh.sh` saves the machine's current kernel modules, installs the 550.54.14 kernel modules. 

`run_baseline_vs_faultmesh.sh` compare UVM and FaultMesh under 10 GPGPU applications.

## Restore the original driver

After the experiments, you can put back your kernel using:

```bash
sudo bash scripts/env/restore_host_nvidia.sh
```
