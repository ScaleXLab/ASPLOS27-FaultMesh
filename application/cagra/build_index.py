#!/usr/bin/env python3
"""Build the CAGRA kNN graph of a SIFT .fbin dataset with cuVS (pip install cuvs-cu12).

cagra_micro reads the graph from --graph_out. For the graph used by
scripts/experiment/run_cagra.sh (about 13 minutes on an A100):

  python3 build_index.py --base /mnt/nvme0n1/sift100m/base.100M.fbin \
      --host --graph_out /mnt/nvme0n1/sift100m/cagra_100M.graph.ibin

--host keeps the dataset in host memory when it is larger than the GPU.
"""

import argparse
import struct
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--n_base', type=int, default=0)
    parser.add_argument('--graph_degree', type=int, default=64)
    parser.add_argument('--intermediate_degree', type=int, default=0,
                        help='intermediate kNN graph degree (default 2x graph_degree); the intermediate '
                             'graph is moved to the GPU, so 100M needs 64 on a 40 GB GPU')
    parser.add_argument('--out', default='', help='saved cuVS index (omit to skip)')
    parser.add_argument('--graph_out', default='',
                        help='also write the graph as .ibin (uint32 n, uint32 degree, rows)')
    parser.add_argument('--host', action='store_true',
                        help='keep the dataset in host memory (datasets larger than the GPU)')
    parser.add_argument('--n_probes', type=int, default=0,
                        help='IVF-PQ probes for the initial kNN graph, searched with float32 LUT and '
                             'distances (0 keeps the cuVS heuristic, which leaves too many invalid '
                             'neighbors at 100M)')
    args = parser.parse_args()

    import rmm
    rmm.reinitialize(pool_allocator=False, managed_memory=False)
    from cuvs.neighbors import cagra
    from pylibraft.common import device_ndarray

    with open(args.base, 'rb') as f:
        n, dim = struct.unpack('II', f.read(8))
    if args.n_base:
        n = min(n, args.n_base)
    data = np.memmap(args.base, dtype=np.float32, mode='r', offset=8, shape=(n, dim))
    print(f"dataset: N={n}, dim={dim}, {n * dim * 4 / 2**30:.2f} GiB", flush=True)

    t0 = time.time()
    if args.host:
        dataset = np.ascontiguousarray(data)
        print(f"loaded into host memory in {time.time() - t0:.1f}s", flush=True)
    else:
        dataset = device_ndarray(np.ascontiguousarray(data))
        print(f"copied to device in {time.time() - t0:.1f}s", flush=True)

    extra = {}
    if args.n_probes:
        from cuvs.neighbors import ivf_pq
        extra['ivf_pq_search_params'] = ivf_pq.SearchParams(
            n_probes=args.n_probes, lut_dtype=np.float32, internal_distance_dtype=np.float32)
    params = cagra.IndexParams(graph_degree=args.graph_degree,
                               intermediate_graph_degree=args.intermediate_degree or args.graph_degree * 2,
                               **extra)
    t0 = time.time()
    index = cagra.build(params, dataset)
    print(f"{'host' if args.host else 'device'}-memory build time: {time.time() - t0:.1f}s", flush=True)

    if args.out:
        cagra.save(args.out, index, include_dataset=True)
        print(f"saved {args.out}", flush=True)

    if args.graph_out:
        graph = index.graph.copy_to_host()
        with open(args.graph_out, 'wb') as f:
            f.write(struct.pack('II', graph.shape[0], graph.shape[1]))
            graph.astype(np.uint32, copy=False).tofile(f)
        print(f"saved graph {graph.shape} to {args.graph_out}", flush=True)


if __name__ == '__main__':
    main()
