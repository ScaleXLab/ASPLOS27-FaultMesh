// CAGRA-style ANN microbenchmark on UVM, with the two phases timed separately.
//
// build : one kNN-graph refinement pass (NN-descent local join). Each warp owns
//         a node u, scores u's current neighbors plus neighbors-of-neighbors,
//         keeps the best DEG and writes the new row.
// search: CAGRA single-CTA style greedy best-first traversal. Each warp owns a
//         query, keeps an itopk candidate list and expands the best unvisited
//         node per iteration.
//
// Dataset and graph live in cudaMallocManaged memory and are moved to the CPU
// before each phase, so every phase starts from cold GPU residency. With
// --frontlib 1 every dataset/graph access goes through FaultMesh-FrontLib.
//
//   cagra_micro --base base.fbin --graph graph.ibin --query query.fbin
//               [--phase build|search|both] [--frontlib 0|1]
// The number of vectors is the row count in the graph header.

#include <cuda_runtime.h>

#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "frontend_prefault_common.cuh"

#define CHECK(x)                                                                   \
    do {                                                                           \
        cudaError_t e_ = (x);                                                      \
        if (e_ != cudaSuccess) {                                                   \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,              \
                    cudaGetErrorString(e_));                                       \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

constexpr int DIM = 128;
constexpr int VEC4 = DIM / 4;           // float4 per vector
constexpr int DEG = 64;                 // graph degree
constexpr int LIST = 64;                // itopk / build list size
constexpr int TEAM = 8;                 // lanes per distance computation
constexpr int TEAMS = 32 / TEAM;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BUILD_R1 = 8;             // neighbors expanded in build
constexpr int BUILD_R2 = 16;            // neighbors-of-neighbors per expanded neighbor
constexpr int BUILD_CAND = BUILD_R1 * BUILD_R2;
constexpr int MAX_SEARCH_WIDTH = 4;
constexpr int MAX_CAND = (BUILD_CAND + DEG) > (MAX_SEARCH_WIDTH * DEG) ? (BUILD_CAND + DEG) : (MAX_SEARCH_WIDTH * DEG);
constexpr uint32_t VISITED = 0x80000000u;
constexpr uint32_t INVALID = 0xffffffffu;

// Plain UVM access and FrontLib access share one kernel body.
template <typename T, bool PF>
struct access_t;

template <typename T>
struct access_t<T, false> {
    T* p;
    __device__ access_t(const prefault_array_host_t<T>& a) : p(a.ptr) {}
    __device__ __forceinline__ T load(uint64_t i) const { return p[i]; }
    __device__ __forceinline__ void store(uint64_t i, T v) const { p[i] = v; }
    __device__ __forceinline__ void ensure(uint64_t) const {}
};

template <typename T>
struct access_t<T, true> {
    uvm_prefault_array_dedup_t<T> v;
    __device__ access_t(const prefault_array_host_t<T>& a)
        : v(a.ptr, a.n_elems, a.page_shift, a.page_status, nullptr, nullptr) {}
    __device__ __forceinline__ T load(uint64_t i) const { return v.load(i); }
    __device__ __forceinline__ void store(uint64_t i, T x) const { v.store(i, x); }
    __device__ __forceinline__ void ensure(uint64_t i) const
    {
        if (v.checks) v.ensure(v.page_of(i));
    }
};

struct warp_list_t {
    float dist[LIST];
    uint32_t id[LIST];
};

struct warp_smem_t {
    warp_list_t list;
    float4 q[VEC4];
    uint32_t cand[MAX_CAND];
    float cdist[MAX_CAND];
};

__device__ __forceinline__ float team_reduce(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 4, TEAM);
    v += __shfl_xor_sync(0xffffffffu, v, 2, TEAM);
    v += __shfl_xor_sync(0xffffffffu, v, 1, TEAM);
    return v;
}

// Squared L2 distance between the warp's query in smem and dataset row `node`.
// Called by all 32 lanes; each team of 8 lanes handles one node.
template <bool PF>
__device__ __forceinline__ float team_distance(const access_t<float4, PF>& data,
                                               const float4* q, uint32_t node, bool active)
{
    const int t = threadIdx.x % TEAM;
    float acc = 0.f;
    if (active) {
        const uint64_t base = (uint64_t)node * VEC4;
#pragma unroll
        for (int k = 0; k < VEC4 / TEAM; ++k) {
            const int j = t + k * TEAM;
            const float4 x = data.load(base + j);
            const float4 y = q[j];
            const float d0 = x.x - y.x, d1 = x.y - y.y, d2 = x.z - y.z, d3 = x.w - y.w;
            acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
    }
    return team_reduce(acc);
}

// Scores cand[0..m) and writes cdist[0..m).
template <bool PF>
__device__ void score_candidates(const access_t<float4, PF>& data, warp_smem_t& s, int m)
{
    const int lane = threadIdx.x % 32;
    const int team = lane / TEAM;
    if (PF) {
        for (int idx = lane; idx < m; idx += 32) data.ensure((uint64_t)s.cand[idx] * VEC4);
        __syncwarp();
    }
    for (int base = 0; base < m; base += TEAMS) {
        const int idx = base + team;
        const bool active = idx < m;
        const float d = team_distance<PF>(data, s.q, active ? s.cand[idx] : 0u, active);
        if (active && (lane % TEAM) == 0) s.cdist[idx] = d;
    }
    __syncwarp();
}

__device__ __forceinline__ bool list_contains(const warp_list_t& l, uint32_t node)
{
    const int lane = threadIdx.x % 32;
    const bool hit = ((l.id[lane] & ~VISITED) == node) || ((l.id[lane + 32] & ~VISITED) == node);
    return __any_sync(0xffffffffu, hit);
}

// Replaces the worst list entry with each candidate that beats it.
__device__ void merge_candidates(warp_list_t& l, const warp_smem_t& s, int m)
{
    const int lane = threadIdx.x % 32;
    for (int c = 0; c < m; ++c) {
        const float cd = s.cdist[c];
        float w = l.dist[lane];
        int wi = lane;
        if (l.dist[lane + 32] > w) { w = l.dist[lane + 32]; wi = lane + 32; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ow = __shfl_xor_sync(0xffffffffu, w, off);
            const int oi = __shfl_xor_sync(0xffffffffu, wi, off);
            if (ow > w || (ow == w && oi < wi)) { w = ow; wi = oi; }
        }
        if (cd < w && lane == 0) {
            l.dist[wi] = cd;
            l.id[wi] = s.cand[c];
        }
        __syncwarp();
    }
}

// Drops candidates already in the list and duplicates within cand[].
__device__ int dedup_candidates(const warp_list_t& l, warp_smem_t& s, int m)
{
    const int lane = threadIdx.x % 32;
    int out = 0;
    for (int c = 0; c < m; ++c) {
        const uint32_t node = s.cand[c];
        bool dup = node == INVALID || list_contains(l, node);
        for (int k = lane; k < out && !dup; k += 32) dup |= (s.cand[k] == node);
        dup = __any_sync(0xffffffffu, dup);
        __syncwarp();
        if (!dup) {
            if (lane == 0) s.cand[out] = node;
            ++out;
        }
        __syncwarp();
    }
    return out;
}

template <bool PF>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
build_kernel(prefault_array_host_t<float4> data_a, prefault_array_host_t<uint32_t> graph_a,
             prefault_array_host_t<uint32_t> out_a, uint32_t n)
{
    __shared__ warp_smem_t smem[WARPS_PER_BLOCK];
    const access_t<float4, PF> data(data_a);
    const access_t<uint32_t, PF> graph(graph_a);
    const access_t<uint32_t, PF> out(out_a);
    const int lane = threadIdx.x % 32;
    warp_smem_t& s = smem[threadIdx.x / 32];
    const uint32_t u = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
    if (u >= n) return;

    s.q[lane] = data.load((uint64_t)u * VEC4 + lane);
    s.cand[lane] = graph.load((uint64_t)u * DEG + lane);
    s.cand[lane + 32] = graph.load((uint64_t)u * DEG + lane + 32);
    s.list.dist[lane] = FLT_MAX; s.list.dist[lane + 32] = FLT_MAX;
    s.list.id[lane] = INVALID;   s.list.id[lane + 32] = INVALID;
    __syncwarp();

    score_candidates<PF>(data, s, DEG);
    s.list.dist[lane] = s.cdist[lane];      s.list.id[lane] = s.cand[lane];
    s.list.dist[lane + 32] = s.cdist[lane + 32]; s.list.id[lane + 32] = s.cand[lane + 32];
    __syncwarp();

    // Local join: neighbors of the first BUILD_R1 neighbors.
    for (int k = lane; k < BUILD_CAND; k += 32) {
        const uint32_t v = s.list.id[k / BUILD_R2];
        const uint32_t w = graph.load((uint64_t)v * DEG + (k % BUILD_R2));
        s.cand[k] = (w == u) ? INVALID : w;
    }
    __syncwarp();
    const int m = dedup_candidates(s.list, s, BUILD_CAND);
    score_candidates<PF>(data, s, m);
    merge_candidates(s.list, s, m);

    out.store((uint64_t)u * DEG + lane, s.list.id[lane]);
    out.store((uint64_t)u * DEG + lane + 32, s.list.id[lane + 32]);
}

__device__ __forceinline__ uint32_t hash32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

template <bool PF>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
search_kernel(prefault_array_host_t<float4> data_a, prefault_array_host_t<uint32_t> graph_a,
              const float4* __restrict__ queries, uint32_t n, uint32_t n_queries,
              int max_iter, int search_width, int k, uint32_t* __restrict__ result)
{
    __shared__ warp_smem_t smem[WARPS_PER_BLOCK];
    const access_t<float4, PF> data(data_a);
    const access_t<uint32_t, PF> graph(graph_a);
    const int lane = threadIdx.x % 32;
    warp_smem_t& s = smem[threadIdx.x / 32];
    const uint32_t qi = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
    if (qi >= n_queries) return;

    s.q[lane] = queries[(uint64_t)qi * VEC4 + lane];
    s.list.dist[lane] = FLT_MAX; s.list.dist[lane + 32] = FLT_MAX;
    s.list.id[lane] = INVALID;   s.list.id[lane + 32] = INVALID;
    // Random entry points, as CAGRA seeds its search.
    s.cand[lane] = hash32(qi * 64u + lane) % n;
    s.cand[lane + 32] = hash32(qi * 64u + lane + 32) % n;
    __syncwarp();
    int m = dedup_candidates(s.list, s, DEG);
    score_candidates<PF>(data, s, m);
    merge_candidates(s.list, s, m);

    for (int it = 0; it < max_iter; ++it) {
        // Expand the search_width best unvisited entries.
        int picked = 0;
        for (int w = 0; w < search_width; ++w) {
            float b = FLT_MAX;
            int bi = -1;
            for (int e = lane; e < LIST; e += 32) {
                if (!(s.list.id[e] & VISITED) && s.list.dist[e] < b) { b = s.list.dist[e]; bi = e; }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, b, off);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                if (ob < b || (ob == b && oi < bi && oi >= 0)) { b = ob; bi = oi; }
            }
            if (bi < 0) break;
            const uint32_t p = s.list.id[bi];
            __syncwarp();
            if (lane == 0) s.list.id[bi] = p | VISITED;
            s.cand[picked * DEG + lane] = graph.load((uint64_t)p * DEG + lane);
            s.cand[picked * DEG + lane + 32] = graph.load((uint64_t)p * DEG + lane + 32);
            __syncwarp();
            ++picked;
        }
        if (picked == 0) break;
        m = dedup_candidates(s.list, s, picked * DEG);
        score_candidates<PF>(data, s, m);
        merge_candidates(s.list, s, m);
    }

    // Top-k by repeated selection.
    for (int r = 0; r < k; ++r) {
        float b = FLT_MAX;
        int bi = -1;
        for (int e = lane; e < LIST; e += 32) {
            if (s.list.dist[e] < b) { b = s.list.dist[e]; bi = e; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, b, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ob < b || (ob == b && oi < bi && oi >= 0)) { b = ob; bi = oi; }
        }
        if (lane == 0) {
            result[(uint64_t)qi * k + r] = bi >= 0 ? (s.list.id[bi] & ~VISITED) : INVALID;
            if (bi >= 0) s.list.dist[bi] = FLT_MAX;
        }
        __syncwarp();
    }
}

static void read_header(FILE* f, uint32_t& n, uint32_t& d, const char* name)
{
    if (fread(&n, 4, 1, f) != 1 || fread(&d, 4, 1, f) != 1) {
        fprintf(stderr, "cannot read header of %s\n", name);
        exit(1);
    }
}

static void read_rows(const char* path, void* dst, size_t row_bytes, uint64_t rows, uint32_t expect_dim)
{
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    uint32_t n, d;
    read_header(f, n, d, path);
    if (d != expect_dim || rows > n) {
        fprintf(stderr, "%s: file has n=%u dim=%u, need rows=%lu dim=%u\n",
                path, n, d, (unsigned long)rows, expect_dim);
        exit(1);
    }
    const size_t chunk = 256ull << 20;
    const size_t total = row_bytes * rows;
    for (size_t off = 0; off < total; off += chunk) {
        const size_t len = total - off < chunk ? total - off : chunk;
        if (fread(static_cast<char*>(dst) + off, 1, len, f) != len) {
            fprintf(stderr, "short read in %s\n", path);
            exit(1);
        }
    }
    fclose(f);
}

// A CPU fault on managed memory allocates on the faulting CPU's NUMA node only,
// with compaction and reclaim on that node before falling back, which stalls
// once the data outgrows one node. A prefetch to the CPU allocates by the
// process memory policy instead (run under numactl --interleave=all).
static void populate_on_cpu(void* ptr, size_t bytes)
{
    CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());
}

template <typename T>
static void init_view(prefault_array_host_t<T>& a, T* ptr, uint64_t n_elems, bool pf)
{
    if (pf) {
        prefault_array_init(&a, ptr, n_elems, 12, false);
    } else {
        a = {};
        a.ptr = ptr;
        a.n_elems = n_elems;
        a.page_shift = 12;
    }
}

int main(int argc, char** argv)
{
    std::string base = "/mnt/nvme0n1/sift100m/base.100M.fbin";
    std::string graph_path = "/mnt/nvme0n1/sift100m/cagra_100M.graph.ibin";
    std::string query = "/mnt/nvme0n1/sift100m/query.fbin";
    std::string phase = "both";
    uint64_t n = 0;
    uint64_t alloc_n = 0;
    uint32_t n_queries = 10000;
    int max_iter = 64, k = 10, search_width = 1;
    bool pf = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--base") base = next();
        else if (a == "--graph") graph_path = next();
        else if (a == "--query") query = next();
        else if (a == "--n") n = std::stoull(next());
        else if (a == "--alloc_n") alloc_n = std::stoull(next());
        else if (a == "--queries") n_queries = std::stoul(next());
        else if (a == "--phase") phase = next();
        else if (a == "--max_iter") max_iter = std::stoi(next());
        else if (a == "--search_width") search_width = std::stoi(next());
        else if (a == "--frontlib") pf = next() == "1";
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }

    uint32_t gn, gd;
    {
        FILE* f = fopen(graph_path.c_str(), "rb");
        if (!f) { perror(graph_path.c_str()); return 1; }
        read_header(f, gn, gd, graph_path.c_str());
        fclose(f);
    }
    if (gd != DEG) { fprintf(stderr, "graph degree %u, expected %d\n", gd, DEG); return 1; }
    if (n == 0 || n > gn) n = gn;
    if (alloc_n < n) alloc_n = n;
    if (search_width < 1) search_width = 1;
    if (search_width > MAX_SEARCH_WIDTH) search_width = MAX_SEARCH_WIDTH;

    setvbuf(stdout, nullptr, _IOLBF, 0);
    const auto wall0 = std::chrono::steady_clock::now();
    auto stamp = [&](const char* what) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        printf("[host %8.1fs] %s\n", s, what);
    };

    pf_uvm_warmup();

    float4* data = nullptr;
    uint32_t* graph = nullptr;
    uint32_t* out_graph = nullptr;
    // Arrays hold alloc_n rows; the kernels only touch the first n.
    CHECK(cudaMallocManaged(&data, alloc_n * DIM * sizeof(float)));
    CHECK(cudaMallocManaged(&graph, alloc_n * DEG * sizeof(uint32_t)));
    populate_on_cpu(data, alloc_n * DIM * sizeof(float));
    populate_on_cpu(graph, alloc_n * DEG * sizeof(uint32_t));
    stamp("managed allocations populated on CPU");
    read_rows(base.c_str(), data, DIM * sizeof(float), alloc_n, DIM);
    stamp("dataset read");
    read_rows(graph_path.c_str(), graph, DEG * sizeof(uint32_t), n, DEG);
    stamp("graph read");
    // Graph ids beyond n (graph built on a larger prefix) are remapped.
    if (n < gn)
        for (uint64_t i = 0; i < n * DEG; ++i)
            if (graph[i] >= n) graph[i] %= n;

    std::vector<float> hq((size_t)n_queries * DIM);
    {
        FILE* f = fopen(query.c_str(), "rb");
        uint32_t qn, qd;
        read_header(f, qn, qd, query.c_str());
        if (qd != DIM) { fprintf(stderr, "query dim %u\n", qd); return 1; }
        if (n_queries > qn) n_queries = qn;
        if (fread(hq.data(), sizeof(float), (size_t)n_queries * DIM, f) != (size_t)n_queries * DIM) {
            fprintf(stderr, "short query read\n"); return 1;
        }
        fclose(f);
    }
    float4* d_queries = nullptr;
    uint32_t* d_result = nullptr;
    CHECK(cudaMalloc(&d_queries, (size_t)n_queries * DIM * sizeof(float)));
    CHECK(cudaMalloc(&d_result, (size_t)n_queries * k * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_queries, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice));

    printf("N=%lu dim=%d degree=%d queries=%u\n", (unsigned long)n, DIM, DEG, n_queries);
    printf("managed: dataset %.2f GiB, graph %.2f GiB\n",
           alloc_n * DIM * 4.0 / (1 << 30), alloc_n * DEG * 4.0 / (1 << 30));
    if (alloc_n > n) printf("allocated rows=%lu, touched rows=%lu\n", (unsigned long)alloc_n, (unsigned long)n);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    prefault_array_host_t<float4> data_a;
    prefault_array_host_t<uint32_t> graph_a, out_a;
    const int threads = WARPS_PER_BLOCK * 32;

    if (phase == "build" || phase == "both") {
        CHECK(cudaMallocManaged(&out_graph, alloc_n * DEG * sizeof(uint32_t)));
        populate_on_cpu(out_graph, alloc_n * DEG * sizeof(uint32_t));
        memset(out_graph, 0, alloc_n * DEG * sizeof(uint32_t));
        stamp("build: output graph zeroed");
        init_view(data_a, data, alloc_n * VEC4, pf);
        init_view(graph_a, graph, alloc_n * DEG, pf);
        init_view(out_a, out_graph, alloc_n * DEG, pf);
        prefault_to_cpu(data, alloc_n * DIM * sizeof(float));
        prefault_to_cpu(graph, alloc_n * DEG * sizeof(uint32_t));
        prefault_to_cpu(out_graph, alloc_n * DEG * sizeof(uint32_t));
        stamp("build: inputs on CPU, launching");
        const unsigned blocks = (unsigned)((n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        CHECK(cudaEventRecord(t0));
        if (pf) build_kernel<true><<<blocks, threads>>>(data_a, graph_a, out_a, (uint32_t)n);
        else    build_kernel<false><<<blocks, threads>>>(data_a, graph_a, out_a, (uint32_t)n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaGetLastError());
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        stamp("build: kernel done");
        populate_on_cpu(out_graph, n * DEG * sizeof(uint32_t));
        unsigned long long sum = 0;
        for (uint64_t i = 0; i < n * DEG; i += 997) sum += out_graph[i];
        printf("Build phase: %.6fs checksum=%llu\n", ms / 1000.0, sum);
        if (pf) { prefault_array_destroy(&data_a); prefault_array_destroy(&graph_a); prefault_array_destroy(&out_a); }
        CHECK(cudaFree(out_graph));
        stamp("build: done");
    }

    if (phase == "search" || phase == "both") {
        init_view(data_a, data, alloc_n * VEC4, pf);
        init_view(graph_a, graph, alloc_n * DEG, pf);
        prefault_to_cpu(data, alloc_n * DIM * sizeof(float));
        prefault_to_cpu(graph, alloc_n * DEG * sizeof(uint32_t));
        stamp("search: inputs on CPU, launching");
        const unsigned blocks = (n_queries + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CHECK(cudaEventRecord(t0));
        if (pf) search_kernel<true><<<blocks, threads>>>(data_a, graph_a, d_queries, (uint32_t)n, n_queries, max_iter, search_width, k, d_result);
        else    search_kernel<false><<<blocks, threads>>>(data_a, graph_a, d_queries, (uint32_t)n, n_queries, max_iter, search_width, k, d_result);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaGetLastError());
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::vector<uint32_t> res((size_t)n_queries * k);
        CHECK(cudaMemcpy(res.data(), d_result, res.size() * 4, cudaMemcpyDeviceToHost));
        unsigned long long sum = 0;
        for (uint32_t v : res) sum += v;
        printf("Search phase: %.6fs checksum=%llu first=[%u %u %u]\n",
               ms / 1000.0, sum, res[0], res[1], res[2]);
        if (pf) { prefault_array_destroy(&data_a); prefault_array_destroy(&graph_a); }
        stamp("search: done");
    }
    return 0;
}
