// Count how many threads' values reach the host after a warp-shuffle block
// reduction that follows a data-dependent early-exit loop.
// Expected: total count == n for every variant.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// V: 0 = __activemask() (as in quantcuda), 1 = __syncwarp() then __activemask(),
//    2 = full mask 0xffffffff
template <int V>
__global__ void k(unsigned long long* block_count, int n, int steps) {
    unsigned long long cnt = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)n; i += stride) {
        bool alive = true;
        for (int j = 0; j < steps && alive; ++j) {
            if (hash32((uint32_t)i * 977u + j) % 50u == 0) { alive = false; break; }
        }
        if (!alive) { ++cnt; continue; }
        ++cnt;
    }
    if (V == 1) __syncwarp();
    unsigned mask = (V == 2) ? 0xffffffffu : __activemask();
    int lane = threadIdx.x & 31;
    int active = __popc(mask);
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long o = __shfl_down_sync(mask, cnt, off);
        if (lane + off < active) cnt += o;
    }
    __shared__ unsigned long long sm[32];
    int warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = cnt;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    cnt = threadIdx.x < nw ? sm[threadIdx.x] : 0;
    if (warp == 0) {
        unsigned m2 = (V == 2) ? 0xffffffffu : __activemask();
        int a2 = __popc(m2);
        for (int off = 16; off > 0; off >>= 1) {
            unsigned long long o = __shfl_down_sync(m2, cnt, off);
            if (lane + off < a2) cnt += o;
        }
    }
    if (threadIdx.x == 0) block_count[blockIdx.x] = cnt;
}

template <int V>
bool run(int n, int steps, int bs) {
    int grid = (n + bs - 1) / bs; if (grid > 65535) grid = 65535;
    unsigned long long* d; cudaMalloc(&d, grid * 8);
    cudaMemset(d, 0, grid * 8);
    k<V><<<grid, bs>>>(d, n, steps);
    cudaError_t e = cudaDeviceSynchronize();
    unsigned long long h[65535]; cudaMemcpy(h, d, grid * 8, cudaMemcpyDeviceToHost);
    unsigned long long t = 0; for (int i = 0; i < grid; ++i) t += h[i];
    cudaFree(d);
    printf("variant %d n=%d steps=%d block=%d: counted %llu %s%s\n", V, n, steps, bs, t,
           t == (unsigned long long)n ? "OK" : "MISMATCH", e ? " (cuda error)" : "");
    return t == (unsigned long long)n;
}

int main() {
    int bad = 0;
    int ns[] = {1000, 100000};
    for (int n : ns) for (int steps : {1, 52}) {
        bad += !run<0>(n, steps, 256);
        bad += !run<1>(n, steps, 256);
        bad += !run<2>(n, steps, 256);
    }
    printf("%s\n", bad ? "FAILED" : "ALL OK");
    return bad != 0;
}
