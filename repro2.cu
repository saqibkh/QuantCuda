// Early-exit Monte Carlo loop followed by a warp-shuffle block reduction,
// structured like QuantCUDA's barrier mode 0 kernel. Every thread reaches the
// reduction, so a correct run counts exactly n paths.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kernels_common.cuh"
using namespace quantcuda::gpu;

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__global__ void init_k(curandStatePhilox4_32_10_t* st, int n) {
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < (size_t)n; i += (size_t)gridDim.x*blockDim.x)
        curand_init(42ull, i, 0, &st[i]);
}
__device__ unsigned long long g_bad_entry;  // threads that entered the reduction with a non-full mask

// RNG: 0 curand_normal_double, 1 integer hash. RED: 0 __activemask, 1 __syncwarp + __activemask, 2 0xffffffff
template <int RNG, int RED>
__global__ void k(curandStatePhilox4_32_10_t* states, dev::MomentPartial* out, int n, int steps) {
    double dt = 1.0 / steps, drift = (0.05 - 0.5*0.25*0.25)*dt, diff = 0.25*sqrt(dt);
    double b = 140.0, kk = 105.0;
    dev::RunningMoments<double> m;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < (size_t)n; i += stride) {
        curandStatePhilox4_32_10_t st; if (RNG == 0) st = states[i];
        double s = 100.0; bool alive = s < b;
        for (int j = 0; j < steps && alive; ++j) {
            double z = RNG == 0 ? curand_normal_double(&st)
                                : ((int)(hash32((uint32_t)i*977u + j) % 2001u) - 1000) / 577.35;
            s *= exp(drift + diff*z);
            if (s >= b) { alive = false; break; }
        }
        if (!alive) { dev::moments_add(m, 0.0); continue; }
        dev::moments_add(m, fmax(s - kk, 0.0));
    }
    if (RED == 1) __syncwarp();
    unsigned mask = RED == 2 ? 0xffffffffu : __activemask();
    if (__popc(mask) != 32) atomicAdd(&g_bad_entry, 1ull);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int active = __popc(mask);
    for (int off = 16; off > 0; off >>= 1) {
        dev::RunningMoments<double> o;
        o.count = __shfl_down_sync(mask, m.count, off);
        o.mean = __shfl_down_sync(mask, m.mean, off);
        o.m2 = __shfl_down_sync(mask, m.m2, off);
        if (lane + off < active) m = dev::moments_merge(m, o);
    }
    __shared__ dev::RunningMoments<double> sm[32];
    if (lane == 0) sm[warp] = m;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    m = threadIdx.x < nw ? sm[threadIdx.x] : dev::RunningMoments<double>{};
    if (warp == 0) {
        unsigned m2 = RED == 2 ? 0xffffffffu : __activemask();
        int a2 = __popc(m2);
        for (int off = 16; off > 0; off >>= 1) {
            dev::RunningMoments<double> o;
            o.count = __shfl_down_sync(m2, m.count, off);
            o.mean = __shfl_down_sync(m2, m.mean, off);
            o.m2 = __shfl_down_sync(m2, m.m2, off);
            if (lane + off < a2) m = dev::moments_merge(m, o);
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = dev::moments_partial(m);
}

template <int RNG, int RED>
bool run(int n, int steps, int bs) {
    int grid = (n + bs - 1) / bs; if (grid > 65535) grid = 65535;
    curandStatePhilox4_32_10_t* st; cudaMalloc(&st, (size_t)n * sizeof(*st));
    init_k<<<grid, bs>>>(st, n);
    dev::MomentPartial* d; cudaMalloc(&d, grid * sizeof(*d)); cudaMemset(d, 0, grid * sizeof(*d));
    unsigned long long zero = 0; cudaMemcpyToSymbol(g_bad_entry, &zero, 8);
    k<RNG, RED><<<grid, bs>>>(st, d, n, steps);
    cudaError_t e = cudaDeviceSynchronize();
    static dev::MomentPartial h[65535]; cudaMemcpy(h, d, grid * sizeof(*d), cudaMemcpyDeviceToHost);
    unsigned long long t = 0; double sum = 0;
    for (int i = 0; i < grid; ++i) { t += h[i].count; sum += h[i].mean * h[i].count; }
    unsigned long long bad = 0; cudaMemcpyFromSymbol(&bad, g_bad_entry, 8);
    cudaFree(st); cudaFree(d);
    printf("rng=%s reduce=%-9s n=%6d steps=%2d: counted %6llu (expected %d) %s, threads entering reduction with partial warp mask: %llu%s\n",
           RNG ? "hash  " : "curand", RED == 0 ? "activemask" : RED == 1 ? "syncwarp" : "full", n, steps, t, n,
           t == (unsigned long long)n ? "OK      " : "MISMATCH", bad, e ? " (cuda error)" : "");
    return t == (unsigned long long)n;
}

int main() {
    int bad = 0;
    for (int n : {1000, 100000}) for (int steps : {2, 52}) {
        bad += !run<0,0>(n, steps, 256);
        bad += !run<0,1>(n, steps, 256);
        bad += !run<0,2>(n, steps, 256);
        bad += !run<1,0>(n, steps, 256);
        bad += !run<1,2>(n, steps, 256);
    }
    printf("%s\n", bad ? "FAILED" : "ALL OK");
    return bad != 0;
}
