// QuantCUDA Risk Engine - shared device helpers for all CUDA kernels.
//
// Design rules (from the project specification):
//   * no virtual dispatch on the device - one specialized kernel per
//     instrument type,
//   * fused simulation + payoff + block-level reduction in the optimized
//     generations, with every earlier generation preserved for benchmarks,
//   * template on simulation real type / accumulator type to support the
//     float / double / mixed-precision study.
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>

namespace quantcuda::gpu::dev {

// ---------------------------------------------------------------------------
// Instrument parameters passed to kernels BY VALUE (fast constant-cache path)
// ---------------------------------------------------------------------------
struct OptionParams {
    double spot;
    double strike;
    double rate;
    double dividend;
    double volatility;
    double maturity;
    double barrier;
    int is_call;      // 1 call, 0 put
    int barrier_dir;  // 0 none, +1 up-and-out, -1 down-and-out
    int time_steps;
};

// SoA batch layout (V1+): adjacent threads read adjacent fields.
struct DeviceOptionBatch {
    double* spots;
    double* strikes;
    double* rates;
    double* dividends;
    double* volatilities;
    double* maturities;
    double* barriers;
    int* is_call;
    int* barrier_dir;
    int* time_steps;
    int count;
};

// AoS layout retained ONLY for the V0 naïve baseline benchmark.
struct OptionAoS {
    double spot, strike, rate, dividend, volatility, maturity, barrier;
    int is_call, barrier_dir, time_steps;
};

// Mergeable streaming moments used by pricing and Greeks kernels.  Keeping M2
// (the sum of squared deviations from the running mean) avoids forming the
// cancellation-prone E[X^2] - E[X]^2 expression.  A thread updates its local
// moments with Welford's recurrence; blocks and the host combine them with
// Chan's parallel recurrence.
template <typename T>
struct RunningMoments {
    unsigned long long count = 0;
    T mean = T(0);
    T m2 = T(0);
};

// Type-erased block result copied to the host.  Float-accumulation kernels
// deliberately do their arithmetic in float, then widen the final partial.
struct MomentPartial {
    unsigned long long count = 0;
    double mean = 0.0;
    double m2 = 0.0;
};

template <typename T>
__host__ __device__ inline void moments_add(RunningMoments<T>& moments, T value) {
    const unsigned long long next_count = moments.count + 1;
    const T delta = value - moments.mean;
    moments.mean += delta / static_cast<T>(next_count);
    const T delta2 = value - moments.mean;
    moments.m2 += delta * delta2;
    moments.count = next_count;
}

template <typename T>
__host__ __device__ inline RunningMoments<T> moments_merge(
    RunningMoments<T> lhs, const RunningMoments<T>& rhs) {
    if (lhs.count == 0) return rhs;
    if (rhs.count == 0) return lhs;
    const unsigned long long combined_count = lhs.count + rhs.count;
    const T lhs_count = static_cast<T>(lhs.count);
    const T rhs_count = static_cast<T>(rhs.count);
    const T combined = static_cast<T>(combined_count);
    const T delta = rhs.mean - lhs.mean;
    lhs.mean += delta * (rhs_count / combined);
    lhs.m2 += rhs.m2 + delta * delta * (lhs_count * (rhs_count / combined));
    lhs.count = combined_count;
    return lhs;
}

template <typename T>
__host__ __device__ inline MomentPartial moments_partial(
    const RunningMoments<T>& moments) {
    return {moments.count, static_cast<double>(moments.mean),
            static_cast<double>(moments.m2)};
}

// ---------------------------------------------------------------------------
// Device math
// ---------------------------------------------------------------------------
__device__ __forceinline__ double norm_cdf_d(double x) {
    return 0.5 * erfc(-x * 0.70710678118654752440);
}

template <typename Real>
__device__ __forceinline__ Real gbm_drift(Real r, Real q, Real sigma, Real dt) {
    return (r - q - Real(0.5) * sigma * sigma) * dt;
}

template <typename Real>
__device__ __forceinline__ Real gbm_diffusion(Real sigma, Real dt) {
    return sigma * sqrt(dt);
}

template <typename Real>
__device__ __forceinline__ Real european_payoff_d(int is_call, Real st, Real k) {
    return is_call ? fmax(st - k, Real(0)) : fmax(k - st, Real(0));
}

// ---------------------------------------------------------------------------
// Reductions - all four generations kept for the comparative benchmark:
//   0: global atomicAdd per thread (V0)
//   1: shared-memory tree reduction
//   2: warp-shuffle reduction
//   3: warp-shuffle + one shared value per warp (default in fused kernels)
// ---------------------------------------------------------------------------
template <typename AccT>
__device__ __forceinline__ AccT warp_reduce(AccT v) {
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int active_lanes = __popc(mask);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        AccT other = __shfl_down_sync(mask, v, offset);
        if (lane + offset < active_lanes) v += other;
    }
    return v;
}

// Shared-memory tree reduction. smem must hold blockDim.x elements. The
// rounded-up active count keeps every element even for internal launches whose
// block size is not a power of two (public pricing APIs require powers of two).
template <typename AccT>
__device__ AccT block_reduce_smem(AccT v, AccT* smem) {
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int active = blockDim.x; active > 1; active = (active + 1) / 2) {
        const int next = (active + 1) / 2;
        if (tid + next < active) smem[tid] += smem[tid + next];
        __syncthreads();
    }
    return smem[0];
}

// Warp shuffle + one shared slot per warp (few shared bytes, few syncs).
template <typename AccT>
__device__ AccT block_reduce_warp(AccT v, AccT* smem /* >= 32 slots */) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : AccT(0);
    if (warp == 0) v = warp_reduce(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

// Moment-specific reductions use Chan merges instead of scalar addition.
template <typename T>
__device__ __forceinline__ RunningMoments<T> warp_reduce_moments(
    RunningMoments<T> value) {
    __syncwarp();
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int active_lanes = __popc(mask);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        RunningMoments<T> other;
        other.count = __shfl_down_sync(mask, value.count, offset);
        other.mean = __shfl_down_sync(mask, value.mean, offset);
        other.m2 = __shfl_down_sync(mask, value.m2, offset);
        if (lane + offset < active_lanes) value = moments_merge(value, other);
    }
    return value;
}

template <typename T>
__device__ RunningMoments<T> block_reduce_moments_smem(
    RunningMoments<T> value, RunningMoments<T>* smem) {
    const int tid = threadIdx.x;
    smem[tid] = value;
    __syncthreads();
    for (int active = blockDim.x; active > 1; active = (active + 1) / 2) {
        const int next = (active + 1) / 2;
        if (tid + next < active)
            smem[tid] = moments_merge(smem[tid], smem[tid + next]);
        __syncthreads();
    }
    return smem[0];
}

template <typename T>
__device__ RunningMoments<T> block_reduce_moments_warp(
    RunningMoments<T> value, RunningMoments<T>* smem /* >= 32 slots */) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_moments(value);
    if (lane == 0) smem[warp] = value;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    value = threadIdx.x < nwarps ? smem[threadIdx.x] : RunningMoments<T>{};
    if (warp == 0) value = warp_reduce_moments(value);
    if (threadIdx.x == 0) smem[0] = value;
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------
__device__ __forceinline__ std::size_t global_thread_id() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ std::size_t grid_stride() {
    return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

inline int grid_for(std::size_t paths, int block_size) {
    std::size_t blocks = (paths + static_cast<std::size_t>(block_size) - 1) /
                         static_cast<std::size_t>(block_size);
    const std::size_t cap = 65535;  // plenty of headroom; grid-stride covers rest
    return static_cast<int>(blocks > cap ? cap : blocks);
}

}  // namespace quantcuda::gpu::dev
