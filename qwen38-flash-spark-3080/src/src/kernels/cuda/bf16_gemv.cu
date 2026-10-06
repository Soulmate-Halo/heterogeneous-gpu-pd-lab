// src/kernels/cuda/bf16_gemv.cu - the BF16 GEMV.  See the header for why it is not `s_gemv`.
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

/// One thread per output row, walking it contiguously.  Kept as the naive reference the split version is
/// checked against, and used directly for the shapes where the output width is already large.
__global__ void bf16_gemv_naive_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                       float* __restrict__ y, long long n_in, long long n_out) {
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = 0; i < n_in; ++i)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    y[o] = acc;
}

/// One WARP per output row, lanes striding the reduction axis.  For a fixed `i` consecutive lanes touch
/// consecutive addresses in this layout, so the load is coalesced - the property that took `gr_read` from
/// 262 ms/token to 8.9.
__global__ void bf16_gemv_warp_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                      float* __restrict__ y, long long n_in, long long n_out) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long o = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (o >= n_out) return;
    const int lane = threadIdx.x & 31;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = lane; i < n_in; i += 32)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[o] = acc;
}

/// TPR lanes cooperate on ONE row's reduction: `threads_per_row` threads each take a strided slice and the
/// block reduces through shared memory.  Used when the output width is too small to fill the machine.
__global__ void bf16_gemv_split_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                       float* __restrict__ y, long long n_in, long long n_out, int tpr) {
    extern __shared__ float scratch[];
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int t = threadIdx.x;                 // 0 .. tpr-1
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = t; i < n_in; i += tpr)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    // block reduction; `tpr` is at most a few hundred, so a tree in shared is enough
    scratch[t] = acc;
    __syncthreads();
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        __syncthreads();
    }
    if (t == 0) y[o] = scratch[0];
}

inline void finish(void* stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream != nullptr) return;
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(s));
        std::exit(1);
    }
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // **THIS USED TO ALWAYS USE THE NAIVE KERNEL, AND THE NAIVE KERNEL IS UNCOALESCED.**
    //
    // `bf16_gemv_naive_kernel` gives one THREAD per output row and walks the row contiguously, so at a fixed
    // `i` the 32 threads of a warp read `w[(o+k)*n_in + i]` - addresses `n_in * 2` bytes apart.  Every load in
    // the kernel's inner loop costs 32 transactions instead of 1.  The comment above that kernel claims it is
    // "used directly for the shapes where the output width is already large", but nothing ever made that
    // choice: this function called it unconditionally, at every shape.
    //
    // What that costs, from the nsys per-kernel table: **4.66 ms/token across 24 calls, 194 us each** - the
    // third largest kernel in the engine at the time, moving 1.19 GiB/token at ~210 GB/s where a plain read at
    // this geometry measures 641 GB/s (`bench/micro/p32_floor.cu`).
    //
    // `bf16_gemv_warp_kernel` exists, is checked against this one by `bf16_gemv_parity`, and is coalesced -
    // consecutive lanes touch consecutive addresses - which is the property `gr_read`'s comment records as
    // having taken it from 262 ms/token to 8.9.  The threshold below is where warp-per-row has enough rows to
    // fill the machine: 8 warps per block, so 64 rows is 8 blocks and 128 is 16, against 48 SMs.
    if (n_out >= 64) {
        const int warps = THREADS / 32;
        const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
        bf16_gemv_warp_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(x, w, y, n_in, n_out);
        finish(stream, "bf16_gemv(warp)");
        return;
    }
    // Below the threshold the warp kernel would leave most of the machine idle, and the naive one is at least
    // not wasting warps.  It is still uncoalesced, so this is the branch to revisit if a small-output caller
    // ever shows up hot in the profile.
    const unsigned grid = (unsigned) ((n_out + THREADS - 1) / THREADS);
    bf16_gemv_naive_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(x, w, y, n_in, n_out);
    finish(stream, "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // WARP-PER-ROW when the caller asks for 32, because that path needs no shared memory and no barrier - and
    // with 48 output rows it is the configuration that fills the machine.  A `tpr` that is not 32 and not a
    // power of two is refused rather than quietly rounded, because the tree reduction below needs one.
    if (threads_per_row == 32) {
        const int warps = THREADS / 32;
        const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
        bf16_gemv_warp_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(x, w, y, n_in, n_out);
        finish(stream, "bf16_gemv_split(warp)");
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                             "warp-per-row path)\n", threads_per_row);
        std::exit(1);
    }
    const unsigned grid = (unsigned) n_out;
    bf16_gemv_split_kernel<<<grid, threads_per_row, (size_t) threads_per_row * sizeof(float),
                             (cudaStream_t) stream>>>(x, w, y, n_in, n_out, threads_per_row);
    finish(stream, "bf16_gemv_split");
}

}  // namespace strata::kernels
