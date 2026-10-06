// src/kernels/cuda/router_top10.cu - P2.S2: the MoE router.
//
// P2.S2's spec: "BF16 GEMV, softmax / top-k / renormalize per docs/semantics.md; emits (expert_id, weight) x 10
// per token".  This is the softmax/top-k/renormalize half; the BF16 GEMV that produces the logits is a
// separate kernel and is NOT here.
//
// The semantics are transcribed from `ref/moe.py::router`, which is itself transcribed from llama-graph.cpp:
//
//     p   = softmax(logits)                       over ALL experts, not over the selected subset
//     ids = stable argsort(-p)[:k]                ties break by INDEX, ascending
//     w   = gather(p, ids)
//     s   = max(sum(w), 2**-14)                   ggml_clamp(..., 2**-14, INF)
//     return ids, w / s
//
// The two-step structure matters and is not cosmetic: computing softmax over the selected subset instead is
// mathematically identical (softmax is shift-invariant) but it moves the renormalisation, and the CLAMP is
// part of the renormalisation.  Reproducing the gather form is what makes the clamp land in the same place.
//
// WHY THIS IS NO LONGER "NAIVE BY DESIGN", AND WHAT THE OLD REASONING GOT WRONG.
//
// The first version was ONE THREAD PER TOKEN: k passes over the experts, each pass recomputing
// `exp((double) l[e] - mx)` for every expert.  Its comment justified that with
//
//     "5,120 operations per token against 2.36e9 weights of expert matvec - three orders of magnitude smaller
//      than the thing it feeds, so the simplest correct version is also fast enough"
//
// **That is a statement about OPERATION COUNT and it says nothing about TIME, because all 5,120 operations were
// on ONE THREAD while the matvec has thousands.**  Measured (round 218, stage-by-stage inside one block):
//
//     moe_layer   4.077 ms      router_top10   3.387 ms      bf16 gemv   0.270 ms
//
// 3.39 ms per layer x 48 layers = 163 ms of a 289 ms token - **56% of the whole forward pass, in a kernel that
// reads 2.6 MB.**  It was invisible in `router_top10_parity` because that test runs the kernel ONCE and checks
// the right ids, and invisible in every per-layer test for the same reason.
//
// WHAT IS PARALLELISED AND WHAT IS DELIBERATELY NOT:
//
//   * the MAX is a tree reduction of `fmaxf`, which is exact and order-independent - bit-identical to the
//     serial scan it replaces;
//   * the 512 EXPONENTIALS are computed ONCE each, in parallel.  The old kernel computed them ELEVEN TIMES
//     (once for the softmax, then again inside every one of the k passes).  This is where the 3.4 ms was;
//   * the DOUBLE SUM is still accumulated on ONE thread in ASCENDING order.  512 double adds is about a
//     microsecond and it keeps the accumulation order - and therefore the last bits - identical to the
//     reference-faithful serial version.  Parallelising it would be a free speedup and a silent change to the
//     values, and the exp was the cost, not this;
//   * the SELECTION is by RANK rather than by k repeated maxima:
//
//         rank(e) = #{ f : p[f] > p[e] }  +  #{ f < e : p[f] == p[e] }
//
//     which IS "stable descending argsort, ties by ascending index" - the same order the repeated-maximum loop
//     produced - computed in ONE parallel pass over the experts instead of k serial ones.  It is O(n^2)
//     comparisons and that is the right trade here: n = 512, every comparison is independent, and the old
//     version was O(k*n) with a serial `exp` inside.
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

/// One BLOCK per token, so the reductions have somewhere to happen.  `n_tokens` is 1 in decode; the grid keeps
/// the batch case working without a second code path.
__global__ void router_top10_kernel(const float* __restrict__ logits, int n_tokens, int n_expert, int k,
                                    int* __restrict__ ids, float* __restrict__ weights) {
    // **`s_taken` IS FIRST SO THE OTHER TWO KEEP THEIR ALIGNMENT WITHOUT AN OFFSET PARAMETER**, and `s_p`'s
    // existing `s_ex + n_expert` stays correct because it is relative to `s_ex`.  One byte per expert.
    extern __shared__ unsigned char s_raw[];
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    unsigned char* s_taken = s_raw;
    double* s_ex = (double*) (s_raw + taken_bytes);
    __shared__ float s_red[RT_MAX_THREADS / 32];
    __shared__ int s_rid[RT_MAX_THREADS / 32];
    __shared__ double s_sum;

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const float* l = logits + (size_t) t * n_expert;

    // ---- softmax over ALL experts, for stability: the max.  A tree of `fmaxf` is EXACT and order-independent,
    // so this is bit-identical to the serial scan.
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += nt) mx = fmaxf(mx, l[e]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    if ((tid & 31) == 0) s_red[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        const int nw = (nt + 31) >> 5;
        float v = (tid < nw) ? s_red[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (tid == 0) s_red[0] = v;
    }
    __syncthreads();
    mx = s_red[0];

    // ---- THE 512 EXPONENTIALS, ONCE EACH AND IN PARALLEL.  `exp` in double is software-emulated on this die
    // and was 5,632 serial calls before; it is 512 parallel ones now.
    for (int e = tid; e < n_expert; e += nt) s_ex[e] = exp((double) l[e] - (double) mx);
    __syncthreads();

    // ---- the sum, ascending, on one thread: see the note above on why this is NOT parallelised.
    if (tid == 0) {
        double sum = 0.0;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    __syncthreads();
    const float inv = (float) (1.0 / s_sum);

    // ---- the selection, by rank.  `p[e] = (float)(s_ex[e] * inv)` reproduces the old expression exactly:
    // `(float)(exp((double) l[e] - (double) mx) * inv)` with `inv` a FLOAT.
    // ---- p[] ONCE.  The rank loop below compares every expert against every other, so computing
    // `(float) (s_ex[f] * inv)` INSIDE it did **512 x 512 = 262,144** double multiplies and float converts per
    // token per layer instead of 512 - and a consumer die runs FP64 at a small fraction of FP32, which is why
    // this was worth more than the branch.  The shared load stays either way; the ARITHMETIC is what goes.
    //
    // THIS IS BIT-IDENTICAL, not merely equivalent: `p[e]` is the same expression it was, computed once instead
    // of 513 times, and the comparisons then see exactly the same floats.
    float* s_p = (float*) (s_ex + n_expert);
    for (int e = tid; e < n_expert; e += nt) s_p[e] = (float) (s_ex[e] * inv);
    __syncthreads();

    // ================================ THE SELECTION, IN k PASSES ================================
    //
    // **THE RANK-BY-COUNTING LOOP COMPARED EVERY EXPERT AGAINST EVERY OTHER**: 512 x 512 = 262,144 shared
    // loads and compares per token per layer, to choose ten things out of 512 - measured at 34.35 us, 63% of
    // `moe_route`'s accounted cost and the largest single item in the R3 plan.  The comment above records that
    // the FP64 ARITHMETIC inside that loop was hoisted out once already ("262,144 double multiplies per token
    // per layer instead of 512"); **the O(n^2) STRUCTURE was left, and it is the structure that costs.**
    //
    // `k` passes of a block-wide argmax is 10 x 512 = 5,120 compares - **51x less work** - and it produces the
    // SAME ORDER.  That is the whole correctness argument: scanning `e` ascending with a strict `>` keeps the
    // LOWEST index on a tie, which is exactly the rule the rank loop spelled out as
    // `else if (f < e && pf == pe) ++rank`.  A stable descending top-k, ties by index.
    for (int e = tid; e < n_expert; e += nt) s_taken[e] = 0;
    __syncthreads();

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;               // a sentinel that loses to every real index
        for (int e = tid; e < n_expert; e += nt) {
            if (s_taken[e]) continue;
            const float pe = s_p[e];
            if (pe > bv) { bv = pe; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { s_red[tid >> 5] = bv; s_rid[tid >> 5] = bi; }
        __syncthreads();
        if (tid < 32) {
            const int nw = (nt + 31) >> 5;
            float v = (tid < nw) ? s_red[tid] : -INFINITY;
            int ix = (tid < nw) ? s_rid[tid] : n_expert;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, v, off);
                const int oi = __shfl_down_sync(0xffffffffu, ix, off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (tid == 0 && ix < n_expert) {
                ids[(size_t) t * k + i] = ix;
                weights[(size_t) t * k + i] = v;
                s_taken[ix] = 1;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // ---- renormalise, with ggml's lower clamp.  Order preserved.
    if (tid == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
        const double sc = fmax(s, 6.103515625e-05);       // 2**-14
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
    }
}

}  // namespace

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full warp, for the reductions
    // the selection's taken-mask, then n_expert doubles for the exponentials, then n_expert floats for the
    // probabilities.  The mask is first so the two aligned arrays need no offset parameter.
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem =
        taken_bytes + (size_t) n_expert * sizeof(double) + (size_t) n_expert * sizeof(float);
    router_top10_kernel<<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
        logits, n_tokens, n_expert, k, ids, weights);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "router_top10 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "router_top10: %s\n", cudaGetErrorString(s));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
