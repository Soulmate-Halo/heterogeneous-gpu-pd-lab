// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's order.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose
// default is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } (`common/common.h`
// at 3cf03257) - ONE penalties stage, first, and TEMPERATURE AFTER THE TRUNCATION FILTERS.  (Issue #53: this file
// used to apply the penalties a second time after the temperature, and min_p before top_p - both taken from the
// order of the `case` labels in `common/sampling.cpp`, which is not the order the chain runs.)  Every order
// produces a valid token, so only a comparison at the distribution level can tell them apart; the parity test
// does that against an independently computed distribution.
//
// `sampler_greedy_kernel` is one block per token over the vocabulary.  The sampled path used to be the same
// shape (k full-vocabulary reductions inside one block).  It is now ONE grid-stride pass: each thread keeps a
// top-k under the serial total order, each block merges its threads, and a second kernel merges the block
// lists and runs the unchanged top_p / min_p / temperature / Philox tail.  See `sampler_select.hpp`.
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/sampler_select.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    return sampler_history_count(h, n, v);
}

__device__ __forceinline__ float apply_penalties(float logit, int count, const SamplerParams& p) {
    return sampler_apply_penalties(logit, count, p);
}

/// **THE GREEDY ARGMAX, ONE BLOCK PER TOKEN, COVERING THE VOCABULARY.**
///
/// **WHY THIS IS A SEPARATE KERNEL AND NOT A BRANCH.**  `sampler_kernel` is launched as a grid over TOKENS
/// with 64 threads and a `if (t >= n_tokens) return;` at the top.  The decode path has `n_tokens == 1`, so
/// that launch was `<<<1, 64>>>`, 63 threads exited on the first line, and ONE THREAD walked all 248,320
/// logits in a dependent loop on one SM of 48.  Measured in isolation (`bench/micro/sampler_cost.cu`):
/// **3.11 ms per token**, 5.7% of an ~54 ms token, and the whole of round 309's `sample` phase - the two
/// synchronisations around it are 0.03 ms each.
///
/// The obvious repair is to parallelise the scan inside `sampler_kernel`, and it is WRONG: with one thread
/// per token, a block reduction over the vocabulary has nothing to reduce, and the threads that returned
/// early are not there for `__syncthreads` or `__shfl_down_sync`.  The first attempt did exactly that and
/// produced the token `5120` thirty-two times.  The grid has to be over tokens with the BLOCK over the
/// vocabulary, which is a different launch configuration and therefore a different kernel.
///
/// **THE TIE RULE IS UNCHANGED AND THAT IS THE WHOLE CORRECTNESS ARGUMENT.**  The serial scan walked `v`
/// ascending with `if (s > bv)`, so the LOWEST index wins a tie.  Each thread keeps that rule over its own
/// strided subset and the reduction resolves two candidates by taking the larger value and, on equality, the
/// SMALLER index - the same total order, so `sampler_parity` and C1 see no change.
__global__ void sampler_greedy_kernel(const float* __restrict__ logits, int n_vocab,
                                      const int* __restrict__ history, int history_len, const SamplerParams p,
                                      int pmin, int plen, int* __restrict__ out) {
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    (void) pmin;
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // PENALTY MEMBERSHIP AS A BITMAP.  The history touches at most `hlen` tokens of a quarter-million
    // vocabulary, but the naive `history_count` per candidate per argmax round costs O(k x n_vocab x hlen)
    // integer compares (~318 M per token at k=20, hlen=64 - measured 45 -> 31 tok/s on a real workload).
    // A shared bitmap gives an O(1) membership test, and only the (at most hlen) hits pay the count scan;
    // the counts - and therefore every sampled value - are exactly what the per-candidate scan produced.
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // The gate needs a NON-EMPTY WINDOW (`hlen > 0`): the launch sizes the shared bitmap only when penalties
    // are on, so a caller handing over a history buffer with `penalty_last_n == 0` must not touch it.
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        const float s = apply_penalties(l[v], hit_count(v), p);
        if (s > bv) { bv = s; best = v; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) ((blockDim.x + 31) >> 5);
        float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        int wi = lane < nw ? si[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
        }
        // A tie between two `-inf` candidates leaves `wi == n_vocab`, and the serial version answered 0.
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

// Scratch for ONE in-flight sampled launch (a chunk of at most kSamplerSelectMaxTokens rows).  generate reads
// the token back on this stream before the next sample; verify syncs the stream; a null stream syncs below.
// Two non-greedy sample_tokens overlapping on different streams would race here.  Nothing is allocated at
// launch, so a captured graph does not see cudaMalloc.
__device__ int g_part_n[kSamplerSelectMaxTokens * kSamplerSelectMaxBlocks];
__device__ int g_part_i[kSamplerSelectMaxTokens * kSamplerSelectMaxBlocks * kSamplerSelectKMax];
__device__ float g_part_v[kSamplerSelectMaxTokens * kSamplerSelectMaxBlocks * kSamplerSelectKMax];

// k-way merge of per-thread best-first lists.  Pairwise `sampler_better` is a total order on distinct ids, so
// the warp/block tournament picks the same head the serial scan would.  A list entry exists only when its
// score is strictly above -inf; when every head is empty the winner stays (-inf, n_vocab) and the merge stops.
// Block size is a multiple of 32 (the launch uses 256).  Every thread calls this, so the syncthreads are uniform.
__device__ void sampler_merge_heads(float* vs, int* ids, int n, int k, int n_vocab, float* red_v, int* red_i,
                                    float* win_v, int* win_i, int* out_i, float* out_v, int* out_n) {
    const int lane = (int) (threadIdx.x & 31);
    const int warp = (int) (threadIdx.x >> 5);
    if (threadIdx.x == 0) *out_n = 0;
    __syncthreads();
    int cursor = 0;
    int alive = 1;
    for (int r = 0; r < k; ++r) {
        float bv = sampler_neg_inf();
        int bi = n_vocab;
        if (cursor < n) {
            bv = vs[cursor];
            bi = ids[cursor];
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
            if (sampler_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            red_v[warp] = bv;
            red_i[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            const int nw = (int) (blockDim.x >> 5);
            float wv = lane < nw ? red_v[lane] : sampler_neg_inf();
            int wi = lane < nw ? red_i[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
                const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
                if (sampler_better(ov, oi, wv, wi)) {
                    wv = ov;
                    wi = oi;
                }
            }
            if (lane == 0) {
                *win_v = wv;
                *win_i = wi;
            }
        }
        __syncthreads();
        // Uniform: every thread sees the same broadcast winner, and every thread hits both barriers.
        if (!(*win_v > sampler_neg_inf())) alive = 0;
        if (alive && threadIdx.x == 0) {
            out_i[*out_n] = *win_i;
            out_v[*out_n] = *win_v;
            ++(*out_n);
        }
        if (alive && cursor < n && ids[cursor] == *win_i) ++cursor;
        __syncthreads();
    }
}

/// **SAMPLED PATH, PASS 1.**  Grid (blocks, tokens-in-chunk), 256 threads.  Each thread walks a grid-stride
/// slice once and keeps its top-k; the block then merges to at most k real winners.  Replaces the old
/// `<<<n_tokens, 1024>>>` kernel whose k rounds re-read the whole vocabulary on one block (decode, n_tokens
/// == 1, was `<<<1, 1024>>>`, about 1.05 ms of the 21.96 ms token on a 3080).  One pass moves ~n_vocab floats
/// instead of k of them, across enough blocks to fill the SMs.
__global__ void sampler_partial_kernel(const float* __restrict__ logits, int n_vocab,
                                       const int* __restrict__ history, int history_len, const SamplerParams p) {
    const int shard = (int) blockIdx.x;
    const int t = (int) blockIdx.y;
    const float* l = logits + (size_t) t * (size_t) n_vocab;
    const int k = sampler_k_of(p, n_vocab);

    const int* hrow = history ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // The launch only uses this kernel when each thread visits at most kSamplerSelectThreadKeep ids, so the
    // local top-k fits in a few hundred bytes of stack (a 64-wide list would blow the default 1 KB stack).
    float vs[kSamplerSelectThreadKeep];
    int ids[kSamplerSelectThreadKeep];
    int n = 0;
    const int stride = (int) blockDim.x * (int) gridDim.x;
    // visits <= ThreadKeep on this launch, so min(k, ThreadKeep) keeps the thread's whole top-k and stays in range.
    const int cap = k < kSamplerSelectThreadKeep ? k : kSamplerSelectThreadKeep;
    for (int v = shard * (int) blockDim.x + (int) threadIdx.x; v < n_vocab; v += stride) {
        const float s = apply_penalties(l[v], hit_count(v), p);
        sampler_consider(vs, ids, n, cap, s, v);
    }

    __shared__ float red_v[32];
    __shared__ int red_i[32];
    __shared__ float win_v;
    __shared__ int win_i;
    __shared__ int bout_i[64];
    __shared__ float bout_v[64];
    __shared__ int produced;
    sampler_merge_heads(vs, ids, n, k, n_vocab, red_v, red_i, &win_v, &win_i, bout_i, bout_v, &produced);
    if (threadIdx.x == 0) {
        const int slot = t * kSamplerSelectMaxBlocks + shard;
        g_part_n[slot] = produced;
        int* di = g_part_i + (size_t) slot * (size_t) kSamplerSelectKMax;
        float* dv = g_part_v + (size_t) slot * (size_t) kSamplerSelectKMax;
        for (int i = 0; i < produced; ++i) {
            di[i] = bout_i[i];
            dv[i] = bout_v[i];
        }
    }
}

/// **SAMPLED PATH, PASS 2.**  One block per token in the chunk.  Thread b holds block b's sorted list; the
/// same merge yields the global top-k.  Slots the serial scan could not fill (no remaining logit strictly
/// above -inf) are padded with id 0 and logit -inf, including duplicates of 0 — that is what an empty pass
/// wrote.  Thread 0 then runs the old top_p / min_p / temperature / Philox chain (`sampler_finish_pick`).
__global__ void sampler_finalize_kernel(int n_vocab, const SamplerParams p, int* __restrict__ out,
                                        int token_base, int nblocks) {
    const int t = (int) blockIdx.x;
    const int k = sampler_k_of(p, n_vocab);
    // Point straight at the block list in scratch.  A thread with no shard keeps a dummy so the pointer is
    // never null; n == 0 means the merge does not read it.
    float sink_v = 0.f;
    int sink_i = 0;
    float* sv = &sink_v;
    int* si = &sink_i;
    int n = 0;
    if ((int) threadIdx.x < nblocks) {
        const int slot = t * kSamplerSelectMaxBlocks + (int) threadIdx.x;
        n = g_part_n[slot];
        si = g_part_i + (size_t) slot * (size_t) kSamplerSelectKMax;
        sv = g_part_v + (size_t) slot * (size_t) kSamplerSelectKMax;
    }
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    __shared__ float win_v;
    __shared__ int win_i;
    __shared__ int bout_i[64];
    __shared__ float bout_v[64];
    __shared__ int produced;
    sampler_merge_heads(sv, si, n, k, n_vocab, red_v, red_i, &win_v, &win_i, bout_i, bout_v, &produced);
    if (threadIdx.x == 0) {
        int m = produced;
        const float ninf = sampler_neg_inf();
        for (int i = m; i < k; ++i) {
            bout_i[i] = 0;
            bout_v[i] = ninf;
        }
        const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) token_base + (uint64_t) t);
        out[t] = sampler_finish_pick(bout_i, bout_v, k, p.top_p, p.min_p, p.min_keep, inv_t, u);
    }
}

// Vocabularies so large that a thread would see more than kSamplerSelectThreadKeep ids.  Same k strict-argmax
// passes as the pre-parallel kernel (one block, 1024 threads).  Decode vocabs (~1.5e5..2.5e5) take the
// one-pass path instead; this exists so a wider vocab cannot overflow the 16-wide local list.
__global__ void sampler_kround_kernel(const float* __restrict__ logits, int n_vocab,
                                      const int* __restrict__ history, int history_len, const SamplerParams p,
                                      int* __restrict__ out) {
    const int t = (int) blockIdx.x;
    const float* l = logits + (size_t) t * (size_t) n_vocab;
    const int k = sampler_k_of(p, n_vocab);
    const int* hrow = history ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;
    }
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };
    __shared__ int sel_ids[64];
    __shared__ float sel_logit[64];
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int i = 0; i < k; ++i) {
        float bv = sampler_neg_inf();
        int best = n_vocab;
        for (int v = (int) threadIdx.x; v < n_vocab; v += (int) blockDim.x) {
            bool taken = false;
            for (int j = 0; j < i; ++j)
                if (sel_ids[j] == v) {
                    taken = true;
                    break;
                }
            if (taken) continue;
            const float s = apply_penalties(l[v], hit_count(v), p);
            if (s > bv) {
                bv = s;
                best = v;
            }
        }
        const int lane = (int) (threadIdx.x & 31);
        const int warp = (int) (threadIdx.x >> 5);
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
            if (sampler_better(ov, oi, bv, best)) {
                bv = ov;
                best = oi;
            }
        }
        if (lane == 0) {
            sv[warp] = bv;
            si[warp] = best;
        }
        __syncthreads();
        if (warp == 0) {
            const int nw = (int) (blockDim.x >> 5);
            float wv = lane < nw ? sv[lane] : sampler_neg_inf();
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
                const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
                if (sampler_better(ov, oi, wv, wi)) {
                    wv = ov;
                    wi = oi;
                }
            }
            if (lane == 0) {
                sel_ids[i] = wi < n_vocab ? wi : 0;
                sel_logit[i] = wv;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        out[t] = sampler_finish_pick(sel_ids, sel_logit, k, p.top_p, p.min_p, p.min_keep, inv_t, u);
    }
}

}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    const unsigned shmem = (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
                               ? (unsigned) ((n_vocab + 31) / 32) * sizeof(unsigned)   // the penalty bitmap
                               : 0;
    if (p.greedy || p.temperature <= 0.0f) {
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        const int gthreads = 1024;
        sampler_greedy_kernel<<<(unsigned) n_tokens, gthreads, shmem, (cudaStream_t) stream>>>(
            logits, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n, out);
    } else {
        // Grid is a function of n_vocab, n_tokens and the compile-time caps only — legal to capture.
        // Chunks reuse the device scratch in stream order; the host interface (launch, then the caller's
        // 4-byte readback on `stream`) is unchanged.  No cudaDeviceSynchronize when stream != nullptr.
        static_assert(kSamplerSelectMaxBlocks <= kSamplerSelectThreads, "one partial list per finalize thread");
        static_assert(kSamplerSelectKMax == 64, "shortlist width");
        static_assert(kSamplerSelectThreadKeep == 16, "partial kernel local list");
        const int threads = kSamplerSelectThreads;
        const int blocks = sampler_select_blocks(n_vocab);
        const int stride = threads * blocks;
        const int per = (n_vocab + stride - 1) / stride;
        if (per <= kSamplerSelectThreadKeep) {
            for (int base = 0; base < n_tokens; base += kSamplerSelectMaxTokens) {
                const int nt = (n_tokens - base) < kSamplerSelectMaxTokens ? (n_tokens - base)
                                                                           : kSamplerSelectMaxTokens;
                const float* row = logits + (size_t) base * (size_t) n_vocab;
                const int* hrow = history != nullptr ? history + (size_t) base * (size_t) history_len : nullptr;
                sampler_partial_kernel<<<dim3((unsigned) blocks, (unsigned) nt), (unsigned) threads, shmem,
                                         (cudaStream_t) stream>>>(row, n_vocab, hrow, history_len, p);
                sampler_finalize_kernel<<<(unsigned) nt, (unsigned) threads, 0, (cudaStream_t) stream>>>(
                    n_vocab, p, out + base, base, blocks);
            }
        } else {
            sampler_kround_kernel<<<(unsigned) n_tokens, 1024, shmem, (cudaStream_t) stream>>>(
                logits, n_vocab, history, history_len, p, out);
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "sample_tokens launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
