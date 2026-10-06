// include/strata/kernels/sampler_select.hpp - shared top-k order for the sampled kernel.
//
// The sampled kernel used to take k sequential argmax passes over the whole vocabulary.  A value wins a pass
// only when it is STRICTLY greater than the current best (ties keep the lower index; -inf and NaN never win).
// A pass that finds nothing still records id 0 with logit -inf.  The parallel kernel does that in one
// grid-stride pass: each thread keeps its local top-k under the same total order, blocks merge those lists,
// and a second kernel merges the block lists.  Anything not strictly above -inf is dropped and the tail of
// the k-list is padded with (0, -inf), which is what the empty passes used to append.
#pragma once

#include "strata/kernels/sampler.hpp"

#include <cmath>
#include <cstring>

namespace strata::kernels {

#if defined(__CUDACC__)
#define STRATA_HD __host__ __device__
#else
#define STRATA_HD
#endif

// Grid caps.  `sampler_select_blocks` reads only n_vocab plus these constants, so a captured launch's grid
// is fixed for the life of the graph.  Finalize assigns one partial list per thread, hence MaxBlocks <= Threads.
// Unscoped enumerators so device-array bounds stay integral constant expressions on older nvcc.
enum {
    kSamplerSelectThreads = 256,
    kSamplerSelectMaxBlocks = 256,
    kSamplerSelectMaxTokens = 8,   // chunk length; wider batches replay the same scratch
    kSamplerSelectKMax = 64,       // sampled path: 1..63 as given; 0 or >= 64 keep 64
    // Per-thread local list.  With MaxBlocks*Threads stride, a vocab up to stride*ThreadKeep visits at most
    // ThreadKeep ids per thread, so the list never needs to be wider than this (and never wider than k).
    kSamplerSelectThreadKeep = 16
};

STRATA_HD inline int sampler_k_of(const SamplerParams& p, int n_vocab) {
    int k = (p.top_k > 0 && p.top_k < kSamplerSelectKMax) ? p.top_k : kSamplerSelectKMax;
    if (k > n_vocab) k = n_vocab;
    if (k < 0) k = 0;
    return k;
}

inline int sampler_select_blocks(int n_vocab) {
    if (n_vocab < 1) return 1;
    int blocks = (n_vocab + kSamplerSelectThreads - 1) / kSamplerSelectThreads;
    if (blocks > kSamplerSelectMaxBlocks) blocks = kSamplerSelectMaxBlocks;
    if (blocks < 1) blocks = 1;
    return blocks;
}

STRATA_HD inline float sampler_neg_inf() {
#if defined(__CUDA_ARCH__)
    return __int_as_float(0xff800000);
#else
    const unsigned u = 0xff800000u;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
#endif
}

// Total order of a round: larger penalised logit wins; on equality the smaller index wins.
STRATA_HD inline bool sampler_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

STRATA_HD inline int sampler_history_count(const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

// Transcribed from llama_sampler_penalties_apply.  count <= 0 leaves the logit untouched.  The repeat
// penalty multiplies non-positive logits and divides positive ones.  Presence is boolean, not the count.
STRATA_HD inline float sampler_apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

// Insert (s, id) into a best-first list of length n, capacity cap.  Scores that are not strictly greater
// than -inf (including every NaN) are refused: the serial scan's `s > bv` starting from -inf never kept them.
STRATA_HD inline void sampler_consider(float* vs, int* ids, int& n, int cap, float s, int id) {
    if (cap <= 0) return;
    if (!(s > sampler_neg_inf())) return;
    if (n == cap && !sampler_better(s, id, vs[n - 1], ids[n - 1])) return;
    int pos = (n < cap) ? n : cap - 1;
    while (pos > 0 && sampler_better(s, id, vs[pos - 1], ids[pos - 1])) {
        vs[pos] = vs[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
    }
    vs[pos] = s;
    ids[pos] = id;
    if (n < cap) ++n;
}

STRATA_HD inline float sampler_fmax(float a, float b) {
#if defined(__CUDA_ARCH__)
    return fmaxf(a, b);
#else
    return std::fmax(a, b);
#endif
}

STRATA_HD inline float sampler_logf(float x) {
#if defined(__CUDA_ARCH__)
    return logf(x);
#else
    return std::log(x);
#endif
}

STRATA_HD inline double sampler_expd(double x) {
#if defined(__CUDA_ARCH__)
    return exp(x);
#else
    return std::exp(x);
#endif
}

// top_p -> min_p -> temperature -> one uniform draw.  Arithmetic matches the old sampler kernel, including
// the fmaxf sweep (the list is already sorted, so the max is sel_logit[0] unless a pad of -inf is in front,
// which only happens when every entry is the empty-pass pad).
STRATA_HD inline int sampler_finish_pick(const int* sel_ids, const float* sel_logit, int k,
                                         float top_p, float min_p, int min_keep,
                                         float inv_t, float u) {
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = sampler_fmax(mx, sel_logit[i]);
    if (top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += sampler_expd((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += sampler_expd((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) top_p) { cut = i + 1; break; }
        }
        if (cut < min_keep) cut = min_keep < k ? min_keep : k;
        n_keep = cut;
    }
    if (min_p > 0.0f) {
        const float thresh = sel_logit[0] + sampler_logf(min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = sampler_fmax(smx, sel_logit[i] * inv_t);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += sampler_expd((double) (sel_logit[i] * inv_t) - (double) smx);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += sampler_expd((double) (sel_logit[i] * inv_t) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
    return pick;
}

#undef STRATA_HD

}  // namespace strata::kernels
