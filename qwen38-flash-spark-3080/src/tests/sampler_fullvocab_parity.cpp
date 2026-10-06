// Host parity for the sampled full-vocabulary kernel.  No CUDA: the old k-round scan is
// `serial_fullvocab_select`; the new one-pass shard top-k + k-way merge is `parallel_fullvocab_select`.
// Both call the helpers in sampler_select.hpp that sampler.cu uses.  Same logits => same shortlist, bit for
// bit, including tied maxima, -inf / NaN / inf, and the empty-pass pad (id 0, logit -inf).
#include "strata/kernels/sampler_select.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void fail(const std::string& msg) {
    std::fprintf(stderr, "FAIL %s\n", msg.c_str());
    ++g_fail;
}

uint32_t fbits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// The pre-parallel sampler kernel: k strict-argmax passes.  A pass that finds no logit strictly above -inf
// records id 0 and logit -inf and does not consume a new index, so later empty passes repeat that pair.
static void serial_fullvocab_select(const float* logits, int n_vocab, const int* hist, int hist_len,
                                    const strata::kernels::SamplerParams& p, int* sel_ids, float* sel_logit) {
    using namespace strata::kernels;
    const int k = sampler_k_of(p, n_vocab);
    int hlen = 0;
    const int* hrow = nullptr;
    if (hist != nullptr && hist_len > 0 && p.penalty_last_n > 0) {
        hlen = p.penalty_last_n < hist_len ? p.penalty_last_n : hist_len;
        if (hlen < 0) hlen = 0;
        hrow = hist + (hist_len - hlen);
    }
    std::vector<char> taken((size_t) n_vocab, 0);
    for (int round = 0; round < k; ++round) {
        float bv = sampler_neg_inf();
        int best = n_vocab;
        for (int v = 0; v < n_vocab; ++v) {
            if (taken[(size_t) v]) continue;
            const int c = hrow != nullptr ? sampler_history_count(hrow, hlen, v) : 0;
            const float s = sampler_apply_penalties(logits[v], c, p);
            if (s > bv) {
                bv = s;
                best = v;
            }
        }
        if (best < n_vocab) {
            sel_ids[round] = best;
            sel_logit[round] = bv;
            taken[(size_t) best] = 1;
        } else {
            sel_ids[round] = 0;
            sel_logit[round] = bv;
        }
    }
}

static_assert(strata::kernels::kSamplerSelectKMax == 64, "Shard arrays follow the kernel shortlist width");

struct Shard {
    float vs[strata::kernels::kSamplerSelectKMax];
    int ids[strata::kernels::kSamplerSelectKMax];
    int n = 0;
    int cursor = 0;
};

static void merge_shards(std::vector<Shard>& lists, int k, int n_vocab, Shard& out) {
    using strata::kernels::sampler_better;
    using strata::kernels::sampler_neg_inf;
    out.n = 0;
    out.cursor = 0;
    for (int r = 0; r < k; ++r) {
        float bv = sampler_neg_inf();
        int bi = n_vocab;
        int src = -1;
        for (int t = 0; t < (int) lists.size(); ++t) {
            Shard& s = lists[(size_t) t];
            if (s.cursor >= s.n) continue;
            const float v = s.vs[s.cursor];
            const int i = s.ids[s.cursor];
            if (sampler_better(v, i, bv, bi)) {
                bv = v;
                bi = i;
                src = t;
            }
        }
        if (!(bv > sampler_neg_inf())) break;
        out.vs[out.n] = bv;
        out.ids[out.n] = bi;
        ++out.n;
        if (src >= 0) ++lists[(size_t) src].cursor;
    }
}

// One grid-stride pass, same index map as sampler_partial_kernel: v = block * threads + tid, step threads * blocks.
static void parallel_fullvocab_select(const float* logits, int n_vocab, const int* hist, int hist_len,
                                      const strata::kernels::SamplerParams& p, int blocks, int* sel_ids,
                                      float* sel_logit) {
    using namespace strata::kernels;
    const int threads = kSamplerSelectThreads;
    if (blocks < 1) blocks = 1;
    const int k = sampler_k_of(p, n_vocab);
    int hlen = 0;
    const int* hrow = nullptr;
    if (hist != nullptr && hist_len > 0 && p.penalty_last_n > 0) {
        hlen = p.penalty_last_n < hist_len ? p.penalty_last_n : hist_len;
        if (hlen < 0) hlen = 0;
        hrow = hist + (hist_len - hlen);
    }
    std::vector<Shard> block_lists((size_t) blocks);
    std::vector<Shard> thread_lists((size_t) threads);
    for (int b = 0; b < blocks; ++b) {
        for (auto& s : thread_lists) {
            s.n = 0;
            s.cursor = 0;
        }
        for (int tid = 0; tid < threads; ++tid) {
            Shard& mine = thread_lists[(size_t) tid];
            for (int v = b * threads + tid; v < n_vocab; v += threads * blocks) {
                const int c = hrow != nullptr ? sampler_history_count(hrow, hlen, v) : 0;
                const float s = sampler_apply_penalties(logits[v], c, p);
                sampler_consider(mine.vs, mine.ids, mine.n, k, s, v);
            }
        }
        merge_shards(thread_lists, k, n_vocab, block_lists[(size_t) b]);
    }
    for (auto& s : block_lists) s.cursor = 0;
    Shard merged;
    merge_shards(block_lists, k, n_vocab, merged);
    for (int i = 0; i < merged.n; ++i) {
        sel_ids[i] = merged.ids[i];
        sel_logit[i] = merged.vs[i];
    }
    for (int i = merged.n; i < k; ++i) {
        sel_ids[i] = 0;
        sel_logit[i] = sampler_neg_inf();
    }
}

// Independent transcription of the old kernel's tail (top_p, min_p, temperature, one uniform draw).
static int serial_finish_pick(const int* sel_ids, const float* sel_logit, int k, float top_p, float min_p,
                              int min_keep, float inv_t, float u) {
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, sel_logit[i]);
    if (top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) top_p) {
                cut = i + 1;
                break;
            }
        }
        if (cut < min_keep) cut = min_keep < k ? min_keep : k;
        n_keep = cut;
    }
    if (min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) {
                n_keep = i;
                break;
            }
    }
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, sel_logit[i] * inv_t);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (sel_logit[i] * inv_t) - (double) smx);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) (sel_logit[i] * inv_t) - (double) smx) / sum;
        if ((double) u < cum) {
            pick = sel_ids[i];
            break;
        }
    }
    return pick;
}

static void expect_sel(const char* name, const std::vector<float>& logits, const strata::kernels::SamplerParams& p,
                       const std::vector<int>& hist, int hist_len, int blocks) {
    using namespace strata::kernels;
    const int n_vocab = (int) logits.size();
    const int k = sampler_k_of(p, n_vocab);
    std::vector<int> a((size_t) k), b((size_t) k);
    std::vector<float> av((size_t) k), bv((size_t) k);
    const int* h = hist.empty() ? nullptr : hist.data();
    serial_fullvocab_select(logits.data(), n_vocab, h, hist_len, p, a.data(), av.data());
    parallel_fullvocab_select(logits.data(), n_vocab, h, hist_len, p, blocks, b.data(), bv.data());
    for (int i = 0; i < k; ++i) {
        if (a[(size_t) i] != b[(size_t) i] || fbits(av[(size_t) i]) != fbits(bv[(size_t) i])) {
            char buf[240];
            std::snprintf(buf, sizeof buf, "%s blocks=%d round %d serial id %d (%08x) parallel id %d (%08x)", name,
                          blocks, i, a[(size_t) i], fbits(av[(size_t) i]), b[(size_t) i], fbits(bv[(size_t) i]));
            fail(buf);
            return;
        }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    const float u = 0.17f;
    const int ps = serial_finish_pick(a.data(), av.data(), k, p.top_p, p.min_p, p.min_keep, inv_t, u);
    const int href = sampler_finish_pick(a.data(), av.data(), k, p.top_p, p.min_p, p.min_keep, inv_t, u);
    const int pp = sampler_finish_pick(b.data(), bv.data(), k, p.top_p, p.min_p, p.min_keep, inv_t, u);
    if (ps != href || ps != pp) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s pick serial %d header %d parallel %d", name, ps, href, pp);
        fail(buf);
    }
}

}  // namespace

int main() {
    using namespace strata::kernels;
    const float ninf = sampler_neg_inf();
    const float pinf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float fmax = std::numeric_limits<float>::max();
    const float fmin = -std::numeric_limits<float>::max();

    if (sampler_select_blocks(248320) != 256 || sampler_select_blocks(151936) != 256 ||
        sampler_select_blocks(512) != 2 || sampler_select_blocks(1) != 1) {
        fail("sampler_select_blocks");
    }
    auto per_thread = [](int n_vocab) {
        const int blocks = sampler_select_blocks(n_vocab);
        const int stride = kSamplerSelectThreads * blocks;
        return (n_vocab + stride - 1) / stride;
    };
    if (per_thread(248320) > kSamplerSelectThreadKeep || per_thread(151936) > kSamplerSelectThreadKeep ||
        per_thread(512) > kSamplerSelectThreadKeep) {
        fail("per-thread keep");
    }
    SamplerParams kw;
    kw.top_k = 0;
    if (sampler_k_of(kw, 1000) != 64) fail("top_k 0");
    kw.top_k = 64;
    if (sampler_k_of(kw, 1000) != 64) fail("top_k 64");
    kw.top_k = 100;
    if (sampler_k_of(kw, 1000) != 64) fail("top_k 100");
    kw.top_k = -3;
    if (sampler_k_of(kw, 1000) != 64) fail("top_k -3");
    kw.top_k = 20;
    if (sampler_k_of(kw, 1000) != 20 || sampler_k_of(kw, 10) != 10) fail("top_k clamp");

    {
        SamplerParams p;
        p.top_k = 5;
        p.top_p = 1.0f;
        p.temperature = 1.0f;
        const int blocks = sampler_select_blocks(6);
        expect_sel("tie and nan", {1.f, 3.f, 3.f, 2.f, ninf, nan}, p, {}, 0, blocks);
    }
    {
        SamplerParams p;
        p.top_k = 3;
        p.top_p = 1.0f;
        p.temperature = 1.0f;
        expect_sel("all -inf", {ninf, ninf, ninf, ninf}, p, {}, 0, 1);
        expect_sel("leader then -inf", {pinf, ninf, ninf}, p, {}, 0, 1);
        std::vector<float> infs(8, fmin);
        infs[1] = pinf;
        infs[4] = pinf;
        infs[0] = 0.5f;
        p.top_k = 3;
        expect_sel("+inf tie", infs, p, {}, 0, sampler_select_blocks((int) infs.size()));
        std::vector<float> z(6, -1.f);
        z[0] = -0.0f;
        z[3] = 0.0f;
        p.top_k = 2;
        expect_sel("negative zero tie", z, p, {}, 0, 1);
        std::vector<float> ext(12, -4.f);
        ext[8] = fmax;
        ext[2] = pinf;
        ext[1] = pinf;
        ext[7] = nan;
        ext[9] = ninf;
        p.top_k = 4;
        expect_sel("extremes", ext, p, {}, 0, sampler_select_blocks(12));
    }
    {
        SamplerParams p;
        p.top_k = 3;
        p.top_p = 0.5f;
        p.min_p = 0.05f;
        p.min_keep = 1;
        p.temperature = 0.7f;
        p.penalty_last_n = 4;
        p.penalty_repeat = 1.2f;
        p.penalty_freq = 0.1f;
        p.penalty_present = 0.5f;
        std::vector<float> logits = {1.f, 1.f, 1.f, -2.f, 0.4f, 0.4f, ninf, 0.2f};
        std::vector<int> hist = {-1, 99, 0, 0, 3, 5};
        expect_sel("penalties", logits, p, hist, (int) hist.size(), 1);
        expect_sel("penalties prod blocks", logits, p, hist, (int) hist.size(), sampler_select_blocks(8));
    }
    {
        SamplerParams p;
        p.top_k = 2;
        p.top_p = 1.0f;
        p.temperature = 1.0f;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-2.f, 2.f);
        std::vector<float> logits(1000);
        for (auto& v : logits) v = dist(rng);
        for (int b : {1, 3, 16, 256}) expect_sel("truncate partition", logits, p, {}, 0, b);
    }
    {
        SamplerParams p;
        p.top_k = 20;
        p.top_p = 0.9f;
        p.min_p = 0.02f;
        p.min_keep = 2;
        p.temperature = 0.8f;
        p.seed = 11;
        p.counter = 100;
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.f, 1.f);
        std::vector<float> logits(248320);
        for (auto& v : logits) v = g(rng);
        logits[0] = 4.f;
        logits[100] = 4.f;
        logits[248319] = 4.f;
        logits[50] = ninf;
        logits[51] = nan;
        logits[80] = pinf;
        logits[81] = pinf;
        logits[90] = fmax;
        logits[91] = fmin;
        const int blocks = sampler_select_blocks((int) logits.size());
        expect_sel("vocab 248320", logits, p, {}, 0, blocks);
        p.top_k = 64;
        std::vector<float> mid(4096);
        for (auto& v : mid) v = g(rng);
        mid[1] = mid[4000] = 3.5f;
        mid[2] = ninf;
        mid[3] = nan;
        for (int b : {1, sampler_select_blocks(4096), 256}) expect_sel("k=64 partitions", mid, p, {}, 0, b);
    }
    {
        // Two rows, same contract the kernel applies per token (offset t * n_vocab).  Different block counts
        // on the second row must not change the shortlist.
        SamplerParams p;
        p.top_k = 4;
        p.top_p = 1.0f;
        p.temperature = 1.1f;
        std::vector<float> row0 = {0.2f, 0.2f, -3.f, 1.f, ninf};
        std::vector<float> row1 = {9.f, nan, 9.f, fmin, pinf, -0.0f};
        expect_sel("row0", row0, p, {}, 0, sampler_select_blocks(5));
        expect_sel("row1", row1, p, {}, 0, 1);
        expect_sel("row1 other grid", row1, p, {}, 0, 7);
    }

    if (g_fail) {
        std::fprintf(stderr, "sampler_fullvocab_parity: FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("sampler_fullvocab_parity: PASS\n");
    return 0;
}
