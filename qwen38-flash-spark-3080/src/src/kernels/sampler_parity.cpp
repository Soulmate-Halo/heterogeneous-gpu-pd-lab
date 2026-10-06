// src/kernels/sampler_parity.cpp - P2.S2's test for the sampler chain.
//
// THE CHECK THAT MATTERS IS THAT THE ORDER IS OBSERVABLE.  A sampler in the wrong order still returns a valid
// token, so "it produced a token" proves nothing; and because temperature is MONOTONIC it does not change which
// tokens `top_k` keeps, so a top-k-only fixture cannot see the order either.  What it changes is `top_p`'s CUT:
// at T < 1 the distribution sharpens, the cumulative mass reaches p sooner, and fewer tokens survive.
//
// So this builds a fixture where that happens, computes the greedy pick under BOTH orders, and requires them to
// DIFFER - then requires the kernel to agree with the specified one.  Without the first half, the test would
// pass against either order.
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// The host reference for the specified order: top_k -> top_p -> temperature -> argmax.
// `temp_first` swaps the first and last stages, which is the intuitive-but-wrong order.
int reference_pick(const std::vector<float>& l, const strata::kernels::SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };

    std::vector<int> ids;
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) { best = v; bv = val(v); }
        }
        taken[(size_t) best] = 1;
        ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = (int) i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        ids.resize((size_t) cut);
    }
    return ids[0];      // greedy: the largest SURVIVING logit, and `ids` is in descending order
}

// The number of survivors AFTER top_p, in the given order - the quantity the order actually changes.
int reference_cut(const std::vector<float>& l, const strata::kernels::SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<int> ids;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) { best = v; bv = val(v); }
        }
        taken[(size_t) best] = 1; ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = (int) i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        return cut;
    }
    return (int) ids.size();
}

int run(const char* name, const std::vector<float>& logits, int n_tokens, const strata::kernels::SamplerParams& p,
        const std::vector<int>& want, const std::vector<int>& hist = {}, int hist_len = 0) {
    float* d_l = nullptr;
    int* d_o = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_o, (size_t) n_tokens * sizeof(int)), "malloc out");
    check(cudaMemcpy(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    // -1 in every output slot first: a row the kernel leaves unwritten can never match (a verify window reads
    // every row, so "no output" is a wrong answer, not a skipped one)
    check(cudaMemset(d_o, 0xFF, (size_t) n_tokens * sizeof(int)), "fill out");
    int* d_h = nullptr;
    if (hist_len > 0) {
        check(cudaMalloc(&d_h, hist.size() * sizeof(int)), "malloc hist");
        check(cudaMemcpy(d_h, hist.data(), hist.size() * sizeof(int), cudaMemcpyHostToDevice), "copy hist");
    }
    strata::kernels::sample_tokens(d_l, n_tokens, (int) (logits.size() / n_tokens), d_h, hist_len, p, d_o,
                                   nullptr);
    std::vector<int> got((size_t) n_tokens);
    check(cudaMemcpy(got.data(), d_o, got.size() * sizeof(int), cudaMemcpyDeviceToHost), "back");
    int bad = 0;
    for (int t = 0; t < n_tokens; ++t) if (got[(size_t) t] != want[(size_t) t]) ++bad;
    std::printf("  %-34s %s (%d of %d differ)", name, bad ? "*** WRONG ***" : "matches", bad, n_tokens);
    if (bad) std::printf("   first: want %d got %d", want[0], got[0]);
    std::printf("\n");
    cudaFree(d_l);
    cudaFree(d_o);
    if (d_h) cudaFree(d_h);
    return bad;
}

// The Philox draw, host side - a transcription of the kernel's `philox_uniform` so the SAMPLED pick (not
// just the greedy argmax) can be pinned against a reference.  `__umulhi(a, b)` is the high half of a 32x32
// multiply, spelled `(uint32_t)(((uint64_t) a * b) >> 32)` here.
struct PhiloxRound {
    uint32_t& c0; uint32_t& c1; uint32_t& c2; uint32_t& c3;
    void step(uint32_t k0, uint32_t k1) const {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
};

float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    PhiloxRound r{c0, c1, c2, c3};
    for (int i = 0; i < 10; ++i) r.step((uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// The full SAMPLED chain, host side - the kernel's `sampler_kernel` in serial form, in llama.cpp's order:
// penalties on the raw logits during the top_k selection (ties to the lowest index), top_p's cut in double over
// the top_k list, the min_p prefix cut on its survivors, the temperature, and one Philox draw at
// (seed, counter + row).  One penalties stage (issue #53: this reference used to repeat the kernel's second one).
int sampled_reference(const std::vector<float>& l, const std::vector<int>& hist,
                      const strata::kernels::SamplerParams& p, int row) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;                       // 1..64 as given; 0 (off) and wider keep the widest list, 64
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
    std::vector<int> sel_ids;
    std::vector<float> sel_logit;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1;
        sel_ids.push_back(best); sel_logit.push_back(bv);
    }
    // llama.cpp's order (issue #53): top_p over the whole top_k list, then min_p on its survivors
    const int n_sel = (int) sel_ids.size();
    int n_keep = n_sel;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < n_sel; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]);
        double cum = 0.0;
        int cut = n_sel;
        for (int i = 0; i < n_sel; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < n_sel ? p.min_keep : n_sel;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return sel_logit[(size_t) i] * inv_t; };   // one penalties stage, before (#53)
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    int pick = sel_ids[(size_t) (n_keep - 1)];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[(size_t) i]; break; }
    }
    return pick;
}

// Survivors after the min_p + top_p cuts, in the sampled chain - the quantity an order or a threshold
// actually changes, used to assert a fixture can SEE the feature before asserting the kernel matches.
int sampled_cut(const std::vector<float>& l, const std::vector<int>& hist, const strata::kernels::SamplerParams& p) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
    std::vector<float> sel;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1; sel.push_back(bv);
    }
    int n_minp = (int) sel.size();
    if (p.min_p > 0.0f) {
        const float thresh = sel[0] + std::log(p.min_p);
        for (int i = 0; i < (int) sel.size(); ++i)
            if (sel[(size_t) i] < thresh) { n_minp = i; break; }
    }
    if (p.top_p >= 1.0f) return n_minp;
    double sum = 0.0;
    for (int i = 0; i < n_minp; ++i) sum += std::exp((double) sel[(size_t) i] - (double) sel[0]);
    double cum = 0.0;
    int cut = n_minp;
    for (int i = 0; i < n_minp; ++i) {
        cum += std::exp((double) sel[(size_t) i] - (double) sel[0]) / sum;
        if (cum >= (double) p.top_p) { cut = i + 1; break; }
    }
    if (cut < p.min_keep) cut = p.min_keep < n_minp ? p.min_keep : n_minp;
    return cut;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: sampler_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    const int NV = 512, NT = 4;

    // ---- fixture 1: plain greedy.  top_k = 0 (disabled), top_p = 1 (disabled), T = 1 -> argmax.
    {
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; p.greedy = true;
        std::mt19937 rng(3); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t) want[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV}, p, false);
        bad += run("greedy argmax", l, NT, p, want);
    }

    // ---- fixture 2: THE ORDER FIXTURE.  T = 0.5 sharpens the distribution enough that top_p = 0.5 cuts
    // differently before and after the scaling, and the two orders then pick DIFFERENT tokens.
    {
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 0.5f; p.temperature = 0.5f;
        p.min_keep = 1; p.greedy = true;
        std::vector<float> l((size_t) NV * NT, -1000.0f);
        for (int t = 0; t < NT; ++t) {
            // a flat-ish head so the cumulative mass crosses 0.5 inside it, and one clear leader
            l[(size_t) t * NV + 0] = 3.0f;
            for (int v = 1; v < 8; ++v) l[(size_t) t * NV + v] = 2.6f - 0.05f * (float) v;
        }
        std::vector<int> want((size_t) NT), other((size_t) NT);
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            want[(size_t) t] = reference_pick(row, p, false);      // the SPECIFIED order
            other[(size_t) t] = reference_pick(row, p, true);      // temperature first
        }
        // If the two orders agree on this fixture the test cannot see the order, and saying "the kernel
        // matches the spec" would be vacuous.
        // GREEDY CANNOT SEE THE ORDER, and saying otherwise would be a vacuous check: no filter removes the
        // global argmax, and temperature is monotonic, so the greedy pick is order-independent by
        // construction.  What the order changes is the top_p CUT, so the fixture is asserted to be
        // order-SENSITIVE at the cut, which is a property of the fixture rather than of the kernel.
        int cut_spec = 0, cut_alt = 0;
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            cut_spec += reference_cut(row, p, false);
            cut_alt += reference_cut(row, p, true);
        }
        const bool distinguishable = (cut_spec != cut_alt);
        std::printf("  %-34s %s (survivors: spec %d, temp-first %d)\n", "order is observable on this fixture",
                    distinguishable ? "yes" : "*** NO - THE FIXTURE CANNOT SEE THE ORDER ***", cut_spec,
                    cut_alt);
        if (!distinguishable) ++bad;
        // greedy is still checked here, but as an ARGMAX check, not an order check
        bad += run("greedy over this fixture", l, NT, p, want);
    }

    // ---- fixture 3: greedy consumes NO random number.  Two runs with different seeds must agree, or the
    // seeded streams diverge between greedy and sampled runs - which docs/sampling.md §3 calls out.
    {
        strata::kernels::SamplerParams a; a.top_k = 20; a.top_p = 0.95f; a.temperature = 1.0f; a.greedy = true; a.seed = 1;
        strata::kernels::SamplerParams b = a; b.seed = 999999;
        std::mt19937 rng(5); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> wa((size_t) NT);
        for (int t = 0; t < NT; ++t) wa[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV}, a, false);
        float* d_l = nullptr; int *d_a = nullptr, *d_b = nullptr;
        check(cudaMalloc(&d_l, l.size() * sizeof(float)), "m1");
        check(cudaMalloc(&d_a, (size_t) NT * sizeof(int)), "m2");
        check(cudaMalloc(&d_b, (size_t) NT * sizeof(int)), "m3");
        check(cudaMemcpy(d_l, l.data(), l.size() * sizeof(float), cudaMemcpyHostToDevice), "c1");
        strata::kernels::sample_tokens(d_l, NT, NV, nullptr, 0, a, d_a, nullptr);
        strata::kernels::sample_tokens(d_l, NT, NV, nullptr, 0, b, d_b, nullptr);
        std::vector<int> ga((size_t) NT), gb((size_t) NT);
        check(cudaMemcpy(ga.data(), d_a, ga.size() * sizeof(int), cudaMemcpyDeviceToHost), "g1");
        check(cudaMemcpy(gb.data(), d_b, gb.size() * sizeof(int), cudaMemcpyDeviceToHost), "g2");
        int mismatch = 0, wrong = 0;
        for (int t = 0; t < NT; ++t) {
            if (ga[(size_t) t] != gb[(size_t) t]) ++mismatch;
            if (ga[(size_t) t] != wa[(size_t) t]) ++wrong;
        }
        std::printf("  %-34s %s (seed-independent: %d differ; vs reference: %d wrong)\n",
                    "greedy ignores the seed", (!mismatch && !wrong) ? "matches" : "*** WRONG ***", mismatch,
                    wrong);
        bad += mismatch + wrong;
        cudaFree(d_l); cudaFree(d_a); cudaFree(d_b);
    }


    // ---- fixture 4: PENALTIES.  Two sub-cases, each built so the rule it tests decides the answer.
    {
        // host reference for the penalty stage, transcribed from llama_sampler_penalties_apply
        auto penal = [](float logit, int count, const strata::kernels::SamplerParams& p) {
            if (count <= 0) return logit;
            if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
            logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
            return logit;
        };
        auto pick = [&](const std::vector<float>& l, const std::vector<int>& hist,
                        const strata::kernels::SamplerParams& p, bool divide_unconditionally) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < (int) l.size(); ++v) {
                int c = 0; for (int h : hist) if (h == v) ++c;
                float s;
                if (divide_unconditionally && c > 0) {
                    s = l[(size_t) v] / p.penalty_repeat
                        - (float) c * p.penalty_freq - (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
                } else {
                    s = penal(l[(size_t) v], c, p);
                }
                if (first || s > bv) { bv = s; best = v; first = false; }
            }
            return best;
        };

        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f;
        p.greedy = true; p.penalty_last_n = 4; p.penalty_repeat = 2.0f;

        // A: ALL logits negative, so the multiply-or-divide rule decides the argmax
        std::vector<float> la((size_t) NV2 * NT2, -8.0f);
        for (int t = 0; t < NT2; ++t) {
            la[(size_t) t * NV2 + 0] = -1.0f;      // in the history -> penalised
            la[(size_t) t * NV2 + 1] = -1.2f;      // not penalised -> should win
        }
        std::vector<int> hist_a((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) hist_a[(size_t) t * 4 + 0] = 0;
        std::vector<int> want_a((size_t) NT2), alt_a((size_t) NT2);
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(la.begin() + (size_t) t * NV2, la.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist_a.begin() + (size_t) t * 4, hist_a.begin() + (size_t) (t + 1) * 4);
            want_a[(size_t) t] = pick(row, h, p, false);
            alt_a[(size_t) t] = pick(row, h, p, true);      // divide unconditionally
        }
        const bool A_visible = want_a[0] != alt_a[0];
        std::printf("  %-34s %s (multiply-rule %d, divide-always %d)\n",
                    "multiply-or-divide is observable", A_visible ? "yes" : "*** NO ***", want_a[0],
                    alt_a[0]);
        if (!A_visible) ++bad;
        else bad += run("penalties: repeat on negatives", la, NT2, p, want_a, hist_a, 4);

        // B: the PRESENCE penalty is a boolean, so two occurrences cost the same as one.  The runner's margin
        // is inside the difference between one and two applications.
        std::vector<float> lb((size_t) NV2 * NT2, -8.0f);
        for (int t = 0; t < NT2; ++t) {
            lb[(size_t) t * NV2 + 0] = 5.0f;       // seen twice -> penalised ONCE (presence) + freq*2
            lb[(size_t) t * NV2 + 1] = 3.4f;       // unseen
        }
        std::vector<int> hist_b((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            hist_b[(size_t) t * 4 + 0] = 0;
            hist_b[(size_t) t * 4 + 1] = 0;        // twice
        }
        strata::kernels::SamplerParams q = p; q.penalty_present = 1.5f; q.penalty_freq = 0.0f;
        std::vector<int> want_b((size_t) NT2);
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(lb.begin() + (size_t) t * NV2, lb.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist_b.begin() + (size_t) t * 4, hist_b.begin() + (size_t) (t + 1) * 4);
            want_b[(size_t) t] = pick(row, h, q, false);
        }
        std::printf("  %-34s want token %d (with present=1.5, token 0 goes 5.0/2 - 1.5 = 1.0 vs token 1 at "
                    "3.4)\n", "presence penalty is a boolean", want_b[0]);
        bad += run("penalties: presence is boolean", lb, NT2, q, want_b, hist_b, 4);
    }

    // ---- fixture 5: TEMPERATURE 0 MUST STILL RETURN THE ARGMAX.  Regression test for a real bug.
    //
    // The greedy branch used to read `apply_penalties(l[v] * inv_t, ...)`.  `inv_t` is 0.0f whenever
    // temperature <= 0, so at temperature 0 EVERY logit became 0.0f and the argmax returned index 0 - the
    // sampler emitted token 0 forever, whatever the model predicted.  OpenAI clients send `temperature: 0`
    // for greedy decoding, so this was reachable from any ordinary client.
    //
    // It survived because EVERY other greedy fixture in this file sets temperature = 1.0f, where inv_t = 1.0
    // and the extra multiply is harmless.  The bug needs temperature <= 0 to appear, and no fixture used it.
    // The fixture below makes token 0 the WORST token in every row, so returning 0 is unambiguously wrong.
    {
        const int NV3 = 512, NT3 = 4;
        std::vector<float> l((size_t) NV3 * NT3, -5.0f);
        std::vector<int> want((size_t) NT3);
        for (int t = 0; t < NT3; ++t) {
            const int best = 100 + t;                    // the argmax is never token 0
            l[(size_t) t * NV3 + best] = 3.0f;
            l[(size_t) t * NV3 + 0] = -9.0f;             // token 0 is the worst in the row
            want[(size_t) t] = best;
        }
        strata::kernels::SamplerParams p0;
        p0.top_k = 0; p0.top_p = 1.0f; p0.temperature = 0.0f; p0.greedy = false;
        bad += run("T=0 greedy=false is the argmax", l, NT3, p0, want);

        strata::kernels::SamplerParams p1 = p0; p1.greedy = true;
        bad += run("T=0 greedy=true  is the argmax", l, NT3, p1, want);

        strata::kernels::SamplerParams p2 = p0; p2.greedy = true; p2.temperature = 1.0f;
        bad += run("T=1 greedy=true  is the argmax", l, NT3, p2, want);
    }

    // ---- fixture 6: PENALTIES IN THE SAMPLED CHAIN.  Fixture 4 pins the greedy (argmax) path; the sampled
    // chain gets its own reference (the full chain with the host Philox) and its own observability check: with
    // the penalties on, the history row's favourite must LOSE a pick it would win penalty-free.
    {
        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p;
        p.top_k = 5; p.top_p = 0.9f; p.temperature = 0.8f; p.seed = 9; p.counter = 0;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.2f; p.penalty_present = 0.6f;

        std::vector<float> l((size_t) NV2 * NT2, -8.0f);
        std::vector<int> hist((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            float* row = l.data() + (size_t) t * NV2;
            row[0] = 9.0f; row[1] = 4.5f; row[2] = 4.4f; row[3] = 4.3f;   // token 0 leads clean (9 vs 4.5)
            hist[(size_t) t * 4 + 0] = 0;                                  // and falls to 2.2/2.0 penalised
            hist[(size_t) t * 4 + 1] = t == 1 ? 0 : -1;                    // (repeat 3, freq, presence)
        }
        std::vector<int> want((size_t) NT2), clean((size_t) NT2);
        strata::kernels::SamplerParams clean_p = p;
        clean_p.penalty_last_n = 0; clean_p.penalty_repeat = 1.0f;
        clean_p.penalty_freq = 0.0f; clean_p.penalty_present = 0.0f;
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV2, l.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist.begin() + (size_t) t * 4, hist.begin() + (size_t) (t + 1) * 4);
            want[(size_t) t] = sampled_reference(row, h, p, t);
            clean[(size_t) t] = sampled_reference(row, h, clean_p, t);
        }
        const bool visible = want[0] != clean[0] || want[1] != clean[1];
        std::printf("  %-34s %s (penalised picks %d/%d, clean %d/%d)\n",
                    "sampled penalties are observable", visible ? "yes" : "*** NO ***", want[0], want[1],
                    clean[0], clean[1]);
        if (!visible) ++bad;
        else bad += run("sampled chain: penalties + top_k/p", l, NT2, p, want, hist, 4);
    }

    // ---- fixture 7: MIN_P.  The cut is a PREFIX of the descending top_k list (logit >= max + log(min_p)),
    // so the fixture asserts the survivor count moves with the threshold (the observability half) and that
    // the kernel's pick equals the reference's through the full sampled chain (the correctness half).
    {
        const int NV3 = 8, NT3 = 2;
        std::vector<float> l((size_t) NV3 * NT3, -8.0f);
        for (int t = 0; t < NT3; ++t) {
            float* row = l.data() + (size_t) t * NV3;
            row[0] = 4.0f; row[1] = 3.5f; row[2] = 3.2f; row[3] = 3.1f;   // gaps keep the cut off the
            row[4] = 2.0f;                                                // logf/rounding knife edge
        }
        strata::kernels::SamplerParams base;
        base.top_k = 6; base.top_p = 1.0f; base.temperature = 0.9f; base.seed = 77;

        int c0 = 0, c05 = 0, c09 = 0;
        for (int t = 0; t < NT3; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV3, l.begin() + (size_t) (t + 1) * NV3);
            strata::kernels::SamplerParams q = base; q.min_p = 0.0f;
            c0 += sampled_cut(row, {}, q);
            q.min_p = 0.5f; c05 += sampled_cut(row, {}, q);
            q.min_p = 0.9f; c09 += sampled_cut(row, {}, q);
        }
        const bool visible = c0 > c05 && c05 > c09 && c09 >= NT3;
        std::printf("  %-34s %s (survivors: min_p 0 -> %d, 0.5 -> %d, 0.9 -> %d)\n",
                    "min_p cut is observable", visible ? "yes" : "*** NO ***", c0, c05, c09);
        if (!visible) ++bad;

        for (float mp : {0.0f, 0.5f, 0.9f}) {
            strata::kernels::SamplerParams q = base; q.min_p = mp;
            std::vector<int> want((size_t) NT3);
            for (int t = 0; t < NT3; ++t) {
                const std::vector<float> row(l.begin() + (size_t) t * NV3, l.begin() + (size_t) (t + 1) * NV3);
                want[(size_t) t] = sampled_reference(row, {}, q, t);
            }
            char name[64];
            std::snprintf(name, sizeof name, "sampled chain: min_p=%.1f", (double) mp);
            bad += run(name, l, NT3, q, want);
        }
    }

    // ---- fixture 8: THE PENALTY WINDOW IS THE TAIL.  With an 8-entry history and penalty_last_n = 4, only
    // the LAST four entries count: a token punished in the old half must come back to full strength, and one
    // punished in the tail half stays down.  The reference counts the same tail; the observability check runs
    // the reference once more WITHOUT the clamp (counting all 8) and requires the picks to differ.
    {
        const int NV4 = 8;
        strata::kernels::SamplerParams p;
        p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; p.greedy = true;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.3f; p.penalty_present = 0.5f;

        std::vector<float> l((size_t) NV4, -8.0f);
        l[0] = 6.0f; l[3] = 6.5f;                       // token 0 leads clean; token 3 is the tail offender
        std::vector<int> hist = {0, 0, 0, 0, 3, 3, 3, 3};   // token 0 old (out), token 3 in the tail

        auto pick_clamped = [&](bool clamp) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < NV4; ++v) {
                int c = 0;
                for (int i = 0; i < (clamp ? 4 : 8); ++i) if (hist[(size_t) (8 - (clamp ? 4 : 8) + i)] == v) ++c;
                float logit = l[(size_t) v];
                if (c > 0) { logit = logit <= 0.0f ? logit * p.penalty_repeat : logit / p.penalty_repeat;
                             logit -= (float) c * p.penalty_freq + p.penalty_present; }
                if (first || logit > bv) { bv = logit; best = v; first = false; }
            }
            return best;
        };
        const int want = pick_clamped(true), unclamped = pick_clamped(false);
        const bool visible = want != unclamped;
        std::printf("  %-34s %s (clamped pick %d, full-history pick %d)\n",
                    "penalty window clamp is observable", visible ? "yes" : "*** NO ***", want, unclamped);
        if (!visible) ++bad;
        else bad += run("penalty window: tail only", {l.begin(), l.end()}, 1, p, {want}, hist, 8);
    }

    // ---- fixture 9: ONE PENALTIES STAGE (issue #53), against an independently computed distribution, not the
    // reference above (which had copied the kernel's mistake).  Two tokens with equal logits, token 0 in the
    // history, presence penalty 1.5, temperature 0.7: llama.cpp's chain gives token 0 the logit (0 - 1.5) / 0.7,
    // P = 1 / (1 + exp(1.5 / 0.7)) = 0.1050; a second penalty after the temperature made it 0.0255.  Over 8,000
    // draws the kernel's share of token 0 must be near 0.105 (4 sigma = 0.014), and its picks must equal the
    // reference's draw for draw.
    {
        const int NT9 = 8000;
        strata::kernels::SamplerParams p;
        p.top_k = 2; p.top_p = 1.0f; p.min_p = 0.0f; p.temperature = 0.7f; p.seed = 53; p.counter = 0;
        p.penalty_last_n = 1; p.penalty_repeat = 1.0f; p.penalty_freq = 0.0f; p.penalty_present = 1.5f;
        std::vector<float> l((size_t) 2 * NT9, 0.0f);
        std::vector<int> hist((size_t) NT9, 0);
        std::vector<int> want((size_t) NT9);
        int zeros = 0;
        for (int t = 0; t < NT9; ++t) {
            want[(size_t) t] = sampled_reference({0.0f, 0.0f}, {0}, p, t);
            zeros += want[(size_t) t] == 0;
        }
        const double expect = 1.0 / (1.0 + std::exp(1.5 / 0.7)), share = (double) zeros / NT9;
        const bool near = std::fabs(share - expect) < 0.014;
        std::printf("  %-34s %s (token 0 drawn %.4f of %d, expected %.4f; twice-penalised would be 0.0255)\n",
                    "one penalties stage (#53)", near ? "yes" : "*** NO ***", share, NT9, expect);
        if (!near) ++bad;
        bad += run("sampled chain: #53's example", l, NT9, p, want, hist, 1);
    }

    // ---- fixture 10: TOP_P BEFORE MIN_P (llama.cpp's order).  Probabilities 0.4 / 0.3 / 0.2 / 0.1, top_p 0.75,
    // min_p 0.3 (keeps p >= 0.12): top_p over all four keeps three (0.4 + 0.3 < 0.75 <= 0.9) and min_p keeps them;
    // min_p first would drop 0.1, renormalise, and top_p would then stop at two (0.444 + 0.333 >= 0.75).  So token
    // 2 must be drawn sometimes - never in the old order - and every pick must equal the reference's.
    {
        const int NT10 = 256;
        strata::kernels::SamplerParams p;
        p.top_k = 4; p.top_p = 0.75f; p.min_p = 0.3f; p.temperature = 1.0f; p.seed = 10; p.counter = 0;
        const float lp[4] = {std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};
        std::vector<float> row = {lp[0], lp[1], lp[2], lp[3], -30.0f, -30.0f, -30.0f, -30.0f};
        std::vector<float> l;
        for (int t = 0; t < NT10; ++t) l.insert(l.end(), row.begin(), row.end());
        std::vector<int> want((size_t) NT10);
        int twos = 0;
        for (int t = 0; t < NT10; ++t) {
            want[(size_t) t] = sampled_reference(row, {}, p, t);
            twos += want[(size_t) t] == 2;
        }
        std::printf("  %-34s %s (token 2 drawn %d of %d times)\n", "top_p before min_p is observable",
                    twos > 0 ? "yes" : "*** NO ***", twos, NT10);
        if (twos == 0) ++bad;
        bad += run("sampled chain: top_p then min_p", l, NT10, p, want);
    }

    // ---- fixture 11: A STALE HISTORY WITH last_n = 0 IS INERT (PR #59).  A caller can hand over a history buffer
    // from a previous penalised request while this request disables the penalties - the run must equal the
    // no-history run in both kernels, and the bitmap the launch did not size must stay untouched.
    {
        std::mt19937 rng(11); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * 8, -1);
        for (int t = 0; t < NT; ++t) hist[(size_t) t * 8] = 3;

        strata::kernels::SamplerParams p;
        p.top_k = 20; p.top_p = 0.95f; p.temperature = 0.8f; p.seed = 5;
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t)
            want[(size_t) t] = sampled_reference({l.begin() + (size_t) t * NV,
                                                  l.begin() + (size_t) (t + 1) * NV}, {}, p, t);
        bad += run("stale history, last_n=0 (sampled)", l, NT, p, want, hist, 8);

        strata::kernels::SamplerParams gp = p; gp.greedy = true; gp.top_k = 0; gp.top_p = 1.0f;
        std::vector<int> gwant((size_t) NT);
        for (int t = 0; t < NT; ++t)
            gwant[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV,
                                                l.begin() + (size_t) (t + 1) * NV}, gp, false);
        bad += run("stale history, last_n=0 (greedy)", l, NT, gp, gwant, hist, 8);
    }

    // ---- fixture 12: ONE HISTORY PER ROW (engine 0.1.19).  A verify window samples T rows, and row t's pick
    // follows the drafts 1..t: its penalties must count them.  The engine used to stage row 0 alone, so rows
    // 1..T-1 read slots nobody wrote.  Here every row gets `penalty_rows`' history and is pinned against a
    // per-row scalar reference; the fixture first asserts it can SEE the difference - with row 0's history
    // copied to every row (the nearest well-defined stand-in for the old staging) some row must pick differently.
    // Row t's own newest token (window[t]) is its favourite by a margin the presence penalty overturns.
    {
        auto greedy_pen = [](const std::vector<float>& l, const std::vector<int>& hist,
                             const strata::kernels::SamplerParams& p) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < (int) l.size(); ++v) {
                int c = 0; for (int h : hist) if (h == v) ++c;
                float s = l[(size_t) v];
                if (c > 0) {
                    if (s <= 0.0f) s *= p.penalty_repeat; else s /= p.penalty_repeat;
                    s -= (float) c * p.penalty_freq + p.penalty_present;
                }
                if (first || s > bv) { bv = s; best = v; first = false; }
            }
            return best;
        };
        std::mt19937 rng(12); std::normal_distribution<float> g(0.0f, 1.0f);
        const int TAIL = 5000;                           // longer than the widest window below
        std::vector<int32_t> tail((size_t) TAIL);
        for (auto& v : tail) v = (int32_t) (rng() % NV);
        int observable = 0, rows_checked = 0;
        for (int T : {1, 2, 4, 8}) {
            for (int H : {1, 64, 1024, 4096}) {
                std::vector<int32_t> window((size_t) T);
                for (int t = 0; t < T; ++t) window[(size_t) t] = (int32_t) (100 + 37 * t);   // distinct, in range
                std::vector<int32_t> rows((size_t) T * H);
                strata::kernels::penalty_rows(tail.data(), TAIL, window.data(), T, H, rows.data());
                std::vector<float> l((size_t) T * NV);
                for (auto& v : l) v = g(rng);
                for (int t = 0; t < T; ++t) l[(size_t) t * NV + window[(size_t) t]] = 5.0f;
                strata::kernels::SamplerParams gp;
                gp.greedy = true; gp.temperature = 0.0f; gp.top_k = 0; gp.top_p = 1.0f;
                gp.penalty_last_n = H; gp.penalty_present = 4.0f;
                strata::kernels::SamplerParams sp2;
                sp2.top_k = 20; sp2.top_p = 0.9f; sp2.temperature = 0.7f; sp2.seed = 1000 + (uint64_t) H;
                sp2.counter = 77; sp2.penalty_last_n = H; sp2.penalty_present = 4.0f; sp2.penalty_repeat = 1.1f;
                std::vector<int> gwant((size_t) T), swant((size_t) T), rows_int(rows.begin(), rows.end());
                for (int t = 0; t < T; ++t) {
                    const std::vector<float> lr(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
                    const std::vector<int> own(rows.begin() + (size_t) t * H, rows.begin() + (size_t) (t + 1) * H);
                    const std::vector<int> row0(rows.begin(), rows.begin() + H);
                    gwant[(size_t) t] = greedy_pen(lr, own, gp);
                    swant[(size_t) t] = sampled_reference(lr, own, sp2, t);
                    if (t > 0 && greedy_pen(lr, row0, gp) != gwant[(size_t) t]) ++observable;
                    ++rows_checked;
                }
                char name[64];
                std::snprintf(name, sizeof name, "per-row history T=%d H=%d greedy", T, H);
                bad += run(name, l, T, gp, gwant, rows_int, H);
                std::snprintf(name, sizeof name, "per-row history T=%d H=%d sampled", T, H);
                bad += run(name, l, T, sp2, swant, rows_int, H);
            }
        }
        std::printf("  %-34s %s (%d drafted rows pick differently with row 0's history, of %d rows)\n",
                    "per-row histories are observable", observable > 0 ? "yes" : "*** NO ***", observable,
                    rows_checked);
        if (observable == 0) ++bad;
    }

    // ---- fixture 13: HISTORY IDS OUTSIDE THE VOCABULARY ARE IGNORED.  The bitmap is sized for n_vocab bits;
    // an id >= n_vocab used to set a bit past its end (a shared-memory write out of bounds - compute-sanitizer
    // memcheck reports it).  They can never be a candidate, so the result equals the reference without them.
    {
        std::mt19937 rng(13); std::normal_distribution<float> g(0.0f, 1.0f);
        const int H = 16;
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * H, -1), valid_only((size_t) NT * H, -1);
        const int junk[] = {NV, NV + 1000, 0x7fffffff, -5, 1 << 20};
        for (int t = 0; t < NT; ++t) {
            for (int j = 0; j < H; ++j) {
                const bool bogus = j % 3 == 0;
                const int v = bogus ? junk[(size_t) (j / 3) % 5] : (int) (rng() % NV);
                hist[(size_t) t * H + j] = v;
                if (!bogus) valid_only[(size_t) t * H + j] = v;
            }
            l[(size_t) t * NV + hist[(size_t) t * H + 1]] = 4.0f;     // a penalised favourite, so penalties matter
        }
        strata::kernels::SamplerParams gp;
        gp.greedy = true; gp.temperature = 0.0f; gp.top_k = 0; gp.top_p = 1.0f;
        gp.penalty_last_n = H; gp.penalty_present = 3.0f;
        strata::kernels::SamplerParams sp2 = gp;
        sp2.greedy = false; sp2.temperature = 0.8f; sp2.top_k = 20; sp2.top_p = 0.95f; sp2.seed = 13;
        std::vector<int> gwant((size_t) NT), swant((size_t) NT);
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> lr(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            const std::vector<int> ok(valid_only.begin() + (size_t) t * H, valid_only.begin() + (size_t) (t + 1) * H);
            {   // the greedy reference with the penalty (reference_pick has none)
                int best = 0; float bv = 0; bool first = true;
                for (int v = 0; v < NV; ++v) {
                    int c = 0; for (int h : ok) if (h == v) ++c;
                    float s = lr[(size_t) v];
                    if (c > 0) { if (s <= 0.0f) s *= gp.penalty_repeat; else s /= gp.penalty_repeat; s -= gp.penalty_present; }
                    if (first || s > bv) { bv = s; best = v; first = false; }
                }
                gwant[(size_t) t] = best;
            }
            swant[(size_t) t] = sampled_reference(lr, ok, sp2, t);
        }
        bad += run("out-of-vocab history ids (greedy)", l, NT, gp, gwant, hist, H);
        bad += run("out-of-vocab history ids (sampled)", l, NT, sp2, swant, hist, H);
    }

    // ---- fixture 14: THE top_k CONTRACT.  1..64 as given; 0 ("off") and anything wider use the widest list the
    // kernel keeps, 64 - and every row is written (the sampled kernel used to print an error for 0 and leave the
    // row unwritten, which a verify window then read as a token; `run` pre-fills -1 so that fails here).
    {
        std::mt19937 rng(14); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng) * 0.3f;             // flat: the 64-wide list matters to the draw
        strata::kernels::SamplerParams p64;
        p64.top_k = 64; p64.top_p = 1.0f; p64.temperature = 1.5f; p64.seed = 14;
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t)
            want[(size_t) t] = sampled_reference({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV},
                                                 {}, p64, t);
        bad += run("sampled top_k=64", l, NT, p64, want);
        strata::kernels::SamplerParams p0 = p64; p0.top_k = 0;
        bad += run("sampled top_k=0 means 64", l, NT, p0, want);
        strata::kernels::SamplerParams p100 = p64; p100.top_k = 100;
        bad += run("sampled top_k=100 means 64", l, NT, p100, want);
        strata::kernels::SamplerParams pneg = p64; pneg.top_k = -3;
        bad += run("sampled top_k=-3 means 64", l, NT, pneg, want);
    }

    // ---- fixture 15: `penalty_rows`, host only, against the plain definition: row t = the last h tokens of
    // tail + window[0..t], -1 padded in front.  Covers a tail shorter than, equal to and longer than h, and row 0
    // equal to the single row the engine staged before 0.1.19.
    {
        int wrong = 0, cases = 0;
        for (int n_tail : {0, 1, 5, 63, 64, 65, 300}) {
            for (int T : {1, 3, 8}) {
                for (int h : {1, 4, 64, 100}) {
                    std::vector<int32_t> tail((size_t) n_tail), window((size_t) T);
                    for (int i = 0; i < n_tail; ++i) tail[(size_t) i] = 1000 + i;
                    for (int i = 0; i < T; ++i) window[(size_t) i] = 5000 + i;
                    std::vector<int32_t> rows((size_t) T * h, 12345);
                    strata::kernels::penalty_rows(tail.data(), n_tail, window.data(), T, h, rows.data());
                    for (int t = 0; t < T; ++t) {
                        std::vector<int32_t> seq(tail);
                        seq.insert(seq.end(), window.begin(), window.begin() + t + 1);
                        std::vector<int32_t> expect((size_t) h, -1);
                        const int take = (int) std::min<size_t>((size_t) h, seq.size());
                        for (int j = 0; j < take; ++j) expect[(size_t) (h - take + j)] = seq[seq.size() - take + j];
                        if (!std::equal(expect.begin(), expect.end(), rows.begin() + (size_t) t * h)) ++wrong;
                        ++cases;
                    }
                    // row 0 = the old single-row staging: consumed tail, then the fed-back head last
                    std::vector<int32_t> old((size_t) h, -1);
                    const int take0 = (int) std::min<int64_t>(h, (int64_t) n_tail + 1);
                    for (int j = 0; j < take0 - 1; ++j) old[(size_t) (h - take0 + j)] = tail[(size_t) (n_tail - (take0 - 1) + j)];
                    old[(size_t) (h - 1)] = window[0];
                    if (!std::equal(old.begin(), old.end(), rows.begin())) ++wrong;
                    ++cases;
                }
            }
        }
        std::printf("  %-34s %s (%d of %d rows differ)\n", "penalty_rows layout", wrong ? "*** WRONG ***" : "matches",
                    wrong, cases);
        bad += wrong;
    }

    // A continuous stream and individual decode calls consume the same draw counters.
    {
        constexpr int count = 32, vocab = 16;
        std::vector<float> uniform(count * vocab, 0.0f);
        float* input = nullptr;
        int* output = nullptr;
        check(cudaMalloc(&input, uniform.size() * sizeof(float)), "counter logits");
        check(cudaMalloc(&output, count * sizeof(int)), "counter output");
        check(cudaMemcpy(input, uniform.data(), uniform.size() * sizeof(float), cudaMemcpyHostToDevice), "counter upload");
        strata::kernels::SamplerParams p;
        p.top_k = vocab; p.top_p = 1.0f; p.seed = 123; p.counter = (uint64_t(1) << 32) + 7;
        strata::kernels::sample_tokens(input, count, vocab, nullptr, 0, p, output, nullptr);
        std::vector<int> batch(count), singles(count), repeated(count);
        check(cudaMemcpy(batch.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter batch");
        for (int i = 0; i < count; ++i) {
            auto one = p; one.counter += i;
            strata::kernels::sample_tokens(input, 1, vocab, nullptr, 0, one, output + i, nullptr);
        }
        check(cudaMemcpy(singles.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter singles");
        strata::kernels::sample_tokens(input, count, vocab, nullptr, 0, p, output, nullptr);
        check(cudaMemcpy(repeated.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter repeated");
        bool varies = false;
        for (int i = 1; i < count; ++i) varies |= batch[i] != batch[0];
        const bool valid = batch == singles && batch == repeated && varies;
        std::printf("  sampler draw counter segmentation/repeat: %s\n", valid ? "PASS" : "FAIL");
        bad += !valid;
        cudaFree(input); cudaFree(output);
    }

    std::printf("\nsampler: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("sampler_parity OK\n");
    return 0;
}
