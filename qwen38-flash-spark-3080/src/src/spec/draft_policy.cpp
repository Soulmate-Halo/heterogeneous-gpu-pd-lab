// src/spec/draft_policy.cpp - see include/strata/spec/draft_policy.hpp.
#include "strata/spec/draft_policy.hpp"

#include <algorithm>

namespace strata::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet (the
// measured round times replace it). Between the RTX 5070's measured curves: +10 ms per token with every missed expert
// on the CPU (bench/results/2026-09-27-spec/window-cost), flatter with the default CPU/DMA split.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
constexpr double kCostAlpha = 0.1;    // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it

}  // namespace

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[t] > 0) return cost_[t];
    // scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cost_n_[u] > 0) {
            const double w = std::min(cost_n_[u], 20.0);
            num += w * cost_[u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return 1.0 + 0.7 * (t - 1);       // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) { best = r; best_t = k + 1; }
    }
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (t_full > p.t && cost_n_[t_full] < kProbes && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) {
        cost_[t] = cost_n_[t] > 0 ? (1.0 - kCostAlpha) * cost_[t] + kCostAlpha * round_ms : round_ms;
        cost_n_[t] += 1.0;
    }
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

}  // namespace strata::spec
