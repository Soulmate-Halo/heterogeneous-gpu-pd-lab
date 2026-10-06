// src/kernels/qsa_select_bench.cpp - the prompt path's QSA selection (qsa_select.hpp) timed per stage, block scores
// (the warp kernel and the tensor-core one) and top-k, for a batch of consecutive queries at a given context, and
// the two scorers compared: score difference and how many selections differ (GPU, synthetic, no model).
// Usage: qsa_select_bench [context=131072] [queries=256] [reps=10]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 131072;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = ctx / 4 + 1, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // keys with a shared direction plus noise, so the scores have a spread like a real indexer's
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {   // qsa_step_fill's arithmetic (kept here so the bench links only qsa_select)
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_pooled = up(pooled);
    const float* d_dead = up(dead);
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *sc_old = nullptr, *sc_new = nullptr;
    int32_t *ids_old = nullptr, *ids_new = nullptr;
    ck(cudaMalloc(&sc_old, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&sc_new, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&ids_old, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "malloc");
    const int64_t active = steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1;
    auto run_old = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_old, nullptr, active); };
    auto run_new = [&] {
        if (!k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_new, nullptr, active)) {
            std::fprintf(stderr, "tc scorer refused\n");
            std::exit(2);
        }
    };
    run_old();
    run_new();
    k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr);
    k::qsa_block_topk(sc_new, d_steps, nq, max_blocks, cap, s, ids_new, nullptr);
    int32_t* ids_reg = nullptr;   // the register top-k on the OLD scores: must equal the reference exactly
    ck(cudaMalloc(&ids_reg, (size_t) (nq * cap) * 4), "malloc");
    k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr);
    ck(cudaDeviceSynchronize(), "warm");
    // compare
    std::vector<float> a((size_t) (nq * max_blocks)), b(a.size());
    std::vector<int32_t> ia((size_t) (nq * cap)), ib(ia.size());
    ck(cudaMemcpy(a.data(), sc_old, a.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(b.data(), sc_new, b.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(ia.data(), ids_old, ia.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(ib.data(), ids_new, ib.size() * 4, cudaMemcpyDeviceToHost), "down");
    std::vector<int32_t> ir(ia.size());
    ck(cudaMemcpy(ir.data(), ids_reg, ir.size() * 4, cudaMemcpyDeviceToHost), "down");
    int64_t reg_same = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        reg_same += std::equal(ia.begin() + i * cap, ia.begin() + i * cap + w, ir.begin() + i * cap);
    }
    double max_rel = 0, sum_rel = 0, n_rel = 0;
    int64_t same_sel = 0, cells_diff = 0, cells_all = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        for (int64_t j = 0; j <= st[k::kStepNBid]; ++j) {
            const double x = a[(size_t) (i * max_blocks + j)], y = b[(size_t) (i * max_blocks + j)];
            const double r = std::fabs(x - y) / std::max(1e-6, std::fabs(x));
            max_rel = std::max(max_rel, r);
            sum_rel += r;
            n_rel += 1;
        }
        const int64_t w = st[k::kStepWidth];
        std::vector<int32_t> x(ia.begin() + i * cap, ia.begin() + i * cap + w), y(ib.begin() + i * cap, ib.begin() + i * cap + w);
        same_sel += x == y;
        std::vector<int32_t> d;
        std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(d));
        cells_diff += (int64_t) d.size() / 2;
        cells_all += w;
    }
    // time
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timed = [&](auto f) {
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) f();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };
    const float t_old = timed(run_old), t_new = timed(run_new);
    const float t_tk = timed([&] { k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr); });
    const float t_tk2 = timed([&] { k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr); });
    std::printf("top-k %.3f -> %.3f ms (%.1fx), register top-k identical to the reference %lld/%lld\n", t_tk, t_tk2,
                t_tk / t_tk2, (long long) reg_same, (long long) nq);
    std::printf("ctx %lld, %lld queries x %lld blocks: scores %.3f -> %.3f ms (%.1fx), top-k %.3f ms; score rel diff "
                "mean %.2g max %.2g; selections identical %lld/%lld, cells differing %.4f%%\n", (long long) ctx,
                (long long) nq, (long long) active, t_old, t_new, t_old / t_new, t_tk, sum_rel / std::max(1.0, n_rel),
                max_rel, (long long) same_sel, (long long) nq, cells_all ? 100.0 * (double) cells_diff / (double) cells_all : 0.0);
    return 0;
}
