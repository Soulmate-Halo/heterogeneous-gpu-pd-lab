// src/kernels/qsa_prompt_attn_parity.cpp - perf-review D-1: the tensor-core prompt attention (qsa_prompt_attn.hpp)
// against the FP32 kernel it replaces (`qsa_decode_attn_batch`) and an FP64 host reference (GPU, synthetic, no model).
//
// Int8 and FP16 pools with random codes, scales and queries; selections shaped like the prompt path's (the 2,051
// widest, a recent window plus older cells that drift slowly from one query to the next, so neighbours share most
// of them as they do in a real prompt; short contexts take every cell). Checks:
//   1. against FP64, the new kernel's error is no larger than a small multiple of the old kernel's (both FP32 math);
//   2. the new and old outputs agree to a relative 1e-4 of the output scale;
// then times both over a prompt chunk (the old one in batches of 32, as prefill.cpp calls it).
// Usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_fp16.h>
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
float h2f(uint16_t b) { __half h; *reinterpret_cast<uint16_t*>(&h) = b; return __half2float(h); }
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

int run(int fmt, int64_t ctx, int64_t nq, int reps) {   // fmt 1 int8, 0 fp16
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    // pools; page table a shuffled permutation (the readers must follow it)
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    if (fmt == 1) {
        kq.resize(rows * HD); vq.resize(rows * HD); ks.resize(rows * 4); vs.resize(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        for (auto& x : vs) x = f2h(sc(rng));
    } else {
        kh.resize(rows * HD); vh.resize(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    // queries at positions ctx - nq .. ctx - 1
    const int64_t cap = k::qsa_selection_width(ctx, s);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[i * k::kStepCount + k::kStepWidth] = (int32_t) w;
        if (w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {   // the older cells: a set that drifts ~3% per query
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
    k::QsaAttnPools pl;
    if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr;
    const int64_t batch = 32;
    ck(cudaMalloc(&d_old, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&d_new, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&scratch, batch * k::qsa_decode_attn_scratch_floats(cap, s) * 4), "malloc");
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                     scratch, d_old + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };
    auto new_run = [&]() {
        if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, nullptr)) {
            std::fprintf(stderr, "qsa_prompt_attn_batch refused the pools\n");
            std::exit(2);
        }
    };
    old_run();
    new_run();
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size());
    ck(cudaMemcpy(o.data(), d_old, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(nw.data(), d_new, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
    // 1. FP64 reference on a sample of queries
    double err_old = 0, err_new = 0, ref_scale = 0;
    for (int64_t i = 0; i < nq; i += std::max<int64_t>(1, nq / 16)) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w);
            double mx = -1e300;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d) {
                    const double kv = fmt == 1 ? (double) kq[row * HD + d] * h2f(ks[row * 4 + d / 64]) : h2f(kh[row * HD + d]);
                    a += (double) q[(i * NH + h) * HD + d] * kv;
                }
                sco[c] = a / 16.0;
                mx = std::max(mx, sco[c]);
            }
            double l = 0;
            for (auto& x : sco) { x = std::exp(x - mx); l += x; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t c = 0; c < w; ++c) {
                    const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                    const double vv = fmt == 1 ? (double) vq[row * HD + d] * h2f(vs[row * 4 + d / 64]) : h2f(vh[row * HD + d]);
                    a += sco[c] * vv;
                }
                const double r = a / l;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, std::fabs(o[at] - r));
                err_new = std::max(err_new, std::fabs(nw[at] - r));
            }
        }
    }
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    // 3. speed
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float ms_old = 0, ms_new = 0;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) old_run();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    cudaEventElapsedTime(&ms_old, e0, e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) new_run();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    cudaEventElapsedTime(&ms_new, e0, e1);
    const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-6 * ref_scale);
    const bool ok2 = diff <= 1e-4 * scale;
    std::printf("%s %s ctx %lld, %lld queries: vs FP64 old %.3g new %.3g (output scale %.3g); new vs old %.3g (%.2g of "
                "scale); %.3f -> %.3f ms per chunk (%.2fx)\n",
                ok1 && ok2 ? "PASS" : "FAIL", fmt == 1 ? "int8" : "fp16", (long long) ctx, (long long) nq, err_old,
                err_new, ref_scale, diff, diff / scale, ms_old / reps, ms_new / reps, ms_old / ms_new);
    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q); cudaFree(d_old); cudaFree(d_new);
    cudaFree(scratch);
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.page_table);
    return ok1 && ok2 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    int fails = 0;
    fails += run(1, ctx, nq, reps);
    fails += run(0, ctx, nq, reps);
    fails += run(1, 1500, std::min<int64_t>(nq, 1500), reps);   // short context: the selection is every cell
    fails += run(1, 2100, std::min<int64_t>(nq, 256), reps);    // the identity-to-sparse edge
    std::printf("FAILURES: %d\n", fails);
    return fails;
}
