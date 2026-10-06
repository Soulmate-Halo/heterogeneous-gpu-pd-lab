// src/kernels/kv_fp8_parity.cpp - FP8 e4m3 KV append/gather/attention against a host reference (GPU, no model).
//
// Same shape as kv_q8_parity: random K/V cells through a non-identity page table, into the FP8 pools
// (kv_append_fp8_step) and the FP16 pools (kv_append_step). Checks:
//   1. FP16 scales are BITWISE equal to f16_from_f32(max|x| / 448); a zero scale stores a zero group;
//   2. dequant (float(e4m3) * scale) is within 17 scale-units of x (e4m3 half-ulp at 448 is 16, plus the
//      fp16 rounding of the stored scale);
//   3. the FP8 gather is BITWISE equal to f16_from_f32 of that host dequant (signed zeros compare equal);
//   4. against the FP16 path, the gathered values differ by at most 17 scale-units;
//   5. qsa_decode_attn KV_MODE 2 vs the FP16 pools: relative RMSE of the attention output <= 0.20.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_fp8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

float e4m3_to_f32(uint8_t c) {
    const int neg = c & 0x80;
    const int exp = (c >> 3) & 0x0f;
    const int man = c & 0x07;
    float v;
    if (exp == 0) v = std::ldexp(static_cast<float>(man), -9);
    else if (exp == 15 && man == 7) v = std::numeric_limits<float>::quiet_NaN();
    else v = std::ldexp(1.f + static_cast<float>(man) * 0.125f, exp - 7);
    return neg ? -v : v;
}
bool same_f16(uint16_t a, uint16_t b) {
    if (a == b) return true;
    return (a & 0x7fffu) == 0 && (b & 0x7fffu) == 0;
}
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_FP8_GROUP;
    const int pages = 8, cells = pages * P;
    std::mt19937 rng(19);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");
    uint8_t *kq = dalloc<uint8_t>((size_t) cells * H * D), *vq = dalloc<uint8_t>((size_t) cells * H * D);
    uint16_t *ks = dalloc<uint16_t>((size_t) cells * H * G), *vs = dalloc<uint16_t>((size_t) cells * H * G);
    uint16_t *kp = dalloc<uint16_t>((size_t) cells * H * D), *vp = dalloc<uint16_t>((size_t) cells * H * D);
    float *kcur = dalloc<float>(H * D), *vcur = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    std::vector<std::vector<float>> hk(cells), hv(cells);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n_fill = cells - 37;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        const float scale = (n % 7 == 0) ? 1e-3f : (n % 11 == 0 ? 40.f : 1.f);
        for (auto& x : kv) x = nd(rng) * scale;
        for (auto& x : vv) x = nd(rng) * scale;
        if (n % 13 == 0) std::fill(kv.begin(), kv.begin() + 64, 0.f);
        hk[pos] = kv; hv[pos] = vv;
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        ck(cudaMemcpy(kcur, kv.data(), kv.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vcur, vv.data(), vv.size() * 4, cudaMemcpyHostToDevice), "v");
        k::kv_append_fp8_step(kq, vq, ks, vs, d_table, step, kcur, vcur, s, nullptr);
        k::kv_append_step(kp, vp, d_table, step, kcur, vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "append");
    }
    std::vector<uint8_t> hkq((size_t) cells * H * D), hvq(hkq.size());
    std::vector<uint16_t> hks((size_t) cells * H * G), hvs(hks.size());
    ck(cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hks.data(), ks, hks.size() * 2, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hvs.data(), vs, hvs.size() * 2, cudaMemcpyDeviceToHost), "d2h");
    long bad_scales = 0;
    double worst_qx = 0.0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int kvsel = 0; kvsel < 2; ++kvsel)
            for (int h = 0; h < H; ++h)
                for (int g = 0; g < G; ++g) {
                    const float* x = (kvsel ? hv[pos] : hk[pos]).data() + h * D + g * 64;
                    float amax = 0.f;
                    for (int t = 0; t < 64; ++t) amax = std::fmax(amax, std::fabs(x[t]));
                    const uint16_t sb = k::f16_from_f32(amax / k::KV_FP8_EMAX);
                    const float sf = k::f32_from_f16(sb);
                    const long long row = ((long long) table[pos / P] * H + h) * P + pos % P;
                    if ((kvsel ? hvs : hks)[row * G + g] != sb) ++bad_scales;
                    for (int t = 0; t < 64; ++t) {
                        const uint8_t q = (kvsel ? hvq : hkq)[row * D + g * 64 + t];
                        if (!(sf > 0.f)) {
                            if (q != 0) ++bad_scales;
                            continue;
                        }
                        const double err = std::fabs((double) e4m3_to_f32(q) * sf - (double) x[t]) / (double) sf;
                        worst_qx = std::max(worst_qx, err);
                    }
                }
    }
    if (bad_scales) { std::fprintf(stderr, "FAIL: %ld scales/zero-groups differ from the host reference\n", bad_scales); ++g_fail; }
    // half an e4m3 ulp at 448 is 16 scale-units; the stored fp16 scale adds well under 1
    if (!(worst_qx <= 17.0)) { std::fprintf(stderr, "FAIL: quant-dequant error %.3f scale-units\n", worst_qx); ++g_fail; }

    const int max_ids = 200;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t *k8 = dalloc<uint16_t>((size_t) max_ids * H * D), *v8 = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t *k16 = dalloc<uint16_t>((size_t) max_ids * H * D), *v16 = dalloc<uint16_t>((size_t) max_ids * H * D);
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        const int n_ids = 1 + (int) (rng() % max_ids);
        std::vector<int32_t> ids(n_ids);
        for (auto& id : ids) id = positions[rng() % n_fill];
        ck(cudaMemcpy(d_ids, ids.data(), n_ids * 4, cudaMemcpyHostToDevice), "ids");
        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        k::kv_gather_fp8_step(kq, vq, ks, vs, d_table, d_ids, step, max_ids, s, k8, v8, nullptr);
        k::kv_gather_step(kp, vp, d_table, d_ids, step, max_ids, s, k16, v16, nullptr);
        ck(cudaDeviceSynchronize(), "gather");
        std::vector<uint16_t> a8((size_t) n_ids * H * D), b8(a8.size()), a16(a8.size()), b16(a8.size());
        ck(cudaMemcpy(a8.data(), k8, a8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b8.data(), v8, b8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(a16.data(), k16, a16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b16.data(), v16, b16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        for (int j = 0; j < n_ids; ++j)
            for (int h = 0; h < H; ++h) {
                const long long row = ((long long) table[ids[j] / P] * H + h) * P + ids[j] % P;
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) j * H + h) * D + d;
                    for (int kvsel = 0; kvsel < 2; ++kvsel) {
                        const float sf = k::f32_from_f16((kvsel ? hvs : hks)[row * G + d / 64]);
                        const uint8_t q = (kvsel ? hvq : hkq)[row * D + d];
                        const uint16_t want = k::f16_from_f32(e4m3_to_f32(q) * sf);
                        const uint16_t got = (kvsel ? b8 : a8)[o];
                        if (!same_f16(got, want)) {
                            if (g_fail < 5) std::fprintf(stderr, "FAIL: gather id %d h %d d %d\n", j, h, d);
                            ++g_fail;
                        }
                        const float ref = k::f32_from_f16((kvsel ? b16 : a16)[o]);
                        const double err = std::fabs(k::f32_from_f16(got) - ref) / (sf > 0 ? sf : 1.0);
                        worst = std::max(worst, err);
                    }
                }
            }
    }
    if (!(worst <= 17.0)) { std::fprintf(stderr, "FAIL: FP8 vs FP16 error %.3f scale-units\n", worst); ++g_fail; }

    // attention: KV_MODE 2 over the fp8 pools vs the fp16 pools, one query, 64 selected cells
    {
        const int cap = 64, n_ids = 64;
        std::vector<int32_t> ids(n_ids);
        for (int i = 0; i < n_ids; ++i) ids[i] = positions[i % n_fill];
        ck(cudaMemcpy(d_ids, ids.data(), n_ids * 4, cudaMemcpyHostToDevice), "attn ids");
        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "attn step");
        const int QH = (int) s.n_head;
        std::vector<float> hqv((size_t) QH * D);
        for (auto& x : hqv) x = nd(rng);
        float* dq = dalloc<float>(hqv.size());
        ck(cudaMemcpy(dq, hqv.data(), hqv.size() * 4, cudaMemcpyHostToDevice), "q");
        const uint64_t nscratch = k::qsa_decode_attn_scratch_floats(cap, s);
        float* scratch = dalloc<float>(nscratch);
        float* attn8 = dalloc<float>((size_t) QH * D);
        float* attn16 = dalloc<float>((size_t) QH * D);
        k::QsaAttnPools p8;
        p8.k_fp8 = kq; p8.v_fp8 = vq; p8.k_fp8_scale = ks; p8.v_fp8_scale = vs; p8.page_table = d_table;
        k::QsaAttnPools p16;
        p16.k_pool = kp; p16.v_pool = vp; p16.page_table = d_table;
        k::qsa_decode_attn_step(dq, p8, d_ids, step, cap, s, scratch, attn8, nullptr);
        k::qsa_decode_attn_step(dq, p16, d_ids, step, cap, s, scratch, attn16, nullptr);
        ck(cudaDeviceSynchronize(), "attn");
        std::vector<float> a8o((size_t) QH * D), a16o(a8o.size());
        ck(cudaMemcpy(a8o.data(), attn8, a8o.size() * 4, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(a16o.data(), attn16, a16o.size() * 4, cudaMemcpyDeviceToHost), "d2h");
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < a8o.size(); ++i) {
            const double d = (double) a8o[i] - (double) a16o[i];
            num += d * d;
            den += (double) a16o[i] * (double) a16o[i];
        }
        const double rmse = std::sqrt(num / (den > 0.0 ? den : 1.0));
        if (!(rmse <= 0.20)) { std::fprintf(stderr, "FAIL: attention relative RMSE %.4f\n", rmse); ++g_fail; }
        std::printf("kv_fp8_parity: %s (quant %.3f scale-units, vs FP16 %.3f, attn RMSE %.4f)\n",
                    g_fail ? "FAILED" : "OK", worst_qx, worst, rmse);
    }
    return g_fail ? 1 : 0;
}
