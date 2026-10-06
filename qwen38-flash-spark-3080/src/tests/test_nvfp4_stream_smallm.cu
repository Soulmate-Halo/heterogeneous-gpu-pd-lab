// Small-M NVFP4 stream GEMM vs marlin. Same threshold as tests/test_marlin_nvfp4.cu:
// max relative error <= 1e-2 with denom max(|ref|, 0.5), over cap in {1,2,4,8}
// and both batch GEMMs (w13: N=1280,K=2560,mul_topk=0 and w2: N=2560,K=640,mul_topk=1).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "moe_wna16_marlin.cuh"
#include "strata/kernels/nvfp4_marlin_expert.hpp"

namespace {

struct Buf {
    void* d = nullptr;
    std::vector<int64_t> shape;
    DLDataType dtype{};
    size_t bytes = 0;
};

void CudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

Buf Alloc(const std::vector<int64_t>& shape, DLDataType dt, const void* host, size_t nbytes) {
    Buf b;
    b.dtype = dt;
    b.shape = shape;
    b.bytes = nbytes;
    CudaCheck(cudaMalloc(&b.d, nbytes ? nbytes : 16), "cudaMalloc");
    if (host && nbytes) CudaCheck(cudaMemcpy(b.d, host, nbytes, cudaMemcpyHostToDevice), "H2D");
    else CudaCheck(cudaMemset(b.d, 0, nbytes ? nbytes : 16), "memset");
    return b;
}

tvm::ffi::TensorView ViewOf(Buf& b) {
    DLTensor dl;
    dl.data = b.d;
    dl.device = {kDLCUDA, 0};
    dl.ndim = static_cast<int>(b.shape.size());
    dl.dtype = b.dtype;
    dl.shape = b.shape.data();
    dl.strides = nullptr;
    dl.byte_offset = 0;
    return tvm::ffi::TensorView(&dl);
}

uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

uint16_t fold_global(float g) {
    const uint16_t hi = f32_to_bf16(g);
    return static_cast<uint16_t>(hi + (119u << 7));
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

// rel threshold copied from tests/test_marlin_nvfp4.cu CompareBf16.
int Compare(const uint16_t* got, const uint16_t* ref, size_t n, double* worst, int* ulp_worst) {
    int bad = 0;
    *worst = 0;
    *ulp_worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const float r = bf16_to_f32(ref[i]);
        const float v = bf16_to_f32(got[i]);
        const double rel = std::fabs(v - r) / std::max(std::fabs(static_cast<double>(r)), 0.5);
        if (rel > *worst) *worst = rel;
        if (rel > 1e-2) ++bad;
        if ((ref[i] >> 15) == (got[i] >> 15)) {
            const int ulp = std::abs(static_cast<int>(ref[i]) - static_cast<int>(got[i]));
            if (ulp > *ulp_worst) *ulp_worst = ulp;
        }
    }
    return bad;
}

int RunCase(int cap, int N, int K, int per_row, int mul_topk, int E, int sms) {
    const int nblk = per_row ? cap : 1;
    const int topk = 1;
    const int block_m = 16;
    const int group = 16;
    uint32_t rng = 0x4e563470u ^ (static_cast<uint32_t>(cap) * 17u) ^ (static_cast<uint32_t>(N) << 1) ^
                   (per_row ? 0x9u : 0x3u);

    const size_t w_bytes = static_cast<size_t>(E) * (static_cast<size_t>(N) * K / 2);
    const size_t s_bytes = static_cast<size_t>(E) * (static_cast<size_t>(K) / group) * N;
    std::vector<uint8_t> h_w(w_bytes), h_s(s_bytes);
    for (size_t i = 0; i < w_bytes; ++i) h_w[i] = static_cast<uint8_t>(lcg(rng) & 0xff);
    // Processed-scale-like positive E4M3 bytes (not 0x7F NaN).
    static const uint8_t kScale[8] = {0xa8, 0xb0, 0xb2, 0xab, 0xac, 0xb4, 0xb6, 0xa0};
    for (size_t i = 0; i < s_bytes; ++i) h_s[i] = kScale[lcg(rng) & 7];

    std::vector<uint16_t> h_g(E), h_a(static_cast<size_t>(cap) * K);
    for (int e = 0; e < E; ++e) h_g[e] = fold_global(0.55f + 0.05f * e);
    for (size_t i = 0; i < h_a.size(); ++i) {
        const float x = (static_cast<int>(lcg(rng) % 2001) - 1000) / 2000.f;
        h_a[i] = f32_to_bf16(x);
    }
    std::vector<float> h_topk(cap);
    for (int r = 0; r < cap; ++r) h_topk[r] = mul_topk ? (0.5f + 0.05f * r) : 1.f;

    std::vector<int32_t> h_sorted(static_cast<size_t>(nblk) * block_m, cap);
    std::vector<int32_t> h_experts(nblk);
    if (per_row) {
        for (int b = 0; b < cap; ++b) {
            h_experts[b] = b % E;
            h_sorted[static_cast<size_t>(b) * block_m] = b;
        }
    } else {
        h_experts[0] = 0;
        for (int j = 0; j < cap; ++j) h_sorted[j] = j;
    }
    const int32_t num_post = nblk * block_m;

    const DLDataType bf16{(uint8_t)4, 16, 1};
    const DLDataType i32{(uint8_t)0, 32, 1};
    const DLDataType f32{(uint8_t)2, 32, 1};
    const DLDataType f8{(uint8_t)8, 8, 1};

    Buf a = Alloc({cap, K}, bf16, h_a.data(), h_a.size() * 2);
    Buf w = Alloc({E, K / 16, N * 2}, i32, h_w.data(), w_bytes);
    Buf sc = Alloc({E, K / group, N}, f8, h_s.data(), s_bytes);
    Buf gm = Alloc({E, 1}, bf16, h_g.data(), h_g.size() * 2);
    Buf sorted = Alloc({num_post}, i32, h_sorted.data(), h_sorted.size() * 4);
    Buf experts = Alloc({nblk}, i32, h_experts.data(), h_experts.size() * 4);
    Buf npost = Alloc({1}, i32, &num_post, 4);
    Buf tw = Alloc({cap}, f32, h_topk.data(), h_topk.size() * 4);
    Buf c_ref = Alloc({cap, N}, bf16, nullptr, static_cast<size_t>(cap) * N * 2);
    Buf c_got = Alloc({cap, N}, bf16, nullptr, static_cast<size_t>(cap) * N * 2);
    Buf bias = Alloc({0}, bf16, nullptr, 0);
    Buf zeros = Alloc({0}, bf16, nullptr, 0);
    Buf gidx = Alloc({0}, i32, nullptr, 0);
    Buf perm = Alloc({0}, i32, nullptr, 0);
    Buf atmp = Alloc({0}, bf16, nullptr, 0);

    const int64_t sorted_len = num_post;
    int64_t ctmp_n = std::min(static_cast<int64_t>(N) * sorted_len, static_cast<int64_t>(sms) * 4 * block_m * 256);
    if (ctmp_n < 1) ctmp_n = 1;
    Buf ctmp = Alloc({ctmp_n}, f32, nullptr, static_cast<size_t>(ctmp_n) * 4);
    Buf locks = Alloc({sms * 4}, i32, nullptr, static_cast<size_t>(sms) * 16);

    const int64_t work_n = static_cast<int64_t>(4) * cap * N;
    Buf work = Alloc({work_n}, f32, nullptr, static_cast<size_t>(work_n) * 4);

    const auto qt = sglang::host::kFE2M1f.id();
    sglang::moe_wna16_marlin_gemm<nv_bfloat16, false, false>(
        ViewOf(a), ViewOf(c_ref), ViewOf(w), ViewOf(bias), ViewOf(sc), ViewOf(gm), ViewOf(zeros), ViewOf(gidx),
        ViewOf(perm), ViewOf(locks), ViewOf(sorted), ViewOf(experts), ViewOf(npost), ViewOf(tw), ViewOf(atmp),
        ViewOf(ctmp), block_m, topk, mul_topk != 0, /*is_ep=*/false, qt, /*size_m=*/cap, /*size_n=*/N,
        /*size_k=*/K, /*has_act_order=*/false, /*has_bias=*/false, /*is_k_full=*/true, /*has_zp=*/false,
        /*num_groups=*/K / group, group, /*use_atomic_add=*/false, /*use_fp32_reduce=*/true, /*is_zp_float=*/false);
    CudaCheck(cudaGetLastError(), "marlin launch");
    CudaCheck(cudaDeviceSynchronize(), "marlin sync");

    strata::kernels::nvfp4_stream_expert_gemm(a.d, c_got.d, static_cast<const uint8_t*>(w.d),
                                              static_cast<const uint8_t*>(sc.d), static_cast<const uint16_t*>(gm.d),
                                              static_cast<const int32_t*>(sorted.d),
                                              static_cast<const int32_t*>(experts.d), static_cast<const float*>(tw.d),
                                              static_cast<float*>(work.d), work_n, cap, nblk, N, K, mul_topk, nullptr);
    CudaCheck(cudaGetLastError(), "stream launch");
    CudaCheck(cudaDeviceSynchronize(), "stream sync");

    const size_t n = static_cast<size_t>(cap) * N;
    std::vector<uint16_t> href(n), hgot(n);
    CudaCheck(cudaMemcpy(href.data(), c_ref.d, n * 2, cudaMemcpyDeviceToHost), "D2H ref");
    CudaCheck(cudaMemcpy(hgot.data(), c_got.d, n * 2, cudaMemcpyDeviceToHost), "D2H got");
    double worst = 0;
    int ulp = 0;
    const int bad = Compare(hgot.data(), href.data(), n, &worst, &ulp);
    float peak = 0.f;
    for (uint16_t b : href) peak = std::max(peak, std::fabs(bf16_to_f32(b)));
    const int zero = peak < 1e-4f;
    std::printf("  cap=%d N=%d K=%d %s mul_topk=%d numel=%zu max_rel=%.6g max_ulp=%d peak=%.4g bad=%d\n", cap, N, K,
                per_row ? "per-row" : "one-expert", mul_topk, n, worst, ulp, peak, bad + zero);
    return bad + zero;
}

}  // namespace

int main() {
    int sms = 0;
    CudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "sms");
    std::printf("test_nvfp4_stream_smallm sms=%d\n", sms);
    const int E = 8;
    int bad = 0;
    const int caps[] = {1, 2, 4, 8};
    struct Shape {
        int N, K, mul;
    };
    const Shape shapes[] = {{1280, 2560, 0}, {2560, 640, 1}};
    for (int cap : caps) {
        for (const Shape& s : shapes) {
            bad += RunCase(cap, s.N, s.K, /*per_row=*/0, s.mul, E, sms);
            if (cap > 1) bad += RunCase(cap, s.N, s.K, /*per_row=*/1, s.mul, E, sms);
        }
    }
    if (bad == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL bad=%d\n", bad);
    return 2;
}
