// src/kernels/cpu/arm_stubs.cpp - non-x86 stand-ins for the AVX-512/AVX2 CPU expert kernels.
//
// Built ONLY when CMAKE_SYSTEM_PROCESSOR is not x86 (see CMakeLists.txt).  The x86 translation units
// (expert.cpp, q2_avx2.cpp, iq_avx512.cpp, iq_avx2.cpp) cannot compile on aarch64, but their symbols are
// referenced by portable code (expert_layout.cpp dispatchers, native_expert.cpp, generate.cpp's startup
// probe).  On a GPU-only deployment (`--no-pool`, every expert admitted to the VRAM cache) none of the
// compute entry points below can be reached; if one is reached anyway the configuration is wrong, so the
// stub says so and stops instead of silently producing garbage.
//
// Deliberately NOT stubbed to a failure:
//   * cpu_require_expert_support() - generate.cpp calls it at startup for Q2_0 packs; on a GPU-only ARM
//     host the CPU never computes an expert, so the check is a no-op with a one-line notice.
//   * iq256_supported()/iq512_supported() - native_expert.cpp uses these to pick a kernel; false simply
//     steers it to the portable ggml-cpu path, which is correct (if slow) and keeps parity tooling usable.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {

[[noreturn]] void x86_only(const char* what) {
    std::fprintf(stderr,
                 "strata: %s is an x86 AVX kernel and this build is aarch64.  The CPU expert pool must not "
                 "receive work on this target - run with --no-pool and admit every expert to the GPU cache.\n",
                 what);
    std::abort();
}

bool g_oracle_q8_0 = false;

}  // namespace

const char* CpuFeatures::reason() const { return "aarch64 build: x86 AVX-512 expert kernels are absent"; }

CpuFeatures cpu_features() { return CpuFeatures{}; }

void cpu_require_expert_support() {
    std::fprintf(stderr, "strata: aarch64 build - CPU expert kernels absent, GPU-only mode required\n");
}

void expert_set_oracle_q8_0(bool enabled) { g_oracle_q8_0 = enabled; }
bool expert_oracle_q8_0_enabled() { return g_oracle_q8_0; }

// The activation quantizer IS needed on aarch64: expert_source.cpp quantizes every layer's activation
// before the hit/miss split, even when every expert is GPU-resident.  This is the scalar reference loop
// from expert.cpp:421 verbatim - the AVX-512 kernel is documented bitwise-identical to it, so the GPU hit
// path's contract (quantize_act.cu reproduces this exactly) still sees the same values.
void act_quant_q8_1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);   // round half away from zero
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }
void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { x86_only("s2_expert_vnni"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { x86_only("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("s2_expert_down_rows_multi");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    x86_only("s2_expert_vnni_multi");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx2");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { x86_only("s2_expert_scalar"); }

bool iq256_supported(int) noexcept { return false; }
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_gu_rows");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_rows");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq4nl256_down_rows");
}

bool iq512_supported(int) noexcept { return false; }
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512_gu_rows");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512_rows");
}

}  // namespace strata::kernels::cpu
