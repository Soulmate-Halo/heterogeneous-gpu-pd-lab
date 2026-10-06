// src/kernels/cpu/q2_avx2.cpp - plan v0.3 P6: the Q2_0 expert rows and the activation quantizer for CPUs
// without AVX-512 (Intel Core 12th-14th gen and Core Ultra, AMD Zen 2/3).
//
// Compiled with AVX2 only, so nothing here can fault on those CPUs.  The arithmetic is the AVX-512 kernels':
// codes 0..3 against the int8 activation per 32-value chunk, times the weight scale and the chunk scale, minus
// the weight scale times the chunk's `hx` (the -1 code offset); the quantizer is the scalar rule, bit for bit.
#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

// 16 bytes of 2-bit codes (value i in byte i/4, bits 2*(i%4)) -> 64 codes in value order, two 32-byte vectors
inline void unpack64(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);   // bytes 0..7
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);   // bytes 8..15
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));  // values 0..31
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));  // values 32..63
}

template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            const __m256i s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i*) q)), ones);
            const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i*) (q + 32))), ones);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[t];
    }
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), zero = _mm256_setzero_ps();
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        __m256 v[4];
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(xb + 8 * i);
            m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m256 vinv = _mm256_set1_ps(inv);
        __m256i sum = _mm256_setzero_si256();
        alignas(32) int32_t qi[QKA];
        for (int i = 0; i < 4; ++i) {
            const __m256 t = _mm256_mul_ps(v[i], vinv);
            const __m256 r = _mm256_add_ps(t, _mm256_blendv_ps(mhalf, half, _mm256_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m256i q = _mm256_cvttps_epi32(r);
            q = _mm256_min_epi32(_mm256_max_epi32(q, lo), hi);
            sum = _mm256_add_epi32(sum, q);
            _mm256_store_si256((__m256i*) (qi + 8 * i), q);
        }
        for (int j = 0; j < QKA; ++j) a.q[k * QKA + j] = (int8_t) qi[j];
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int32_t total = _mm_cvtsi128_si32(s4);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
}

}  // namespace strata::kernels::cpu
