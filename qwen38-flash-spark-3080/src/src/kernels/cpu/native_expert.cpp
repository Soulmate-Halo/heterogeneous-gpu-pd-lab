// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    static const bool avx2 = std::getenv("STRATA_NO_IQ256") == nullptr;
    if (nt >= 2 && iq512_supported(f.gu_type)) {   // one token: ggml-cpu is as fast or faster
        if (avx512) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    if (nt >= 2 && f.d_type == 20 && iq4nl_mt) {
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
