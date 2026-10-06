// src/kernels/cuda/nvfp4_marlin_expert.cu - P2: the NVFP4 hot-expert hit path.  Read the header first.
//
// The marlin call pattern below replicates tests/test_marlin_nvfp4.cu exactly (P2前半, bit-exact against
// the production JIT): same template arguments, same scalar parameters, same c_tmp sizing, same
// fp32-reduce/locks protocol.  Only the tensor SHAPES and the routing buffers differ, and the routing is
// the degenerate one-token-per-expert case described in the header.
#include "strata/kernels/nvfp4_marlin_expert.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <tvm/ffi/extra/c_env_api.h>

#include "moe_wna16_marlin.cuh"

namespace strata::kernels {
namespace {

// ---- scratch carve (host arithmetic shared by the sizing query and both entry points) ----
struct Lay {
    uint64_t a = 0, c1 = 0, act = 0, c2 = 0, sorted = 0, experts = 0, numpost = 0, count = 0, topkw = 0,
             locks = 0, ctmp1 = 0, ctmp2 = 0, dst = 0, tok = 0, total = 0;
};

uint64_t align256(uint64_t x) { return (x + 255) & ~uint64_t{255}; }

Lay layout_of(int64_t cap, int sms, bool batch = false) {
    Lay L;
    uint64_t off = 0;
    L.a = off;       off = align256(off + (uint64_t) NV4_H * 2 * (batch ? cap : 1));
    L.c1 = off;      off = align256(off + (uint64_t) cap * (2 * NV4_FF) * 2);
    L.act = off;     off = align256(off + (uint64_t) cap * NV4_FF * 2);
    L.c2 = off;      off = align256(off + (uint64_t) cap * NV4_H * 2);
    L.sorted = off;  off = align256(off + (uint64_t) cap * NV4_BLOCKM * 4);
    L.experts = off; off = align256(off + (uint64_t) cap * 4);
    L.numpost = off; off = align256(off + 4);
    L.count = off;   off = align256(off + 4);
    L.topkw = off;   off = align256(off + (uint64_t) cap * 4);
    L.locks = off;   off = align256(off + (uint64_t) sms * 4 * 4);
    // P2前半's c_tmp rule: min(size_n * sorted_len, sms * 4 * BLOCKM * 256) f32 per GEMM.
    const uint64_t sorted_len = (uint64_t) cap * NV4_BLOCKM;
    const uint64_t lock_cap = (uint64_t) sms * 4 * NV4_BLOCKM * 256;
    const uint64_t n1 = std::min((uint64_t) (2 * NV4_FF) * sorted_len, lock_cap);
    const uint64_t n2 = std::min((uint64_t) NV4_H * sorted_len, lock_cap);
    L.ctmp1 = off;   off = align256(off + n1 * 4);
    L.ctmp2 = off;   off = align256(off + n2 * 4);
    if (batch) {
        L.dst = off; off = align256(off + (uint64_t) cap * 4);
        L.tok = off; off = align256(off + (uint64_t) cap * 4);
    }
    L.total = off;
    return L;
}

void check_cap(int64_t cap) {
    if (cap < 1 || cap > 128) {
        std::fprintf(stderr, "moe_hit_grouped_nvfp4: cap %lld outside 1..128\n", (long long) cap);
        std::exit(1);
    }
}

// ---- decode-push pinned staging (the nvfp4_push_expand / staged-run pair).  The batch branch's block
// expansion below reads pageable host vectors and stack scalars, which CUDA stream capture forbids; the
// doorbell's graph path therefore expands into this process-persistent PINNED staging first, so every H2D
// source address is fixed and capture-legal.  Doorbell-thread only; 128 rows covers k_cap <= 32 with
// headroom.  Alloc failure disables the staged path (the caller falls back to the direct batch entry). ----
constexpr int32_t NV4_PUSH_STAGE_ROWS = 128;
struct PushPin {
    int32_t* hs = nullptr;   // NV4_PUSH_STAGE_ROWS * NV4_BLOCKM sorted slots (padded blocks)
    int32_t* he = nullptr;   // NV4_PUSH_STAGE_ROWS expert ids, one per block
    int32_t* dst = nullptr;  // NV4_PUSH_STAGE_ROWS wire-order row indices
    int32_t* tok = nullptr;  // NV4_PUSH_STAGE_ROWS token index per row
    int32_t* sc = nullptr;   // [0] = num_post (padded rows), [1] = n (= cap)
};

PushPin& push_pin() {
    static PushPin pin;
    static std::once_flag once;
    std::call_once(once, [] {
        PushPin& p = pin;
        const auto alloc = [](int32_t** q, size_t n) {
            return cudaHostAlloc((void**) q, n * sizeof(int32_t), cudaHostAllocDefault) == cudaSuccess;
        };
        if (!alloc(&p.hs, (size_t) NV4_PUSH_STAGE_ROWS * NV4_BLOCKM) ||
            !alloc(&p.he, (size_t) NV4_PUSH_STAGE_ROWS) || !alloc(&p.dst, (size_t) NV4_PUSH_STAGE_ROWS) ||
            !alloc(&p.tok, (size_t) NV4_PUSH_STAGE_ROWS) || !alloc(&p.sc, 2))
            std::fprintf(stderr, "nvfp4_push staging: cudaHostAlloc failed - staged path disabled\n");
    });
    return pin;
}

bool push_pin_ready(const PushPin& p) { return p.hs && p.he && p.dst && p.tok && p.sc; }

// ---- the small device kernels around the two marlin GEMMs ----

__global__ void nv4_cvt_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = __float2bfloat16(x[i]);
}

__global__ void nv4_gather_kernel(const float* x, __nv_bfloat16* a, const int32_t* tok,
                                  int rows, float* topkw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) topkw[i] = 1.0f;
    if (i < rows * NV4_H)
        a[i] = __float2bfloat16(x[(int64_t) tok[i / NV4_H] * NV4_H + i % NV4_H]);
}

// One block of BM sorted slots per hit: block b holds hit b alone (hits are distinct experts), so the
// vLLM-style routing collapses to `sorted[b*BM] = b`, padding = cap (= size_m*top_k for BOTH GEMMs:
// gemm1 runs size_m=1/top_k=cap, gemm2 size_m=cap/top_k=1), expert_ids[b] = slot[b], -1 past the count
// (the kernel's own skip), num_post = count*BM.
__global__ void nv4_prep_kernel(const int32_t* __restrict__ slot, const int32_t* __restrict__ count, int cap,
                                int32_t* __restrict__ sorted, int32_t* __restrict__ experts,
                                int32_t* __restrict__ numpost) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = *count;
    if (idx == 0) numpost[0] = n * NV4_BLOCKM;
    if (idx >= cap * NV4_BLOCKM) return;
    const int b = idx / NV4_BLOCKM, j = idx % NV4_BLOCKM;
    sorted[idx] = (b < n && j == 0) ? b : cap;
    if (j == 0) experts[b] = (b < n) ? slot[b] : -1;
}

__global__ void nv4_silu_kernel(const __nv_bfloat16* __restrict__ c1, __nv_bfloat16* __restrict__ act,
                                int rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * NV4_FF) return;
    const int r = idx / NV4_FF, j = idx % NV4_FF;
    const float g = __bfloat162float(c1[r * 2 * NV4_FF + j]);
    const float u = __bfloat162float(c1[r * 2 * NV4_FF + NV4_FF + j]);
    act[idx] = __float2bfloat16(g / (1.f + expf(-g)) * u);
}

// Same write pattern as the s2 down_kernel: hit h's row lands at hit_out[dst[h]] - what moe_hit_add reads.
__global__ void nv4_scatter_kernel(const __nv_bfloat16* __restrict__ c2, const int32_t* __restrict__ dst,
                                   const int32_t* __restrict__ count, float* __restrict__ hit_out) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const int64_t row = (int64_t) dst[h] * NV4_H;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NV4_H; i += gridDim.x * blockDim.x)
        hit_out[row + i] = __bfloat162float(c2[(int64_t) h * NV4_H + i]);
}

// ---- tvm TensorView plumbing (tests/test_marlin_nvfp4.cu's pattern: stack DLTensor, contiguous) ----
struct View {
    void* dptr;
    DLDataType dtype;
    int64_t shape[3];
    int ndim;
};
constexpr DLDataType kBf16{(uint8_t) 4, 16, 1}, kF32{(uint8_t) 2, 32, 1}, kI32{(uint8_t) 0, 32, 1},
    kF8{(uint8_t) 8, 8, 1};

tvm::ffi::TensorView tv(View& v, int dev) {
    DLTensor dl;
    dl.data = v.dptr;
    dl.device = {kDLCUDA, dev};
    dl.ndim = v.ndim;
    dl.dtype = v.dtype;
    dl.shape = v.shape;
    dl.strides = nullptr;
    dl.byte_offset = 0;
    return tvm::ffi::TensorView(&dl);
}

uint32_t read_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
uint16_t read_u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
float read_f32(const uint8_t* p) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
}

void run(const Nvfp4HotArena& ar, const int32_t* d_slot, const int32_t* d_dst, const int32_t* d_count,
         int64_t cap, const float* x_f, void* scratch, float* hit_out, cudaStream_t cs,
         const int32_t* h_slot = nullptr, const int32_t* h_dst = nullptr, const int32_t* h_tok = nullptr,
         bool staged = false) {
    // P1②: the TVM env stream set below is PROCESS-GLOBAL - the in-process cold-expert doorbell thread
    // (cold_expert_host.cpp) runs this entry concurrently with the engine's own prefill/decode calls, and
    // an interleaved TVMFFIEnvSetStream would launch a caller's GEMMs on the other caller's stream, where
    // they would race that caller's scratch.  Serialize the whole set/launch/restore window; the hold is a
    // few kernel launches of CPU time and changes no numerics.
    static std::mutex launch_mu;
    const std::lock_guard<std::mutex> launch_lock(launch_mu);
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const bool batch = staged || h_slot != nullptr;
    int nblk_stream = 0;
    const Lay L = layout_of(cap, sms, batch);
    char* s = (char*) scratch;
    __nv_bfloat16* a = (__nv_bfloat16*) (s + L.a);
    __nv_bfloat16* c1 = (__nv_bfloat16*) (s + L.c1);
    __nv_bfloat16* act = (__nv_bfloat16*) (s + L.act);
    __nv_bfloat16* c2 = (__nv_bfloat16*) (s + L.c2);
    int32_t* sorted = (int32_t*) (s + L.sorted);
    int32_t* experts = (int32_t*) (s + L.experts);
    int32_t* numpost = (int32_t*) (s + L.numpost);
    float* topkw = (float*) (s + L.topkw);
    int32_t* locks = (int32_t*) (s + L.locks);
    float* ctmp1 = (float*) (s + L.ctmp1);
    float* ctmp2 = (float*) (s + L.ctmp2);

    if (batch) {
        // Staged mode (the cold-expert doorbell's CUDA-graph path): nvfp4_push_expand has already run the
        // block expansion into the process-persistent pinned staging, so every H2D below reads a fixed
        // pinned address and the whole sequence is stream-capture legal.  The values, their order, the
        // copy sizes and every kernel are identical to the direct path - only the source addresses differ.
        std::vector<int32_t> hs, he;
        const int32_t* hs_p = nullptr;
        const int32_t* he_p = nullptr;
        const int32_t* dst_p = h_dst;
        const int32_t* tok_p = h_tok;
        const void* post_p = nullptr;
        const void* n_p = nullptr;
        const int32_t n = (int32_t) cap;
        int32_t post = 0;
        size_t nblk = 0;
        if (staged) {
            PushPin& pin = push_pin();
            hs_p = pin.hs;
            he_p = pin.he;
            dst_p = pin.dst;
            tok_p = pin.tok;
            post_p = &pin.sc[0];
            n_p = &pin.sc[1];
            nblk = (size_t) pin.sc[0] / NV4_BLOCKM;
        } else {
            hs.reserve((size_t) cap * NV4_BLOCKM);
            he.reserve((size_t) cap);
            for (int b = 0; b < cap;) {
                const int slot = h_slot[b];
                int end = b + 1;
                while (end < cap && h_slot[end] == slot) ++end;
                for (int p = b; p < end; p += NV4_BLOCKM) {
                    he.push_back(slot);
                    for (int j = 0; j < NV4_BLOCKM; ++j) hs.push_back(p + j < end ? p + j : (int) cap);
                }
                b = end;
            }
            post = (int32_t) hs.size();
            hs_p = hs.data();
            he_p = he.data();
            post_p = &post;
            n_p = &n;
            nblk = he.size();
        }
        nblk_stream = static_cast<int>(nblk);
        d_dst = (int32_t*) (s + L.dst);
        d_count = (int32_t*) (s + L.count);
        auto* d_tok = (int32_t*) (s + L.tok);
        cudaMemcpyAsync(sorted, hs_p, nblk * NV4_BLOCKM * 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync(experts, he_p, nblk * 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync(numpost, post_p, 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync((void*) d_count, n_p, 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync((void*) d_dst, dst_p, (size_t) cap * 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync(d_tok, tok_p, (size_t) cap * 4, cudaMemcpyHostToDevice, cs);
        cudaMemsetAsync(locks, 0, (size_t) sms * 16, cs);
        nv4_gather_kernel<<<(unsigned) ((cap * NV4_H + 255) / 256), 256, 0, cs>>>(
            x_f, a, d_tok, (int) cap, topkw);
    } else {
        nv4_cvt_kernel<<<(NV4_H + 255) / 256, 256, 0, cs>>>(x_f, a, NV4_H);
        nv4_prep_kernel<<<(unsigned) ((cap * NV4_BLOCKM + 255) / 256), 256, 0, cs>>>(d_slot, d_count, (int) cap,
                                                                                     sorted, experts, numpost);
    }

    const int64_t E = ar.n_slots;
    View v_a{a, kBf16, {batch ? cap : 1, NV4_H, 0}, 2};
    View v_c1{c1, kBf16, {cap, 2 * NV4_FF, 0}, 2};
    View v_act{act, kBf16, {cap, NV4_FF, 0}, 2};
    View v_c2{c2, kBf16, {cap, NV4_H, 0}, 2};
    // marlin repacked int32 planes: [E, size_k/16, size_n*16/8] (the pack's exact byte counts).
    View v_w13{(void*) ar.w13, kI32, {E, NV4_H / 16, (2 * NV4_FF) * 2}, 3};
    View v_w2{(void*) ar.w2, kI32, {E, NV4_FF / 16, NV4_H * 2}, 3};
    View v_w13s{(void*) ar.w13s, kF8, {E, NV4_H / NV4_GROUP, 2 * NV4_FF}, 3};
    View v_w2s{(void*) ar.w2s, kF8, {E, NV4_FF / NV4_GROUP, NV4_H}, 3};
    View v_gm13{(void*) ar.gm13, kBf16, {E, 1, 0}, 2};
    View v_gm2{(void*) ar.gm2, kBf16, {E, 1, 0}, 2};
    View v_ws{locks, kI32, {sms * 4, 0, 0}, 1};
    View v_sorted{sorted, kI32, {cap * NV4_BLOCKM, 0, 0}, 1};
    View v_experts{experts, kI32, {cap, 0, 0}, 1};
    View v_numpost{numpost, kI32, {1, 0, 0}, 1};
    View v_topkw{topkw, kF32, {cap, 0, 0}, 1};
    const uint64_t n1 = (uint64_t) ((s + L.ctmp2) - (s + L.ctmp1)) / 4;   // aligned region sizes
    const uint64_t n2 = ((batch ? L.dst : L.total) - L.ctmp2) / 4;
    View v_ctmp1{ctmp1, kF32, {(int64_t) n1, 0, 0}, 1};
    View v_ctmp2{ctmp2, kF32, {(int64_t) n2, 0, 0}, 1};
    // Empty stand-ins (bias/zeros/g_idx/perm/a_tmp): a valid device pointer with a zero-length shape.
    View v_bias{s, kBf16, {0, 0, 0}, 1};
    View v_zeros{s, kBf16, {0, 0, 0}, 1};
    View v_gidx{s, kI32, {0, 0, 0}, 1};
    View v_perm{s, kI32, {0, 0, 0}, 1};
    View v_atmp{s, kBf16, {0, 0, 0}, 1};

    const auto qt = sglang::host::kFE2M1f.id();
    // Batch and cap<=8 are both capture-time shape quantities (the ch0 graph key is ne*64+post/16).
    // The branch is not data-dependent: cap>8 and the non-batch path keep the marlin launches verbatim.
    const bool stream_small = batch && cap <= 8 && nblk_stream > 0 &&
                              static_cast<int64_t>(cap) * (2 * NV4_FF) <= static_cast<int64_t>(n1) &&
                              static_cast<int64_t>(cap) * NV4_H <= static_cast<int64_t>(n2);
    if (stream_small) {
        nvfp4_stream_expert_gemm(a, c1, ar.w13, ar.w13s, ar.gm13, sorted, experts, topkw, ctmp1,
                                 static_cast<int64_t>(n1), static_cast<int>(cap), nblk_stream, 2 * NV4_FF, NV4_H,
                                 /*mul_topk=*/0, cs);
        nv4_silu_kernel<<<(unsigned) ((cap * NV4_FF + 255) / 256), 256, 0, cs>>>(c1, act, (int) cap);
        nvfp4_stream_expert_gemm(act, c2, ar.w2, ar.w2s, ar.gm2, sorted, experts, topkw, ctmp2,
                                 static_cast<int64_t>(n2), static_cast<int>(cap), nblk_stream, NV4_H, NV4_FF,
                                 /*mul_topk=*/1, cs);
    } else {
        // The marlin host entry resolves its launch stream ONCE from the tvm env: aim it at our stream so the
        // whole sequence is ordered with the engine and capturable, then put the env back.
        TVMFFIStreamHandle prev = nullptr;
        TVMFFIEnvSetStream(kDLCUDA, dev, (TVMFFIStreamHandle) cs, &prev);

        sglang::moe_wna16_marlin_gemm<nv_bfloat16, false, false>(
            tv(v_a, dev), tv(v_c1, dev), tv(v_w13, dev), tv(v_bias, dev), tv(v_w13s, dev), tv(v_gm13, dev),
            tv(v_zeros, dev), tv(v_gidx, dev), tv(v_perm, dev), tv(v_ws, dev), tv(v_sorted, dev),
            tv(v_experts, dev), tv(v_numpost, dev), tv(v_topkw, dev), tv(v_atmp, dev), tv(v_ctmp1, dev),
            NV4_BLOCKM, batch ? 1 : (int) cap, /*mul_topk_weights=*/false, /*is_ep=*/false, qt,
            /*size_m=*/batch ? cap : 1, /*size_n=*/2 * NV4_FF, /*size_k=*/NV4_H,
            /*has_act_order=*/false, /*has_bias=*/false, /*is_k_full=*/true, /*has_zp=*/false,
            /*num_groups=*/NV4_H / NV4_GROUP, /*group_size=*/NV4_GROUP,
            /*use_atomic_add=*/false, /*use_fp32_reduce=*/true, /*is_zp_float=*/false);

        nv4_silu_kernel<<<(unsigned) ((cap * NV4_FF + 255) / 256), 256, 0, cs>>>(c1, act, (int) cap);

        sglang::moe_wna16_marlin_gemm<nv_bfloat16, false, false>(
            tv(v_act, dev), tv(v_c2, dev), tv(v_w2, dev), tv(v_bias, dev), tv(v_w2s, dev), tv(v_gm2, dev),
            tv(v_zeros, dev), tv(v_gidx, dev), tv(v_perm, dev), tv(v_ws, dev), tv(v_sorted, dev),
            tv(v_experts, dev), tv(v_numpost, dev), tv(v_topkw, dev), tv(v_atmp, dev), tv(v_ctmp2, dev),
            NV4_BLOCKM, /*top_k=*/1, /*mul_topk_weights=*/true, /*is_ep=*/false, qt,
            /*size_m=*/cap, /*size_n=*/NV4_H, /*size_k=*/NV4_FF,
            /*has_act_order=*/false, /*has_bias=*/false, /*is_k_full=*/true, /*has_zp=*/false,
            /*num_groups=*/NV4_FF / NV4_GROUP, /*group_size=*/NV4_GROUP,
            /*use_atomic_add=*/false, /*use_fp32_reduce=*/true, /*is_zp_float=*/false);

        TVMFFIEnvSetStream(kDLCUDA, dev, prev, nullptr);
    }

    nv4_scatter_kernel<<<dim3(4, (unsigned) cap), 256, 0, cs>>>(c2, d_dst, d_count, hit_out);
}

}  // namespace

bool nvfp4_hot_plane_views(const uint8_t* blob, const uint8_t* planes[4], size_t sizes[4], int64_t layer,
                           int64_t expert) {
    if (blob == nullptr || planes == nullptr || sizes == nullptr) return false;
    if (std::memcmp(blob, "STRNVFP4", 8) != 0) return false;
    if (read_u16(blob + 8) != 2) return false;          // blob version
    if (read_u16(blob + 10) != 1) return false;         // flavor: hot
    if (layer >= 0 && (int64_t) read_u32(blob + 12) != layer) return false;
    if (expert >= 0 && (int64_t) read_u32(blob + 16) != expert) return false;
    if (read_u32(blob + 20) != 4) return false;         // the hot flavor's four planes
    if (read_u32(blob + 24) != NV4_W13_BYTES || read_u32(blob + 28) != NV4_W2_BYTES ||
        read_u32(blob + 32) != NV4_W13S_BYTES || read_u32(blob + 36) != NV4_W2S_BYTES)
        return false;
    planes[0] = blob + NV4_O_W13;
    planes[1] = blob + NV4_O_W2;
    planes[2] = blob + NV4_O_W13S;
    planes[3] = blob + NV4_O_W2S;
    sizes[0] = NV4_W13_BYTES;
    sizes[1] = NV4_W2_BYTES;
    sizes[2] = NV4_W13S_BYTES;
    sizes[3] = NV4_W2S_BYTES;
    return true;
}

bool nvfp4_process_globals(const uint8_t* tail24, uint16_t& gm13, uint16_t& gm2) {
    if (tail24 == nullptr) return false;
    // <8sHHIIiI record: gate ws2, gate input_scale, up ws2, up input_scale, down ws2, down input_scale.
    const uint8_t* gate = tail24;
    const uint8_t* up = tail24 + 8;
    const uint8_t* down = tail24 + 16;
    if (read_u32(gate) != read_u32(up)) return false;   // the oracle stack's refusal, worker/kernel.py:61
    auto fold = [](const uint8_t* p, uint16_t& out) {
        uint32_t u = read_u32(p);
        // f32 -> bf16, round to nearest even (identical to torch's .to(bfloat16)).
        u += 0x7FFFu + ((u >> 16) & 1u);
        uint32_t hi = u >> 16;
        const uint32_t exp = (hi >> 7) & 0xFF;
        if (exp == 0 || exp == 0xFF) return false;      // zero/subnormal/inf/nan: not a legal ws2
        if (exp + 119 >= 0xFF) return false;            // sglang's * 2**119 must stay finite
        out = (uint16_t) (hi + (119u << 7));
        return true;
    };
    return fold(gate, gm13) && fold(down, gm2);
}

uint64_t moe_hit_nvfp4_scratch_bytes(int64_t cap, int sms) {
    check_cap(cap);
    if (sms <= 0) {
        std::fprintf(stderr, "moe_hit_nvfp4_scratch_bytes: sms must be positive\n");
        std::exit(1);
    }
    return layout_of(cap, sms).total;
}

void moe_hit_nvfp4_scratch_init(void* scratch, int64_t cap, int sms, void* stream) {
    check_cap(cap);
    const Lay L = layout_of(cap, sms);
    char* s = (char*) scratch;
    cudaStream_t cs = (cudaStream_t) stream;
    cudaMemsetAsync(s + L.locks, 0, (size_t) sms * 4 * 4, cs);
    cudaMemsetAsync(s + L.numpost, 0, 8, cs);   // num_post + the host-count cell
    static float ones[128];
    static bool filled = [] {
        for (int i = 0; i < 128; ++i) ones[i] = 1.0f;
        return true;
    }();
    (void) filled;
    cudaMemcpyAsync(s + L.topkw, ones, (size_t) cap * 4, cudaMemcpyHostToDevice, cs);
}

void moe_hit_grouped_nvfp4_dev(const Nvfp4HotArena& arena, const int32_t* d_slot, const int32_t* d_dst,
                               const int32_t* d_count, int64_t cap, const float* x_f, void* scratch,
                               float* hit_out, void* stream) {
    check_cap(cap);
    if (!arena.on() || d_slot == nullptr || d_dst == nullptr || d_count == nullptr || x_f == nullptr ||
        scratch == nullptr || hit_out == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_nvfp4_dev: incomplete configuration\n");
        std::exit(1);
    }
    run(arena, d_slot, d_dst, d_count, cap, x_f, scratch, hit_out, (cudaStream_t) stream);
}

void moe_hit_grouped_nvfp4(const Nvfp4HotArena& arena, const int32_t* d_slot, const int32_t* d_dst,
                           int64_t n_hits, int64_t cap, const float* x_f, void* scratch, float* hit_out,
                           void* stream) {
    check_cap(cap);
    if (!arena.on() || d_slot == nullptr || d_dst == nullptr || x_f == nullptr || scratch == nullptr ||
        hit_out == nullptr || n_hits < 0 || n_hits > cap) {
        std::fprintf(stderr, "moe_hit_grouped_nvfp4: incomplete configuration\n");
        std::exit(1);
    }
    cudaStream_t cs = (cudaStream_t) stream;
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const Lay L = layout_of(cap, sms);
    const int32_t c = (int32_t) n_hits;
    cudaMemcpyAsync((char*) scratch + L.count, &c, 4, cudaMemcpyHostToDevice, cs);
    run(arena, d_slot, d_dst, (const int32_t*) ((char*) scratch + L.count), cap, x_f, scratch, hit_out, cs);
}

uint64_t moe_prefill_nvfp4_scratch_bytes(int sms) {
    return layout_of(NV4_PREFILL_ROWS, sms, true).total;
}

void moe_prefill_grouped_nvfp4(const Nvfp4HotArena& arena, const int32_t* h_slot,
                              const int32_t* h_dst, const int32_t* h_tok, int n_rows,
                              const float* x_f, void* scratch, float* out, void* stream) {
    if (n_rows == 0) return;
    if (!arena.on() || !h_slot || !h_dst || !h_tok || !x_f || !scratch || !out ||
        n_rows < 0 || n_rows > NV4_PREFILL_ROWS) {
        std::fprintf(stderr, "moe_prefill_grouped_nvfp4: invalid configuration\n");
        std::exit(1);
    }
    for (int i = 0; i < n_rows; ++i) {
        if (h_slot[i] < 0 || h_slot[i] >= arena.n_slots || h_dst[i] < 0 || h_tok[i] < 0) {
            std::fprintf(stderr, "moe_prefill_grouped_nvfp4: invalid row %d\n", i);
            std::exit(1);
        }
    }
    run(arena, nullptr, nullptr, nullptr, n_rows, x_f, scratch, out, (cudaStream_t) stream,
        h_slot, h_dst, h_tok);
}

int32_t nvfp4_push_expand(const int32_t* h_slot, const int32_t* h_dst, const int32_t* h_tok, int32_t cap) {
    if (h_slot == nullptr || h_dst == nullptr || h_tok == nullptr || cap < 1 ||
        cap > NV4_PUSH_STAGE_ROWS)
        return -1;
    PushPin& pin = push_pin();
    if (!push_pin_ready(pin)) return -1;
    // The exact expansion of run()'s batch branch, into the pinned staging: one 16-row block per
    // 16-or-fewer rows of an expert's span, padding row index = cap.
    int32_t nblk = 0;
    for (int32_t b = 0; b < cap;) {
        const int32_t slot = h_slot[b];
        int32_t end = b + 1;
        while (end < cap && h_slot[end] == slot) ++end;
        for (int32_t p = b; p < end; p += NV4_BLOCKM) {
            pin.he[nblk] = slot;
            for (int32_t j = 0; j < NV4_BLOCKM; ++j)
                pin.hs[(size_t) nblk * NV4_BLOCKM + j] = p + j < end ? p + j : cap;
            ++nblk;
        }
        b = end;
    }
    std::memcpy(pin.dst, h_dst, (size_t) cap * 4);
    std::memcpy(pin.tok, h_tok, (size_t) cap * 4);
    pin.sc[0] = nblk * NV4_BLOCKM;
    pin.sc[1] = cap;
    return pin.sc[0];
}

void nvfp4_push_issue_staged(const Nvfp4HotArena& arena, int64_t cap, const float* x_f, void* scratch,
                             float* out, void* stream) {
    if (!arena.on() || x_f == nullptr || scratch == nullptr || out == nullptr || cap < 1 ||
        cap > NV4_PUSH_STAGE_ROWS || !push_pin_ready(push_pin())) {
        std::fprintf(stderr, "nvfp4_push_issue_staged: invalid configuration\n");
        std::exit(1);
    }
    run(arena, nullptr, nullptr, nullptr, cap, x_f, scratch, out, (cudaStream_t) stream, nullptr, nullptr,
        nullptr, true);
}

}  // namespace strata::kernels
