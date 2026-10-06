// src/kernels/cuda/kv_fp8.cu - see include/strata/kernels/kv_fp8.hpp.
#include "strata/kernels/kv_fp8.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_fp8: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_FP8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_fp8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_FP8_GROUP);
        std::exit(1);
    }
}

// One block = one 64-value group of one KV head of K (blockIdx.z = 0) or V (1); 64 threads, one value each.
__global__ void kv_append_fp8_kernel(uint8_t* __restrict__ k_fp8, uint8_t* __restrict__ v_fp8,
                                     uint16_t* __restrict__ k_scale, uint16_t* __restrict__ v_scale,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                     const float* __restrict__ kcur, const float* __restrict__ vcur, int kv_heads,
                                     int head_dim, int page_size, KvHostPools host) {
    const long long pos = (long long) __ldg(step + kStepPos);
    const int h = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const bool is_v = blockIdx.z == 1;
    const int groups = head_dim / KV_FP8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_FP8_GROUP + t];
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float warp_max[2];
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[0], warp_max[1]);
    const uint16_t sbits = f16_from_f32(amax / KV_FP8_EMAX);
    const float sf = f32_from_f16(sbits);                          // quantize against the STORED scale
    uint8_t q = 0;
    if (sf > 0.0f) q = kv_fp8_encode(x / sf);                      // scale 0: the group stays zero
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_fp8 : k_fp8)[row * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_fp8 != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_fp8 : host.k_fp8)[row * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? host.v_fp8_scale : host.k_fp8_scale)[row * groups + g] = sbits;
    }
}

// Prompt path: one block = one token, one KV head, one 64-group, K or V. blockIdx.z packs group and K/V.
__global__ void kv_append_fp8_batch_kernel(uint8_t* __restrict__ k_fp8, uint8_t* __restrict__ v_fp8,
                                           uint16_t* __restrict__ k_scale, uint16_t* __restrict__ v_scale,
                                           const int32_t* __restrict__ table, int64_t pos0, const float* __restrict__ K,
                                           const float* __restrict__ V, int kv_heads, int head_dim, int page_size,
                                           KvHostPools host, KvHostPools stage) {
    const int t = threadIdx.x;
    const int tok = (int) blockIdx.x, h = (int) blockIdx.y;
    const int g = (int) blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int groups = head_dim / KV_FP8_GROUP;
    const long long pos = pos0 + tok;
    const float x = (is_v ? V : K)[((long long) tok * kv_heads + h) * head_dim + g * KV_FP8_GROUP + t];
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float warp_max[2];
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[0], warp_max[1]);
    const uint16_t sbits = f16_from_f32(amax / KV_FP8_EMAX);
    const float sf = f32_from_f16(sbits);
    uint8_t q = 0;
    if (sf > 0.0f) q = kv_fp8_encode(x / sf);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_fp8 : k_fp8)[row * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (host.k_fp8 != nullptr) {
        (is_v ? host.v_fp8 : host.k_fp8)[row_id * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? host.v_fp8_scale : host.k_fp8_scale)[row_id * groups + g] = sbits;
    }
    if (stage.k_fp8 != nullptr) {
        (is_v ? stage.v_fp8 : stage.k_fp8)[row_id * head_dim + g * KV_FP8_GROUP + t] = q;
        if (t == 0) (is_v ? stage.v_fp8_scale : stage.k_fp8_scale)[row_id * groups + g] = sbits;
    }
}

// One thread = 4 consecutive values of one cell and head (as the FP16 gather does with uint2).
__global__ void kv_gather_fp8_kernel(const uint8_t* __restrict__ k_fp8, const uint8_t* __restrict__ v_fp8,
                                     const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                     const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                     uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_FP8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_FP8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_FP8_GROUP]);
    const uint32_t kc = *reinterpret_cast<const uint32_t*>(k_fp8 + row * head_dim + d);
    const uint32_t vc = *reinterpret_cast<const uint32_t*>(v_fp8 + row * head_dim + d);
    const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kc);
    const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vc);
    ushort4 ko, vo;
    ko.x = f16_from_f32(kv_fp8_decode(kb[0]) * ks); ko.y = f16_from_f32(kv_fp8_decode(kb[1]) * ks);
    ko.z = f16_from_f32(kv_fp8_decode(kb[2]) * ks); ko.w = f16_from_f32(kv_fp8_decode(kb[3]) * ks);
    vo.x = f16_from_f32(kv_fp8_decode(vb[0]) * vs); vo.y = f16_from_f32(kv_fp8_decode(vb[1]) * vs);
    vo.z = f16_from_f32(kv_fp8_decode(vb[2]) * vs); vo.w = f16_from_f32(kv_fp8_decode(vb[3]) * vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q4;
    reinterpret_cast<ushort4*>(k_scratch)[dst] = ko;
    reinterpret_cast<ushort4*>(v_scratch)[dst] = vo;
}

}  // namespace

void kv_append_fp8_step(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                        const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append_fp8");
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / KV_FP8_GROUP), 2);
    kv_append_fp8_kernel<<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
        k_fp8, v_fp8, k_scale, v_scale, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_fp8 launch");
}

void kv_append_fp8(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                   int64_t pos0, int64_t T, const float* K, const float* V, const QsaShapes& s, void* stream,
                   const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    validate(s, "kv_append_fp8 batch");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / KV_FP8_GROUP) * 2);
    kv_append_fp8_batch_kernel<<<grid, KV_FP8_GROUP, 0, (cudaStream_t) stream>>>(
        k_fp8, v_fp8, k_scale, v_scale, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{}, stage ? *stage : KvHostPools{});
    check("kv_append_fp8 batch launch");
}

void kv_gather_fp8_step(const uint8_t* k_fp8, const uint8_t* v_fp8, const uint16_t* k_scale, const uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                        const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_fp8");
    if (max_ids <= 0) return;
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    const unsigned blocks = (unsigned) ((total + 255) / 256);
    kv_gather_fp8_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(
        k_fp8, v_fp8, k_scale, v_scale, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, k_scratch, v_scratch);
    check("kv_gather_fp8 launch");
}

}  // namespace strata::kernels
