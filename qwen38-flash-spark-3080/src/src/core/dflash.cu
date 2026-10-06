// src/core/dflash.cu - DFlash block drafter. Weights stay BF16; projections go through the
// existing bf16 GEMV. Attention is non-causal over cat(context KV, this block's KV).
#include "strata/core/dflash.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace strata::core {
namespace {

constexpr int kH = 2560;
constexpr int kFcIn = 12800;
constexpr int kLayers = 5;
constexpr int kQ = 24;
constexpr int kKv = 2;
constexpr int kD = 256;
constexpr int kInter = 7680;
constexpr int kQDim = kQ * kD;    // 6144
constexpr int kKvDim = kKv * kD;  // 512
constexpr float kEps = 1e-6f;
constexpr double kTheta = 1.0e7;
constexpr int kSlots = strata::kernels::kVerifyMaxT;

bool launch_ok(const char* what, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int n) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (i < n) y[i] = __uint_as_float((uint32_t) x[i] << 16);
}

__global__ void pack_taps_kernel(const float* __restrict__ tap, float* __restrict__ packed, int n, int stride) {
    const int idx = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    const int total = n * kFcIn;
    if (idx >= total) return;
    const int t = idx / kFcIn;
    const int i = idx - t * kFcIn;
    const int k = i / kH;
    const int d = i - k * kH;
    packed[idx] = tap[((size_t) k * (size_t) stride + (size_t) t) * kH + d];
}

__global__ void silu_mul_kernel(float* __restrict__ gate, const float* __restrict__ up, int n) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void fill_pos_kernel(int32_t* pos, int n_tok, int n_heads, int pos0) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    const int rows = n_tok * n_heads;
    if (i < rows) pos[i] = pos0 + i / n_heads;
}

__global__ void store_kv_kernel(float* __restrict__ cache, const float* __restrict__ src, int n, int pos0) {
    const int i = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    const int total = n * kKvDim;
    if (i >= total) return;
    const int t = i / kKvDim;
    cache[((size_t) pos0 + (size_t) t) * kKvDim + (i - t * kKvDim)] = src[i];
}

__device__ __forceinline__ void mix_key(const float* qq, const float* kk, const float* vv, int lane, float* acc,
                                          float& m, float& l) {
    float partial = 0.0f;
    for (int d = lane; d < kD; d += 32) partial += qq[d] * kk[d];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, off);
    const float dot = partial * 0.0625f;  // 256^-0.5
    const float m2 = fmaxf(m, dot);
    const float alpha = __expf(m - m2);
    const float p = __expf(dot - m2);
    l = l * alpha + p;
    m = m2;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = acc[i] * alpha + p * vv[lane + i * 32];
}

// One warp per (query token, query head). Non-causal: every query sees context [k0, k1) and the whole block.
__global__ void attend_kernel(const float* __restrict__ q, const float* __restrict__ k_ctx,
                              const float* __restrict__ v_ctx, const float* __restrict__ k_blk,
                              const float* __restrict__ v_blk, float* __restrict__ out, int k0, int k1, int bq) {
    const int lane = threadIdx.x;
    const int qt = (int) blockIdx.x;
    if (qt >= bq * kQ) return;
    const int t = qt / kQ;
    const int h = qt - t * kQ;
    const int kv = h / (kQ / kKv);
    const float* qq = q + ((size_t) t * kQ + h) * kD;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float m = -1.0e30f;
    float l = 0.0f;
    for (int s = k0; s < k1; ++s)
        mix_key(qq, k_ctx + ((size_t) s * kKv + kv) * kD, v_ctx + ((size_t) s * kKv + kv) * kD, lane, acc, m, l);
    for (int s = 0; s < bq; ++s)
        mix_key(qq, k_blk + ((size_t) s * kKv + kv) * kD, v_blk + ((size_t) s * kKv + kv) * kD, lane, acc, m, l);
    float* oo = out + ((size_t) t * kQ + h) * kD;
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) oo[lane + i * 32] = acc[i] * inv;
}

bool parse_tensor(const std::string& json, const std::string& name, int64_t expect0, int64_t expect1,
                  int64_t& off, int64_t& bytes, std::string& err) {
    const std::string key = "\"" + name + "\"";
    const size_t at = json.find(key);
    if (at == std::string::npos) {
        err = "dflash: missing tensor " + name;
        return false;
    }
    const size_t shape = json.find("\"shape\"", at);
    const size_t next = json.find("\n\"", at + key.size());
    if (shape == std::string::npos || (next != std::string::npos && shape > next)) {
        err = "dflash: no shape for " + name;
        return false;
    }
    const size_t lb = json.find('[', shape);
    const size_t rb = json.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) {
        err = "dflash: bad shape for " + name;
        return false;
    }
    int64_t d0 = 0, d1 = 0;
    const std::string body = json.substr(lb + 1, rb - lb - 1);
    if (std::sscanf(body.c_str(), "%lld, %lld", (long long*) &d0, (long long*) &d1) != 2) {
        if (std::sscanf(body.c_str(), "%lld", (long long*) &d0) != 1) {
            err = "dflash: cannot parse shape of " + name;
            return false;
        }
        d1 = 0;
    }
    if (d0 != expect0 || d1 != expect1) {
        err = "dflash: unexpected shape of " + name;
        return false;
    }
    const size_t od = json.find("\"data_offsets\"", at);
    if (od == std::string::npos || (next != std::string::npos && od > next)) {
        err = "dflash: no offsets for " + name;
        return false;
    }
    const size_t olb = json.find('[', od);
    int64_t a = 0, b = 0;
    if (olb == std::string::npos ||
        std::sscanf(json.c_str() + olb, "[%lld, %lld]", (long long*) &a, (long long*) &b) != 2 || b < a) {
        err = "dflash: bad offsets for " + name;
        return false;
    }
    const int64_t n = d1 == 0 ? d0 : d0 * d1;
    if (b - a != n * 2) {
        err = "dflash: byte count mismatch for " + name;
        return false;
    }
    off = a;
    bytes = b - a;
    return true;
}

unsigned grid_n(int n, int threads) { return (unsigned) ((n + threads - 1) / threads); }

}  // namespace

DFlashDrafter::~DFlashDrafter() { release(); }

void DFlashDrafter::release() {
    if (device_ >= 0) {
        OnDevice on(device_);
        cudaFree(weights_);
        cudaFree(norms_);
        cudaFree(rope_cos_);
        cudaFree(rope_sin_);
        cudaFree(kv_);
        cudaFree(taps_);
        cudaFree(scratch_);
        cudaFree(logits_);
        cudaFree(xq_);
        if (stream_ != nullptr) cudaStreamDestroy((cudaStream_t) stream_);
    }
    weights_ = nullptr;
    norms_ = nullptr;
    rope_cos_ = nullptr;
    rope_sin_ = nullptr;
    kv_ = nullptr;
    taps_ = nullptr;
    scratch_ = nullptr;
    logits_ = nullptr;
    xq_ = nullptr;
    stream_ = nullptr;
    fc_ = nullptr;
    hidden_norm_ = nullptr;
    final_norm_ = nullptr;
    packed_ = noise_ = h_ = q_ = k_ = v_ = attn_ = gate_ = up_ = proj_ = nullptr;
    pos_q_ = pos_k_ = tok_ = out_ids_ = nullptr;
    embed_ = nullptr;
    head_ = nullptr;
    for (Layer& ly : layers_) ly = Layer{};
    bound_ = false;
    warned_late_ = false;
    ctx_begin_ = 0;
    ctx_end_ = 0;
    max_seq_ = 0;
    n_vocab_ = 0;
    weight_bytes_ = 0;
    kv_bytes_ = 0;
    scratch_bytes_ = 0;
    device_ = -1;
}

bool DFlashDrafter::gemv(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in,
                         int64_t n_out, int n, std::string& err) {
    try {
        if (n == 1) strata::kernels::bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream_);
        else strata::kernels::bf16_gemv_fp32_mmvf_multi(x, ldx, w, y, ldy, n_in, n_out, n, stream_);
    } catch (const std::exception& e) {
        err = std::string("dflash gemv: ") + e.what();
        return false;
    }
    return true;
}

bool DFlashDrafter::load(const std::string& path, int64_t max_seq, std::string& err) {
    release();
    if (max_seq < kDflashBlock + 2) {
        err = "dflash: --max-context is too small for a block of 7";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "dflash: cannot open " + path;
        return false;
    }
    uint64_t header_n = 0;
    in.read(reinterpret_cast<char*>(&header_n), 8);
    if (!in || header_n == 0 || header_n > 16 * 1024 * 1024) {
        err = "dflash: bad safetensors header";
        return false;
    }
    std::string json((size_t) header_n, '\0');
    in.read(json.data(), (std::streamsize) header_n);
    if (!in) {
        err = "dflash: truncated header";
        return false;
    }
    struct Item { const char* name; int64_t d0, d1; int64_t off, bytes; };
    std::vector<Item> items;
    auto add = [&](const std::string& name, int64_t d0, int64_t d1) -> bool {
        Item it{nullptr, d0, d1, 0, 0};
        if (!parse_tensor(json, name, d0, d1, it.off, it.bytes, err)) return false;
        items.push_back(it);
        return true;
    };
    if (!add("fc.weight", kH, kFcIn) || !add("hidden_norm.weight", kH, 0) || !add("norm.weight", kH, 0)) return false;
    for (int L = 0; L < kLayers; ++L) {
        const std::string pfx = "layers." + std::to_string(L) + ".";
        if (!add(pfx + "input_layernorm.weight", kH, 0) || !add(pfx + "post_attention_layernorm.weight", kH, 0) ||
            !add(pfx + "self_attn.q_proj.weight", kQDim, kH) || !add(pfx + "self_attn.k_proj.weight", kKvDim, kH) ||
            !add(pfx + "self_attn.v_proj.weight", kKvDim, kH) || !add(pfx + "self_attn.o_proj.weight", kH, kQDim) ||
            !add(pfx + "self_attn.q_norm.weight", kD, 0) || !add(pfx + "self_attn.k_norm.weight", kD, 0) ||
            !add(pfx + "mlp.gate_proj.weight", kInter, kH) || !add(pfx + "mlp.up_proj.weight", kInter, kH) ||
            !add(pfx + "mlp.down_proj.weight", kH, kInter))
            return false;
    }
    if ((int) items.size() != 58) {
        err = "dflash: expected 58 tensors";
        return false;
    }
    int64_t data_bytes = 0;
    for (const Item& it : items) data_bytes = std::max(data_bytes, it.off + it.bytes);
    in.seekg(0, std::ios::end);
    const std::streamoff file_end = in.tellg();
    const std::streamoff data_at = (std::streamoff) (8 + header_n);
    if (file_end < data_at + data_bytes) {
        err = "dflash: file is shorter than the tensor table";
        return false;
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        err = "dflash: no CUDA device";
        return false;
    }
    device_ = dev;
    OnDevice on(device_);
    if (cudaStreamCreateWithFlags((cudaStream_t*) &stream_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "dflash: cannot create a stream";
        release();
        return false;
    }
    auto fail = [&](const std::string& m) {
        err = m;
        release();
        return false;
    };
    if (cudaMalloc(&weights_, (size_t) data_bytes) != cudaSuccess) return fail("dflash: weight allocation failed");
    weight_bytes_ = (uint64_t) data_bytes;
    std::vector<char> chunk(8u << 20);
    in.clear();
    in.seekg(data_at);
    for (int64_t off = 0; off < data_bytes;) {
        const int64_t n = std::min<int64_t>((int64_t) chunk.size(), data_bytes - off);
        in.read(chunk.data(), (std::streamsize) n);
        if (!in) return fail("dflash: short read of weights");
        if (cudaMemcpyAsync(weights_ + off, chunk.data(), (size_t) n, cudaMemcpyHostToDevice, (cudaStream_t) stream_) !=
            cudaSuccess)
            return fail("dflash: weight upload failed");
        off += n;
    }
    auto locate = [&](const std::string& name, int64_t d0, int64_t d1) -> const uint16_t* {
        int64_t off = 0, bytes = 0;
        if (!parse_tensor(json, name, d0, d1, off, bytes, err)) return nullptr;
        return reinterpret_cast<const uint16_t*>(weights_ + off);
    };
    fc_ = locate("fc.weight", kH, kFcIn);
    const uint16_t* hn = locate("hidden_norm.weight", kH, 0);
    const uint16_t* fn = locate("norm.weight", kH, 0);
    if (fc_ == nullptr || hn == nullptr || fn == nullptr) return fail(err.empty() ? "dflash: fc/norm missing" : err);

    int64_t norm_floats = (int64_t) kH * 2;
    for (int L = 0; L < kLayers; ++L) norm_floats += (int64_t) kH * 2 + (int64_t) kD * 2;
    if (cudaMalloc(&norms_, (size_t) norm_floats * sizeof(float)) != cudaSuccess) return fail("dflash: norm allocation failed");
    float* nf = norms_;
    auto take_norm = [&](const uint16_t* src, int n) -> float* {
        float* dst = nf;
        nf += n;
        bf16_to_f32_kernel<<<grid_n(n, 128), 128, 0, (cudaStream_t) stream_>>>(src, dst, n);
        return dst;
    };
    hidden_norm_ = take_norm(hn, kH);
    final_norm_ = take_norm(fn, kH);
    for (int L = 0; L < kLayers; ++L) {
        const std::string pfx = "layers." + std::to_string(L) + ".";
        Layer& ly = layers_[L];
        ly.q = locate(pfx + "self_attn.q_proj.weight", kQDim, kH);
        ly.k = locate(pfx + "self_attn.k_proj.weight", kKvDim, kH);
        ly.v = locate(pfx + "self_attn.v_proj.weight", kKvDim, kH);
        ly.o = locate(pfx + "self_attn.o_proj.weight", kH, kQDim);
        ly.gate = locate(pfx + "mlp.gate_proj.weight", kInter, kH);
        ly.up = locate(pfx + "mlp.up_proj.weight", kInter, kH);
        ly.down = locate(pfx + "mlp.down_proj.weight", kH, kInter);
        const uint16_t* inw = locate(pfx + "input_layernorm.weight", kH, 0);
        const uint16_t* pw = locate(pfx + "post_attention_layernorm.weight", kH, 0);
        const uint16_t* qn = locate(pfx + "self_attn.q_norm.weight", kD, 0);
        const uint16_t* kn = locate(pfx + "self_attn.k_norm.weight", kD, 0);
        if (!ly.q || !ly.k || !ly.v || !ly.o || !ly.gate || !ly.up || !ly.down || !inw || !pw || !qn || !kn)
            return fail(err.empty() ? "dflash: a layer tensor is missing" : err);
        ly.in_norm = take_norm(inw, kH);
        ly.post_norm = take_norm(pw, kH);
        ly.q_norm = take_norm(qn, kD);
        ly.k_norm = take_norm(kn, kD);
    }
    if (!launch_ok("dflash norm convert", err)) return fail(err);

    const int half = kD / 2;
    std::vector<float> cos_h((size_t) max_seq * (size_t) half), sin_h((size_t) max_seq * (size_t) half);
    strata::kernels::build_rope_table(kD, kTheta, (int) max_seq, cos_h.data(), sin_h.data());
    const size_t rope_bytes = cos_h.size() * sizeof(float);
    if (cudaMalloc(&rope_cos_, rope_bytes) != cudaSuccess || cudaMalloc(&rope_sin_, rope_bytes) != cudaSuccess)
        return fail("dflash: RoPE table allocation failed");
    cudaMemcpyAsync(rope_cos_, cos_h.data(), rope_bytes, cudaMemcpyHostToDevice, (cudaStream_t) stream_);
    cudaMemcpyAsync(rope_sin_, sin_h.data(), rope_bytes, cudaMemcpyHostToDevice, (cudaStream_t) stream_);

    // K and V, per layer, at absolute positions: [layer][max_seq][kKvDim]
    kv_bytes_ = (uint64_t) kLayers * 2ull * (uint64_t) max_seq * (uint64_t) kKvDim * sizeof(float);
    if (cudaMalloc(&kv_, kv_bytes_) != cudaSuccess) return fail("dflash: KV allocation failed");
    cudaMemsetAsync(kv_, 0, kv_bytes_, (cudaStream_t) stream_);

    const size_t tap_floats = (size_t) kDflashTaps * (size_t) kSlots * (size_t) kH;
    if (cudaMalloc(&taps_, tap_floats * sizeof(float)) != cudaSuccess) return fail("dflash: tap buffer allocation failed");

    const size_t scratch_floats = (size_t) kSlots *
                                   ((size_t) kFcIn + (size_t) kH * 2 + (size_t) kQDim + (size_t) kKvDim * 2 +
                                    (size_t) kQDim + (size_t) kInter * 2 + (size_t) kH);
    scratch_bytes_ = scratch_floats * sizeof(float) + (size_t) kSlots * (kQ + kKv + 2) * sizeof(int32_t);
    if (cudaMalloc(&scratch_, scratch_bytes_) != cudaSuccess) return fail("dflash: scratch allocation failed");
    uint8_t* sp = scratch_;
    auto take_f = [&](size_t n) -> float* {
        float* p = reinterpret_cast<float*>(sp);
        sp += n * sizeof(float);
        return p;
    };
    packed_ = take_f((size_t) kSlots * kFcIn);
    noise_ = take_f((size_t) kSlots * kH);
    h_ = take_f((size_t) kSlots * kH);
    q_ = take_f((size_t) kSlots * kQDim);
    k_ = take_f((size_t) kSlots * kKvDim);
    v_ = take_f((size_t) kSlots * kKvDim);
    attn_ = take_f((size_t) kSlots * kQDim);
    gate_ = take_f((size_t) kSlots * kInter);
    up_ = take_f((size_t) kSlots * kInter);
    proj_ = take_f((size_t) kSlots * kH);
    pos_q_ = reinterpret_cast<int32_t*>(sp);
    sp += (size_t) kSlots * kQ * sizeof(int32_t);
    pos_k_ = reinterpret_cast<int32_t*>(sp);
    sp += (size_t) kSlots * kKv * sizeof(int32_t);
    tok_ = reinterpret_cast<int32_t*>(sp);
    sp += (size_t) kSlots * sizeof(int32_t);
    out_ids_ = reinterpret_cast<int32_t*>(sp);

    max_seq_ = max_seq;
    if (cudaStreamSynchronize((cudaStream_t) stream_) != cudaSuccess) return fail("dflash: weight upload did not finish");
    const double gib = 1073741824.0;
    const uint64_t side = (uint64_t) tap_floats * sizeof(float) + (uint64_t) rope_bytes * 2 + (uint64_t) norm_floats * 4;
    std::fprintf(stderr,
                 "strata dflash: weights %.3f GiB + KV %.3f GiB (%lld positions) + scratch/taps %.3f GiB\n",
                 (double) weight_bytes_ / gib, (double) kv_bytes_ / gib, (long long) max_seq_,
                 (double) (scratch_bytes_ + side) / gib);
    scratch_bytes_ += side;
    return true;
}

bool DFlashDrafter::bind(const NativeEmbed* embed, const NativeHead* head, const WeightTable* wt, int64_t n_vocab,
                         std::string& err) {
    if (weights_ == nullptr) {
        err = "dflash: bind before load";
        return false;
    }
    if (embed == nullptr && (wt == nullptr || wt->find("token_embd.weight") == nullptr)) {
        err = "dflash: the draft head shares the target embedding (pack token_embd.weight or --native)";
        return false;
    }
    if (head == nullptr || !head->loaded()) {
        err = "dflash: the draft head shares the target LM head (--native-head-gguf)";
        return false;
    }
    if (!strata::kernels::native_mmvq_supported(head->type())) {
        err = "dflash: the native head type is not supported by the multi-token MMVQ";
        return false;
    }
    if (n_vocab <= kDflashMaskId) {
        err = "dflash: vocabulary does not contain the mask token";
        return false;
    }
    OnDevice on(device_);
    cudaFree(logits_);
    logits_ = nullptr;
    cudaFree(xq_);
    xq_ = nullptr;
    const size_t logit_bytes = (size_t) kSlots * (size_t) n_vocab * sizeof(float);
    const size_t xq_bytes = strata::kernels::native_q8_1_bytes(kH, kSlots);
    if (cudaMalloc(&logits_, logit_bytes) != cudaSuccess || cudaMalloc(&xq_, xq_bytes) != cudaSuccess) {
        err = "dflash: logit allocation failed";
        cudaFree(logits_);
        logits_ = nullptr;
        cudaFree(xq_);
        xq_ = nullptr;
        return false;
    }
    embed_ = embed;
    head_ = head;
    wt_ = wt;
    n_vocab_ = n_vocab;
    bound_ = true;
    scratch_bytes_ += logit_bytes + xq_bytes;
    const double gib = 1073741824.0;
    std::fprintf(stderr, "strata dflash: LM-head scratch %.3f GiB (vocab %lld x %d). total GPU %.3f GiB\n",
                 (double) (logit_bytes + xq_bytes) / gib, (long long) n_vocab, kSlots,
                 (double) (weight_bytes_ + kv_bytes_ + scratch_bytes_) / gib);
    return true;
}

bool DFlashDrafter::kv_from_hidden(const float* h, int n, int64_t pos0, int layer, float* k_dst, float* v_dst,
                                   bool to_cache, std::string& err) {
    const Layer& ly = layers_[layer];
    float* kk = k_dst != nullptr ? k_dst : k_;
    float* vv = v_dst != nullptr ? v_dst : v_;
    if (!gemv(h, kH, ly.k, kk, kKvDim, kH, kKvDim, n, err)) return false;
    if (!gemv(h, kH, ly.v, vv, kKvDim, kH, kKvDim, n, err)) return false;
    strata::kernels::rms_norm_weighted(kk, ly.k_norm, (int64_t) n * kKv, kD, kEps, stream_);
    if (!launch_ok("dflash k-norm", err)) return false;
    fill_pos_kernel<<<grid_n(n * kKv, 128), 128, 0, (cudaStream_t) stream_>>>(pos_k_, n, kKv, (int) pos0);
    strata::kernels::rope_neox_apply(kk, kk, (int64_t) n * kKv, kD, kD, rope_cos_, rope_sin_, pos_k_, stream_);
    if (!launch_ok("dflash k rope", err)) return false;
    if (to_cache) {
        float* base = kv_ + ((size_t) layer * 2ull * (size_t) max_seq_) * kKvDim;
        float* kc = base;
        float* vc = base + (size_t) max_seq_ * kKvDim;
        store_kv_kernel<<<grid_n(n * kKvDim, 128), 128, 0, (cudaStream_t) stream_>>>(kc, kk, n, (int) pos0);
        store_kv_kernel<<<grid_n(n * kKvDim, 128), 128, 0, (cudaStream_t) stream_>>>(vc, vv, n, (int) pos0);
        if (!launch_ok("dflash kv store", err)) return false;
    }
    return true;
}

bool DFlashDrafter::ingest_window(int n, int64_t pos0, std::string& err) {
    if (!bound_) {
        err = "dflash: ingest before bind";
        return false;
    }
    if (n <= 0) return true;
    if (n > kSlots) {
        err = "dflash: a verify window handed the drafter more than 8 tap rows";
        return false;
    }
    if (pos0 < 0 || pos0 + n > max_seq_) {
        err = "dflash: tap positions run past --max-context";
        return false;
    }
    OnDevice on(device_);
    if (ctx_end_ == 0 && ctx_begin_ == 0 && pos0 > 0) {
        ctx_begin_ = pos0;
        ctx_end_ = pos0;
        if (!warned_late_) {
            warned_late_ = true;
            std::fprintf(stderr,
                         "strata dflash: context KV starts at position %lld (earlier tokens did not pass through "
                         "a verify window)\n",
                         (long long) pos0);
        }
    }
    if (pos0 < ctx_begin_) {
        err = "dflash: tap position is before the context KV";
        return false;
    }
    if (pos0 < ctx_end_) ctx_end_ = pos0;
    if (pos0 != ctx_end_) {
        err = "dflash: gap in context KV at position " + std::to_string(pos0);
        return false;
    }
    const cudaStream_t cs = (cudaStream_t) stream_;
    pack_taps_kernel<<<grid_n(n * kFcIn, 256), 256, 0, cs>>>(taps_, packed_, n, kSlots);
    if (!launch_ok("dflash pack taps", err)) return false;
    if (!gemv(packed_, kFcIn, fc_, h_, kH, kFcIn, kH, n, err)) return false;
    strata::kernels::rms_norm_weighted(h_, hidden_norm_, n, kH, kEps, stream_);
    if (!launch_ok("dflash hidden norm", err)) return false;
    for (int L = 0; L < kLayers; ++L)
        if (!kv_from_hidden(h_, n, pos0, L, nullptr, nullptr, true, err)) return false;
    ctx_end_ = pos0 + n;
    if (cudaStreamSynchronize(cs) != cudaSuccess) {
        err = "dflash: ingest sync failed";
        return false;
    }
    return true;
}

bool DFlashDrafter::draft(int32_t anchor, int64_t anchor_pos, int32_t* drafts, std::string& err) {
    if (!bound_) {
        err = "dflash: draft before bind";
        return false;
    }
    if (drafts == nullptr) {
        err = "dflash: null draft buffer";
        return false;
    }
    if (anchor_pos != ctx_end_) {
        err = "dflash: anchor position " + std::to_string(anchor_pos) + " is not the end of context KV (" +
              std::to_string(ctx_end_) + ")";
        return false;
    }
    if (anchor_pos < 0 || anchor_pos + kDflashBlock > max_seq_) {
        err = "dflash: the draft block runs past --max-context";
        return false;
    }
    OnDevice on(device_);
    const cudaStream_t cs = (cudaStream_t) stream_;
    int32_t host_tok[kDflashBlock];
    host_tok[0] = anchor;
    for (int i = 1; i < kDflashBlock; ++i) host_tok[i] = kDflashMaskId;
    if (cudaMemcpyAsync(tok_, host_tok, sizeof host_tok, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
        err = "dflash: token upload failed";
        return false;
    }
    if (embed_ != nullptr) {
        embed_->gather_dev(tok_, kDflashBlock, noise_, stream_);
    } else {
        // NVFP4 line: gather from the pack's on-device embedding table, the same way MTP does.
        const WeightRef* we = wt_->find("token_embd.weight");
        const auto* codes = (const uint8_t*) we->data;
        const auto* scales = (const float*) (codes + we->codes_bytes);
        const float* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
        strata::kernels::embedding_gather_dev(codes, scales, offsets, tok_, kDflashBlock, we->ne0, we->code_bits,
                                              we->code_bias, we->group_elems,
                                              (uint64_t) (we->ne0 / (8 / we->code_bits)),
                                              (uint64_t) (we->ne0 / we->group_elems), noise_, stream_);
    }
    if (!launch_ok("dflash embed", err)) return false;
    const int k0 = (int) ctx_begin_;
    const int k1 = (int) ctx_end_;
    for (int L = 0; L < kLayers; ++L) {
        const Layer& ly = layers_[L];
        if (cudaMemcpyAsync(h_, noise_, (size_t) kDflashBlock * kH * sizeof(float), cudaMemcpyDeviceToDevice, cs) !=
            cudaSuccess) {
            err = "dflash: residual copy failed";
            return false;
        }
        strata::kernels::rms_norm_weighted(h_, ly.in_norm, kDflashBlock, kH, kEps, stream_);
        if (!gemv(h_, kH, ly.q, q_, kQDim, kH, kQDim, kDflashBlock, err)) return false;
        strata::kernels::rms_norm_weighted(q_, ly.q_norm, (int64_t) kDflashBlock * kQ, kD, kEps, stream_);
        fill_pos_kernel<<<grid_n(kDflashBlock * kQ, 128), 128, 0, cs>>>(pos_q_, kDflashBlock, kQ, (int) anchor_pos);
        strata::kernels::rope_neox_apply(q_, q_, (int64_t) kDflashBlock * kQ, kD, kD, rope_cos_, rope_sin_, pos_q_,
                                         stream_);
        if (!launch_ok("dflash q rope", err)) return false;
        if (!kv_from_hidden(h_, kDflashBlock, anchor_pos, L, k_, v_, false, err)) return false;
        const float* kbase = kv_ + ((size_t) L * 2ull * (size_t) max_seq_) * kKvDim;
        const float* vbase = kbase + (size_t) max_seq_ * kKvDim;
        attend_kernel<<<kDflashBlock * kQ, 32, 0, cs>>>(q_, kbase, vbase, k_, v_, attn_, k0, k1, kDflashBlock);
        if (!launch_ok("dflash attend", err)) return false;
        if (!gemv(attn_, kQDim, ly.o, proj_, kH, kQDim, kH, kDflashBlock, err)) return false;
        add_kernel<<<grid_n(kDflashBlock * kH, 256), 256, 0, cs>>>(noise_, proj_, kDflashBlock * kH);
        if (cudaMemcpyAsync(h_, noise_, (size_t) kDflashBlock * kH * sizeof(float), cudaMemcpyDeviceToDevice, cs) !=
            cudaSuccess) {
            err = "dflash: mlp copy failed";
            return false;
        }
        strata::kernels::rms_norm_weighted(h_, ly.post_norm, kDflashBlock, kH, kEps, stream_);
        if (!gemv(h_, kH, ly.gate, gate_, kInter, kH, kInter, kDflashBlock, err)) return false;
        if (!gemv(h_, kH, ly.up, up_, kInter, kH, kInter, kDflashBlock, err)) return false;
        silu_mul_kernel<<<grid_n(kDflashBlock * kInter, 256), 256, 0, cs>>>(gate_, up_, kDflashBlock * kInter);
        if (!gemv(gate_, kInter, ly.down, proj_, kH, kInter, kH, kDflashBlock, err)) return false;
        add_kernel<<<grid_n(kDflashBlock * kH, 256), 256, 0, cs>>>(noise_, proj_, kDflashBlock * kH);
        if (!launch_ok("dflash layer", err)) return false;
    }
    if (cudaMemcpyAsync(h_, noise_, (size_t) kDflashBlock * kH * sizeof(float), cudaMemcpyDeviceToDevice, cs) !=
        cudaSuccess) {
        err = "dflash: final norm copy failed";
        return false;
    }
    strata::kernels::rms_norm_weighted(h_, final_norm_, kDflashBlock, kH, kEps, stream_);
    try {
        strata::kernels::native_quantize_q8_1(h_, xq_, kH, kDflashBlock, stream_);
        strata::kernels::native_mmvq(head_->type(), head_->weights(), xq_, logits_, kH, (int) n_vocab_, kDflashBlock,
                                     stream_);
    } catch (const std::exception& e) {
        err = std::string("dflash head: ") + e.what();
        return false;
    }
    strata::kernels::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.0f;
    strata::kernels::sample_tokens(logits_, kDflashBlock, (int) n_vocab_, nullptr, 0, sp, out_ids_, stream_);
    if (cudaMemcpyAsync(drafts, out_ids_, (size_t) kDflashBlock * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) !=
            cudaSuccess ||
        cudaStreamSynchronize(cs) != cudaSuccess) {
        err = "dflash: reading the drafts back failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
