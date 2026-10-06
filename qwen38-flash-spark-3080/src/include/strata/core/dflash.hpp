// include/strata/core/dflash.hpp - DFlash block-diffusion draft head (speculative decode).
//
// Off unless load() succeeds. The verify window's captured graph gains tap copies only after
// Verifier::set_dflash_taps; with a null sink the graph is unchanged.
//
// Taps are the attention-half gated-residual mix (native width 2560) of layers
// kDflashCaptureLayers = trained target ids {3,15,23,35,43} + 1. Block length is 7:
// position 0 is the anchor token, positions 1..6 are the mask id, and every position is
// greedy (sample_from_anchor). Logit i predicts the token at anchor_pos + i + 1.
// Draft KV is cat(context, block); the block is dropped and context keeps accepted tokens.
#pragma once

#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

class NativeEmbed;
class NativeHead;
class WeightTable;

inline constexpr int kDflashTaps = 5;
inline constexpr int kDflashCaptureLayers[kDflashTaps] = {4, 16, 24, 36, 44};
inline constexpr int kDflashBlock = 7;
inline constexpr int kDflashMaskId = 248077;

class DFlashDrafter {
public:
    DFlashDrafter() = default;
    ~DFlashDrafter();
    DFlashDrafter(const DFlashDrafter&) = delete;
    DFlashDrafter& operator=(const DFlashDrafter&) = delete;

    /// BF16 safetensors. `max_seq` sizes the context KV (printed with the weight bytes).
    bool load(const std::string& path, int64_t max_seq, std::string& err);
    /// Target embedding and LM head. The embedding comes from `embed` (GGUF table, Q2 line) or, when that is
    /// null, from the pack's on-device token_embd.weight via `wt` (NVFP4 line, same gather as MTP).
    /// Call after both exist and before the expert cache is sized
    /// if the logits buffer should not eat the reserve (the weights and KV are taken in load).
    bool bind(const NativeEmbed* embed, const NativeHead* head, const WeightTable* wt, int64_t n_vocab,
              std::string& err);
    bool loaded() const { return weights_ != nullptr; }
    bool bound() const { return bound_; }

    /// Device buffer the verifier fills. Token t, tap k (k = 0..4 in capture-layer order):
    /// `dst[(k * stride + t) * 2560]`. stride is kVerifyMaxT.
    float* tap_buffer() const { return taps_; }
    int tap_stride() const { return strata::kernels::kVerifyMaxT; }

    /// Project the last window's first `n` tap rows into context KV at absolute positions
    /// pos0 .. pos0+n-1. Block KV is not involved. One stream sync so the next verify window
    /// may overwrite the tap buffer.
    bool ingest_window(int n, int64_t pos0, std::string& err);
    /// Seven greedy drafts. Context KV must already cover every position before `anchor_pos`.
    /// The block's own KV is discarded.
    bool draft(int32_t anchor, int64_t anchor_pos, int32_t* drafts, std::string& err);

    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t kv_bytes() const { return kv_bytes_; }
    uint64_t scratch_bytes() const { return scratch_bytes_; }
    int64_t context_begin() const { return ctx_begin_; }
    int64_t context_end() const { return ctx_end_; }

private:
    bool kv_from_hidden(const float* h, int n, int64_t pos0, int layer, float* k_dst, float* v_dst,
                        bool to_cache, std::string& err);
    bool gemv(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in,
              int64_t n_out, int n, std::string& err);
    void release();

    struct Layer {
        const uint16_t* q = nullptr;
        const uint16_t* k = nullptr;
        const uint16_t* v = nullptr;
        const uint16_t* o = nullptr;
        const uint16_t* gate = nullptr;
        const uint16_t* up = nullptr;
        const uint16_t* down = nullptr;
        float* in_norm = nullptr;
        float* post_norm = nullptr;
        float* q_norm = nullptr;
        float* k_norm = nullptr;
    };

    int device_ = -1;
    void* stream_ = nullptr;
    uint8_t* weights_ = nullptr;
    float* norms_ = nullptr;
    float* rope_cos_ = nullptr;
    float* rope_sin_ = nullptr;
    float* kv_ = nullptr;          ///< [layer][max_seq][2][2][256] K then V
    float* taps_ = nullptr;
    uint8_t* scratch_ = nullptr;
    float* logits_ = nullptr;
    void* xq_ = nullptr;
    const NativeEmbed* embed_ = nullptr;
    const NativeHead* head_ = nullptr;
    const WeightTable* wt_ = nullptr;
    Layer layers_[5] = {};
    const uint16_t* fc_ = nullptr;
    float* hidden_norm_ = nullptr;
    float* final_norm_ = nullptr;
    int64_t max_seq_ = 0;
    int64_t n_vocab_ = 0;
    int64_t ctx_begin_ = 0;
    int64_t ctx_end_ = 0;
    uint64_t weight_bytes_ = 0;
    uint64_t kv_bytes_ = 0;
    uint64_t scratch_bytes_ = 0;
    bool bound_ = false;
    bool warned_late_ = false;

    float* packed_ = nullptr;
    float* noise_ = nullptr;
    float* h_ = nullptr;
    float* q_ = nullptr;
    float* k_ = nullptr;
    float* v_ = nullptr;
    float* attn_ = nullptr;
    float* gate_ = nullptr;
    float* up_ = nullptr;
    float* proj_ = nullptr;
    int32_t* pos_q_ = nullptr;
    int32_t* pos_k_ = nullptr;
    int32_t* tok_ = nullptr;
    int32_t* out_ids_ = nullptr;
};

}  // namespace strata::core
