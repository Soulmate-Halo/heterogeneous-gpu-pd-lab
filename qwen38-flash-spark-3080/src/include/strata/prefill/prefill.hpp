// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/nvfp4_marlin_expert.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::core {
class RDMAExpertTier;
}

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    double ms_ple = 0;
};

class Prefill {
public:
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    /// `remote`: an RDMA cold-expert tier with push mode armed, or null.  When set, every non-resident expert
    /// goes to the peer as ONE push window per layer per chunk (channel 1) instead of being streamed from the
    /// host arena; `init` registers the activation and result blocks with the tier (arm_push_prefill).
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
              core::RDMAExpertTier* remote = nullptr);

    // Set before init; enabled for the explicit STRATA_PF_NVFP4_RESIDENT=1 A/B arm.
    void set_nvfp4_hot(const kernels::Nvfp4HotArena* arena) { nv4_hot_ = arena; }

    /// Batch ladder phase 2: point the prompt path at another session between prompts.  Everything
    /// session-specific is read through the stored pointer at run time (the RDMA push channel and the
    /// work buffers are session-agnostic and stay put); the stream must be idle (between prompts).
    void set_session(core::SessionState& ss);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }

private:
    const kernels::Nvfp4HotArena* nv4_hot_ = nullptr;
    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)
    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
