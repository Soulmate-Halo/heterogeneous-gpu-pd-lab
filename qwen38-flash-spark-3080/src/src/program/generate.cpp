// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.  Everything below it is a
// component; this is the thing that composes them into a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and `tools/strata_tokenizer.py` exist,
// and a C++ BPE is Phase 1's deliverable rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs, because C1 compares logits against
// llama.cpp on the SAME ids, and a tokenizer on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure and the logit dump.

#include "strata/core/cold_expert_host.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/rdma_expert_tier.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
// ---- P4c (NVFP4): the hot expert blob source over a pack-hot directory.  The pack-hot layout is one file
// per layer (`layer-%05d-hot-nvfp4.bin`) holding n_expert canonical v2 blobs of NV4_BLOB bytes back to back
// (tools/nvfp4_pack.py, evidence/nvfp4-pack-p1-20261002.json) - no index, no GGUF.  mmap per layer file, so
// the 68 GB pack costs address space, not RAM, exactly like the Q2 pack's FileExpertSource.  Linux only:
// the NVFP4 tier runs on the 3080 host; opening it elsewhere is refused, not emulated.
class Nvfp4HotSource final : public strata::core::ExpertSource {
public:
    ~Nvfp4HotSource() override { close(); }
    bool open(const std::string& dir, int64_t n_layers, int64_t n_expert, std::string& err) {
#if defined(_WIN32)
        (void) dir; (void) n_layers; (void) n_expert;
        err = "the NVFP4 hot pack source is implemented for the Linux tier only";
        return false;
#else
        close();
        n_expert_ = n_expert;
        char path[1024];
        for (int64_t l = 0; l < n_layers; ++l) {
            std::snprintf(path, sizeof path, "%s/layer-%05lld-hot-nvfp4.bin", dir.c_str(), (long long) l);
            const int fd = ::open(path, O_RDONLY);
            if (fd < 0) { err = std::string("nvfp4 hot pack: cannot open ") + path; close(); return false; }
            struct stat st {};
            if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != (uint64_t) n_expert * (uint64_t) strata::kernels::NV4_BLOB) {
                err = std::string("nvfp4 hot pack: ") + path + " is not n_expert x NV4_BLOB";
                ::close(fd);
                close();
                return false;
            }
            void* m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) { err = std::string("nvfp4 hot pack: mmap failed for ") + path; close(); return false; }
            files_.push_back({(const uint8_t*) m, (uint64_t) st.st_size});
        }
        return true;
#endif
    }
    void close() {
#if !defined(_WIN32)
        for (auto& f : files_)
            if (f.base != nullptr) munmap((void*) f.base, (size_t) f.bytes);
#endif
        files_.clear();
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (layer < 0 || (size_t) layer >= files_.size() || expert < 0 || expert >= n_expert_) return nullptr;
        ++reads_;
        return files_[(size_t) layer].base + (uint64_t) expert * (uint64_t) strata::kernels::NV4_BLOB;
    }
    int64_t reads() const override { return reads_; }

private:
    struct Mapped { const uint8_t* base = nullptr; uint64_t bytes = 0; };
    std::vector<Mapped> files_;
    int64_t n_expert_ = 0;
    mutable int64_t reads_ = 0;
};

// The PLE table served out of the worker's RAM (--ple-io rdma): rows are pulled one-sided from the region
// the worker published (RDMAExpertTier::fetch_ple_info).  A bounce buffer registered on the stage's PD
// receives each chained read; rows are then copied out contiguously.
class PleRowsOverRdma final : public strata::ngram::PleRemoteRows {
public:
    PleRowsOverRdma(strata::core::RemoteStage* stage, uint64_t base, uint32_t rkey, uint32_t row_bytes)
        : stage_(stage), base_(base), rkey_(rkey), rb_(row_bytes) {}
    ~PleRowsOverRdma() override { std::free(bounce_); }
    bool open(std::string& err) {
        bounce_ = (uint8_t*) std::malloc(CAP * rb_);
        if (bounce_ == nullptr) { err = "ple rdma: bounce allocation failed"; return false; }
        uint32_t rk = 0;
        return stage_->expose_region(bounce_, CAP * rb_, false, false, rk, err);
    }
    bool read_rows(const uint32_t* rows, size_t n, uint8_t* dst, std::string& err) override {
        size_t done = 0;
        while (done < n) {
            const size_t m = n - done < CAP ? n - done : CAP;
            if (!stage_->rdma_read_rows(base_, rkey_, rows + done, m, rb_, bounce_, err)) return false;
            std::memcpy(dst + done * rb_, bounce_, m * rb_);
            done += m;
        }
        return true;
    }

private:
    static constexpr size_t CAP = 4096;   // 360 KiB (720 KiB FP8) per chained read
    strata::core::RemoteStage* stage_;
    uint64_t base_;
    uint32_t rkey_;
    uint32_t rb_;
    uint8_t* bounce_ = nullptr;
};
}  // namespace
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/dflash.hpp"
#include "strata/core/mtp.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/program/logits_selection.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/core/progress.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>

#if defined(STRATA_ENABLE_NVFP4)
namespace strata::core {
void nvfp4_cold_swap_begin();
void nvfp4_cold_swap_end();
}
#endif

namespace {
// perf-review D-4: the lent slots are refilled with queued copies and one wait; STRATA_REFILL_BLOCKING=1 waits on each
bool refill_blocking() {
    static const bool v = std::getenv("STRATA_REFILL_BLOCKING") != nullptr;
    return v;
}

using Clock = std::chrono::steady_clock;

struct Options {
    std::string pack = "pack/full";
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;          // one line of logits per generated position
    int64_t logits_stride = 1;        // storage selection; all prompt tokens remain conditioned
    /// **THE RESIDUAL, SO THE HEAD CAN BE CHECKED WITHOUT THE LAYERS.**
    ///
    /// C1 fails (LEDGER L116) and the pipeline is `embed -> 48 layers -> head`.  Dumping `R` splits it in half:
    /// the head is one norm, two bf16 projections and one 794 MB GEMV, all of which can be recomputed in Python
    /// from the manifest.  If Python agrees with the engine on the same `R`, the head is right and the layers
    /// are wrong; if it disagrees, the head is wrong.  Nothing else in the engine can be split that cheaply.
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    std::string ple_fp8;              // the NVFP4 stack's bare FP8 PLE table: local NVMe Direct reads, no GGUF header
    double ple_fp8_scale = 0.0;       // the bare FP8 table's one global scale (model-plefp8-scale.safetensors)
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool check_logits = false;        // optional full-vocabulary finite scan
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 64;
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default), int8, or fp8 e4m3
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the 794 MB GEMV does not.
    std::string dump_mixed;
    /// One residual snapshot per layer per position: `n_layers * hc * n_embd` floats per position, appended in
    /// position order.  This is the C1 BISECTION LADDER - it is what `llama-debug --tensor-filter l_last` prints
    /// for the reference, so the first layer whose `sum` diverges is the layer that holds the bug.  It needs the
    /// captured path (the expert pool only exists there), so it is refused with `--no-capture`.
    std::string dump_layers;
    /// `2 * n_embd + 2 * hc` floats per layer per position: the attention half's block output, the MoE half's,
    /// and the two injection vectors.  It separates `linear_attn_out-<l>` from `ffn_out-<l>`, which the residual
    /// ladder cannot.  **CAPTURED INTO THE LAYER GRAPHS**, so it must be armed before `session_capture`.
    std::string dump_halves;
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    bool sync_every_layer = false;
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< R0.3: pre + post + head, the true per-token GPU floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`src/kernels/cuda/s2_expert_grouped.cu`), it is wired at line ~660
    /// via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    /// The RDMA cold-expert tier (docs/RDMA_EXPERTS.md): every expert miss - the rows neither the CUDA0 cache
    /// nor a CUDA1..3 tier owns - is computed by a cold_expert_worker on a REMOTE machine and pulled back over
    /// the one-sided flag ring.  For hosts whose CPU cannot run the AVX-512 kernel at all (its startup check
    /// is skipped when this is set, because the CPU pool then has nothing to compute).
    std::string remote_rdma;              ///< worker host; empty = off
    int remote_rdma_port = 39580;
    std::string remote_rdma_device;       ///< local HCA name; empty = first verbs device
    int remote_rdma_gid = 0;
    int remote_rdma_slots = 4;
    int64_t remote_rdma_slot_bytes = 4 << 20;
    int remote_rdma_wait_ms = 30000;
    /// Push mode (default ON): the worker one-sided-reads each layer's miss list and activation and
    /// RDMA-writes the rows and the doorbell flag - this host's verbs leave the layer loop entirely.
    /// --no-remote-rdma-push keeps the pull path for A/B.
    bool remote_rdma_push = true;
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// The PLE gather's prefetch, as an A/B arm.  The gather measured 2.10-2.61 ms/token because its sixteen
    /// row reads are sixteen SEPARATE page faults into a 26.8 GB mapping; see `ple_prefetch_enable`.
    bool no_ple_prefetch = false;
    /// One-beat PLE pipeline (default on). Once the next token id is known, its 16 row reads start, and
    /// `ple_finish_token` on that token only waits out what is left. `--no-ple-ahead` and `STRATA_PLE_AHEAD=0`
    /// issue at the start of the token again. This is not `--no-ple-prefetch` (that switch is the mmap hint).
    bool no_ple_ahead = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// P4c: the expert weight encoding.  q2 (default): the canonical Q2_0 pack.  nvfp4: `--pack` names a
    /// pack-hot directory, the cache goes planar and hits run the marlin kernel (P2); the remote worker
    /// must run the matching --format (the push bind carries the format, fail-closed).
    std::string expert_format = "q2";
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    /// R0.9: capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  Every expert a chunk
    /// routes to is streamed once per chunk, so a bigger chunk streams fewer bytes per token (the "ubatch" effect).
    bool prefill_auto = false;
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// NVFP4 standalone (the Spark single-machine stack): rank EVERY (layer, expert) pair so the planar
    /// tier holds all experts - no --expert-profile, no remote tier, prefill/decode never miss.
    bool expert_resident_all = false;
    /// P1②: serve the RDMA cold-expert protocol IN-PROCESS (the standalone worker's doorbell loop as a
    /// thread), computing cold windows on the SAME resident marlin plane via moe_prefill_grouped_nvfp4 -
    /// one weight copy covers local prefill AND the dual-machine decode peer (needs --expert-resident-all).
    bool cold_expert_inproc = false;
    std::string cold_expert_device;      ///< HCA name; empty = first verbs device
    int cold_expert_port = 39580;
    int cold_expert_gid = 0;
    int cold_expert_slots = 4;
    long long cold_expert_slot_bytes = 4 << 20;
    /// Plan v0.3 P5 validation: batch only positions [0, P) and run the rest of the prompt through the token path
    /// (teacher-forced), so the logits of positions >= P - which depend on the batched state - can be scored
    /// against the oracle at many positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    /// Plan v0.3 P6: after every processed position, append the residual after the last layer (hc x n_embd
    /// floats, the MTP draft head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for
    /// the exactness test); `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
    int spec = 0;
    std::string spec_oracle;
    int spec_corrupt = 0;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    /// DFlash block drafter (BF16 safetensors). Empty: the drafter is not loaded and every captured
    /// graph stays the one this build had without it.
    std::string dflash;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// NVFP4: learn the resident set from live misses and refill the marlin planes in place.
    /// Default off, so a run without this flag keeps today's residency (the memcpy adapt path included).
    /// Also enabled by STRATA_HOT_LEARN=1.
    bool hot_learn = false;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (strata/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// A control vector on the residual stream (strata/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
};

void usage() {
    std::fprintf(stderr,
                 "strata generate --pack DIR --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-gguf PATH      required PLE table (original second GGUF shard)\n"
                 "  --ple-fp8 PATH       NVFP4 stack: the bare FP8 PLE table on the local NVMe (Direct reads,\n"
                 "                       no GGUF header) - the same rows RDMA would serve from the worker's RAM\n"
                 "  --ple-fp8-scale F    the bare FP8 table's global scale (required with --ple-fp8)\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 64)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv fp8             FP8 e4m3 codes + fp16 scale per 64 values (1,056 B/cell, same as int8);\n"
                 "                       software conversion on sm_86 (no FP8 tensor cores). Default stays fp16\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision (see bench/results/2026-09-27-kv-q4)\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); not with --kv-resident\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --check-logits       copy and check all logits in the stream-token path\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --dflash PATH        DFlash block drafter (BF16 safetensors); default off, needs --spec\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 8192 whose buffers the expert cache can lend\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --sync-every-layer   debug: synchronise after every layer\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --expert-format F    q2 (default) | nvfp4: --pack is a pack-hot dir, planar cache, marlin hits.\n"
                 "  --hot-learn          NVFP4: record live routing (including the single-token path) and,\n"
                 "                       every --adapt-every rounds, refill marlin slots in place (high-usage\n"
                 "                       misses in, lowest-usage residents out). Experts that missed during\n"
                 "                       the current request are pinned and are not eviction victims; the pin\n"
                 "                       set is cleared when the request ends. Default off. STRATA_HOT_LEARN=1\n"
                 "                       is the same switch.\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of\n"
                 "                       issuing them in one PrefetchVirtualMemory call.\n"
                 "  --no-ple-ahead        issue the PLE's 16 row reads at the start of the token. Default: issue\n"
                 "                       them as soon as that token's id is known (the previous sample, or the\n"
                 "                       prompt when the next id is already there). STRATA_PLE_AHEAD=0 matches.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-resident-all  NVFP4 standalone: rank every (layer, expert) pair so the planar\n"
                 "                       tier holds ALL experts (use with --expert-cache 24576; replaces\n"
                 "                       --expert-profile; the engine then runs with no remote expert tier).\n"
                 "  --cold-expert-inproc  P1②: run the cold-expert worker's RDMA doorbell as an in-process\n"
                 "                       thread on the resident marlin plane (needs --expert-resident-all);\n"
                 "                       one weight copy serves local prefill AND the dual-machine decode peer.\n"
                 "  --cold-expert-device HCA / --cold-expert-port N / --cold-expert-gid N\n"
                 "  --cold-expert-slots N / --cold-expert-slot-bytes B  ring geometry (must match the peer)\n"
                 "  --expert-cache-device1 N  pre-fill N experts on CUDA1 (experimental)\n"
                 "  --expert-cache-device2 N  pre-fill N more experts on CUDA2\n"
                 "  --expert-cache-device3 N  pre-fill N more experts on CUDA3\n"
                 "  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole\n"
                 "                       layers across CUDA1..3 (default: stripe)\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on.  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep CPU misses resident in ordinary RAM.\n"
                 "                       Borrowed GPU-cache entries may read from mmap during prompt prefill.\n");
}

/// All shards of a split GGUF, from shard 1's path ("...-00001-of-00002.gguf"); just the path when it is not split.
std::vector<std::string> model_shards(const std::string& first) {
    const std::string tag = "-00001-of-";
    const size_t at = first.rfind(tag);
    if (at == std::string::npos || first.size() < at + tag.size() + 10) return {first};
    const int total = std::atoi(first.substr(at + tag.size(), 5).c_str());
    std::vector<std::string> out;
    for (int i = 1; i <= total && i <= 99; ++i) {
        char num[8];
        std::snprintf(num, sizeof num, "%05d", i);
        std::string p = first;
        p.replace(at + 1, 5, num);
        if (std::ifstream(p, std::ios::binary)) out.push_back(p);
    }
    return out.empty() ? std::vector<std::string>{first} : out;
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
};

/// Session pin + single-token usage for --hot-learn.  Off unless the flag (or STRATA_HOT_LEARN) is set,
/// so the default adapt path does not see these counts.
struct HotLearnState {
    bool on = false;
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    std::vector<uint8_t> pin;   ///< 1 = this request has already missed (layer, expert)
};
HotLearnState g_hot;

void hot_learn_clear_pin() {
    if (!g_hot.pin.empty()) std::fill(g_hot.pin.begin(), g_hot.pin.end(), 0);
}

bool hot_pinned(int64_t layer, int32_t expert) {
    if (!g_hot.on || g_hot.n_expert <= 0) return false;
    const size_t at = (size_t) layer * (size_t) g_hot.n_expert + (size_t) expert;
    return at < g_hot.pin.size() && g_hot.pin[at] != 0;
}

void hot_learn_note(strata::core::ExpertDispatch& d, int64_t layer, const int32_t* ids, int64_t n, bool add_usage) {
    if (!g_hot.on || ids == nullptr || layer < 0 || layer >= g_hot.n_layers || n < 1) return;
    const int64_t ne = d.n_expert;
    if (ne != g_hot.n_expert || g_hot.pin.size() != (size_t) g_hot.n_layers * (size_t) ne) return;
    float* u = (add_usage && !d.usage.empty()) ? d.usage.data() + (size_t) layer * (size_t) ne : nullptr;
    const int32_t* res = d.host_res;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || (int64_t) e >= ne) continue;
        if (u != nullptr) u[(size_t) e] += 1.0f;
        const size_t at = (size_t) layer * (size_t) ne + (size_t) e;
        if (res == nullptr || res[at] < 0) g_hot.pin[at] = 1;
    }
}

#if defined(STRATA_ENABLE_NVFP4)
/// One in-place marlin refill: four planes from the hot blob plus the two per-slot globals, on `stream`.
/// Residency tables are the caller's to publish after the stream is synchronized.
bool nvfp4_fill_resident_slot(strata::core::ExpertCache& cache, int32_t slot, int64_t layer, int64_t expert,
                              const uint8_t* blob, std::vector<uint16_t>& gm13, std::vector<uint16_t>& gm2,
                              uint16_t* gm13_dev, uint16_t* gm2_dev, cudaStream_t stream, std::string& err) {
    const uint8_t* planes[4] = {};
    size_t plane_sizes[4] = {};
    if (blob == nullptr || slot < 0 || gm13_dev == nullptr || gm2_dev == nullptr ||
        (size_t) slot >= gm13.size() || (size_t) slot >= gm2.size()) {
        err = "NVFP4 hot-learn: slot or blob is not refillable";
        return false;
    }
    if (!strata::kernels::nvfp4_hot_plane_views(blob, planes, plane_sizes, layer, expert)) {
        err = "NVFP4 hot-learn: the hot blob is not a v2 pack-hot expert";
        return false;
    }
    if (!cache.fill_slot_planar(slot, planes, stream, err)) return false;
    if (!strata::kernels::nvfp4_process_globals(strata::kernels::nvfp4_blob_globals(blob),
                                                gm13[(size_t) slot], gm2[(size_t) slot])) {
        err = "NVFP4 hot-learn: global-scale fold refused the blob";
        return false;
    }
    if (cudaMemcpyAsync(gm13_dev + slot, &gm13[(size_t) slot], sizeof(uint16_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(gm2_dev + slot, &gm2[(size_t) slot], sizeof(uint16_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        err = "NVFP4 hot-learn: uploading the per-slot global scales failed";
        return false;
    }
    return true;
}

/// Rank like the Q2 adapt path, then actually land the winner in the planar slot.  The cold doorbell is
/// held across the copy and the host_res / gm / cold_slot_of publish, and the caller only invokes this
/// between decode-graph replays.  `nswap` is the number of slots overwritten.  Usage decays either way.
bool hot_learn_planar_adapt(strata::core::ExpertCache& cache, std::vector<float>& usage,
                            std::vector<int32_t>& host_res, int32_t* d_res,
                            std::vector<uint16_t>& gm13, std::vector<uint16_t>& gm2,
                            uint16_t* gm13_dev, uint16_t* gm2_dev,
                            std::vector<int32_t>& cold_slot_of, strata::core::ExpertSource* src,
                            int64_t n_layers, int64_t n_expert, int adapt_swaps, cudaStream_t stream,
                            int64_t& nswap, std::string& err) {
    nswap = 0;
    struct Hold {
        Hold() { strata::core::nvfp4_cold_swap_begin(); }
        ~Hold() { strata::core::nvfp4_cold_swap_end(); }
    } hold;
    // The doorbell is quiescent. Drain every stream (the token graph may not be the stream we copy on)
    // before overwriting a slot the next replay will read.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = "NVFP4 hot-learn: the device was not idle before the refill";
        return false;
    }
    if (src == nullptr || usage.size() != (size_t) n_layers * (size_t) n_expert ||
        host_res.size() != usage.size()) {
        err = "NVFP4 hot-learn: residency tables are not armed";
        return false;
    }
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + l * n_expert;
        const int32_t* r = host_res.data() + l * n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
            else if (!hot_pinned(l, e)) vict.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if (adapt_swaps >= 0 && (int) swaps.size() > adapt_swaps) swaps.resize((size_t) adapt_swaps);
    struct Item { int32_t layer, in, out, slot; };
    std::vector<Item> landed;
    landed.reserve(swaps.size());
    for (const Swap& s : swaps) {
        const size_t out = (size_t) s.layer * (size_t) n_expert + (size_t) s.out;
        const int32_t slot = host_res[out];
        const uint8_t* b = src->blob(s.layer, s.in);
        if (slot < 0 || !nvfp4_fill_resident_slot(cache, slot, s.layer, s.in, b, gm13, gm2, gm13_dev, gm2_dev,
                                                  stream, err)) {
            if (err.empty()) err = "NVFP4 hot-learn: an adaptive refill failed";
            return false;
        }
        landed.push_back({s.layer, s.in, s.out, slot});
    }
    if (!landed.empty()) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            err = "NVFP4 hot-learn: the planar refill did not finish";
            return false;
        }
        for (const Item& it : landed) {
            const size_t in = (size_t) it.layer * (size_t) n_expert + (size_t) it.in;
            const size_t out = (size_t) it.layer * (size_t) n_expert + (size_t) it.out;
            host_res[out] = strata::core::kNotResident;
            host_res[in] = it.slot;
            if (!cold_slot_of.empty() && in < cold_slot_of.size() && out < cold_slot_of.size()) {
                cold_slot_of[out] = -1;
                cold_slot_of[in] = it.slot;
            }
        }
        if (d_res != nullptr &&
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            err = "NVFP4 hot-learn: the residency table did not reach the device";
            return false;
        }
    }
    for (float& v : usage) v *= 0.7f;
    nswap = (int64_t) landed.size();
    return true;
}
#endif

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    // The multi-token path already counts every routed id.  This path did not; --hot-learn counts them
    // the same way and pins the misses for the rest of the request.
    if (g_hot.on && !t->d.failed) hot_learn_note(t->d, t->d.layers - 1, ids, k, /*add_usage=*/true);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS, AND THE FIRST TRACE WAS OFF BY ONE
        // BECAUSE OF IT.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  `tools/make_profile.py` caught it
        // with a bounds check when the trace turned out to span 1..48 instead of 0..47.  The hit-rate CURVE was
        // unaffected - it is a per-layer split, and shifting every layer by one preserves both metrics - but
        // anything keyed on the layer index, which is exactly what a cache profile is, would have been wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..47\n", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

/// Batch ladder phase 2: the pool entry for `session_run_token_dual`.  The two slots share ONE Drive (one hot
/// cache, one remote tier, one RDMA connection) and pool calls stay serialized on the dual runner's thread, so
/// the only thing the interleave cannot share is the dispatch's self-incrementing layer counter - it is set
/// explicitly here instead.  `pull` forces the legacy pull request for this slot's call (see
/// RDMAExpertTier::set_decode_pull): the push channel has one registered buffer set, so both slots pull.
struct PoolAt {
    Drive* t = nullptr;
    strata::core::RDMAExpertTier* tier = nullptr;   // null when the remote tier is not open
    bool pull = false;
    int channel = 0;   // the push channel this slot's single-token requests publish on (0 = slot A, 3 = slot B)
};
void drive_pool_at(void* user, int64_t layer, const float* x_f, const int32_t* ids, const float* weights,
                   int64_t n_embd, int64_t k, float* out) {
    PoolAt* pa = (PoolAt*) user;
    if (pa->tier != nullptr) {
        pa->tier->set_decode_pull(pa->pull);
        pa->tier->set_decode_channel(pa->channel);
    }
    pa->t->d.layers = layer;
    drive_pool(pa->t, x_f, ids, weights, n_embd, k, out);
}

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    // Usage is already counted inside the multi dispatch.  Pin only the misses of this request.
    if (g_hot.on && !t->d.failed) hot_learn_note(t->d, layer, ids, n_tok * k, /*add_usage=*/false);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: tools/make_profile.py ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    strata::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};
void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::NativeHead head;
    strata::core::SessionState ss;
    cudaStream_t stream = nullptr;
    strata::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    strata::core::Verifier ver;
    strata::prefill::Prefill sp;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard (Linux) / all (Windows)
};
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    strata::core::Progress& p = strata::core::progress();
    std::fprintf(f, "strata serve: stall report (engine %s): stage \"%s %lld\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, p.where.load(),
                 (long long) p.detail.load(), (long long) ((strata::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = strata::core::diag_pool_fn().load()) fn(f);
        if (auto fn = strata::core::diag_verify_fn().load()) fn(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
#if defined(_WIN32)
                     "committed", m.avail_mib, m.faults, "page faults so far"
#else
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
#endif
        );
        std::fflush(f);
    }
#if defined(_WIN32)
    // every thread's stack, to read against this build's PDB: a few MB beside the engine's working directory
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        using Fn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        if (auto write = (Fn) GetProcAddress(dbg, "MiniDumpWriteDump")) {
            char path[64];
            std::snprintf(path, sizeof path, "strata-stall-%lu.dmp", (unsigned long) GetCurrentProcessId());
            HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                const int kThreadInfo = 0x1000;   // MiniDumpWithThreadInfo; MiniDumpNormal = 0
                const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), h, kThreadInfo, nullptr, nullptr, nullptr);
                CloseHandle(h);
                char full[MAX_PATH];
                if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
                std::fprintf(f, "  %s the thread stacks to %s (attach it to the issue)\n", ok ? "wrote" : "could not write",
                             full);
            }
        }
    }
#endif
}

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const bool on = std::getenv("STRATA_TRACE") != nullptr;
    if (!on) return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again is what made a long session wait minutes for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
struct ImgKey {
    int64_t start = 0;      ///< the image's first <|image_pad|> position
    uint64_t hash = 0;      ///< its embeddings and grid: the pad tokens alone are the same for every picture
    bool operator==(const ImgKey& o) const { return start == o.start && hash == o.hash; }
};

struct ConvCheckpoint {
    std::vector<int32_t> ids;     ///< the tokens this state has consumed
    std::vector<ImgKey> imgs;     ///< the images among them
    std::vector<uint8_t> gdn, ple, tails;
    uint64_t used = 0;            ///< last-use stamp for the retention policy (conv_cache.hpp)
    std::vector<ConvCheckpoint> stage_parts;   ///< a layer split's later stages: their sessions' running state
};

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct ConvStateSizes {
    size_t gdn = 0, ple = 0, tail = 0;
};

ConvStateSizes conv_state_sizes(const strata::core::ModelGeometry& g) {
    ConvStateSizes z;
    z.gdn = (size_t) g.n_gdn_layers() *
            ((size_t) g.ssm_state_size * (size_t) g.ssm_v_heads * (size_t) g.ssm_state_size +
             (size_t) g.ssm_conv_channels * (size_t) (g.ssm_d_conv - 1)) * sizeof(float);
    z.ple = (size_t) strata::kernels::NG_HIST * (size_t) strata::kernels::NG_HC_DIM * sizeof(float);
    z.tail = (size_t) (strata::kernels::qsa_real_shapes().idx_block - 1) * (size_t) g.idx_key_dim * sizeof(float);
    return z;
}

/// STRATA_G2_STATE=1 (debug): fingerprints one session's state (R, GDN, PLE history, QSA indexer tails,
/// fp8 KV cells [0, L)) per component, so two runs of the same prompt can be compared field by field.
/// STRATA_G2_STATE_LAYERS=1 adds a per-QSA-layer KV tail.
void state_hash_dump(const strata::core::ModelGeometry& g, strata::core::SessionState& s, const char* tag,
                     int64_t L) {
    if (std::getenv("STRATA_G2_STATE") == nullptr) return;
    cudaDeviceSynchronize();
    auto hdev = [&](const void* p, size_t bytes, uint64_t h) {
        std::vector<uint8_t> b(bytes);
        if (bytes) cudaMemcpy(b.data(), p, bytes, cudaMemcpyDefault);
        return fnv1a(b.data(), b.size(), h);
    };
    const ConvStateSizes z = conv_state_sizes(g);
    const uint64_t F = 1469598103934665603ull;
    const uint64_t h_r = hdev(s.block.R, (size_t) g.hc * g.n_embd * 4, F);
    const uint64_t h_gdn = hdev(s.gdn_state, (size_t) z.gdn, F);
    const uint64_t h_ple = hdev(s.ple_hist, (size_t) z.ple, F);
    strata::kernels::QsaShapes qs = strata::kernels::qsa_real_shapes();
    qs.n_head_kv = g.n_head_kv; qs.head_dim = g.head_dim; qs.idx_dim = g.idx_key_dim;
    const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
    uint64_t h_tail = F, h_kv = F;
    std::string per;
    char pb[16];
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        const strata::core::QsaState& st = s.qsa_states[i];
        h_tail = hdev(st.idx_tail, (size_t) z.tail, h_tail);
        if (!st.kv_fp8) continue;
        const bool hc = st.kv_mode != 0;
        const std::pair<const void*, int64_t> arrs[4] = {
            {hc ? (const void*) st.host.k_fp8 : (const void*) st.k_fp8, kvb},
            {hc ? (const void*) st.host.v_fp8 : (const void*) st.v_fp8, kvb},
            {hc ? (const void*) st.host.k_fp8_scale : (const void*) st.k_fp8_scale, scb},
            {hc ? (const void*) st.host.v_fp8_scale : (const void*) st.v_fp8_scale, scb}};
        uint64_t hl = F;
        for (const auto& [pool, w] : arrs) {
            if (pool == nullptr) continue;
            const int64_t ps = qs.page_size;
            for (int64_t pg = 0; pg * ps < L; ++pg)
                for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                    const int64_t e = std::min(L - pg * ps, ps);
                    const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps) * w);
                    hl = hdev((const uint8_t*) pool + off, (size_t) (e * w), hl);
                }
        }
        h_kv ^= hl + 0x9e3779b97f4a7c15ull + (h_kv << 6) + (h_kv >> 2);
        if (std::getenv("STRATA_G2_STATE_LAYERS") != nullptr) {
            std::snprintf(pb, sizeof(pb), " %lld:%04llx", (long long) i, (unsigned long long) (hl & 0xffff));
            per += pb;
        }
    }
    std::fprintf(stderr,
                 "G2STATE %s L=%lld r=%016llx gdn=%016llx ple=%016llx tail=%016llx kv=%016llx "
                 "ple_prev=%d,%d%s\n", tag, (long long) L, (unsigned long long) h_r,
                 (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                 (unsigned long long) h_kv, s.ple_prev[0], s.ple_prev[1], per.c_str());
    std::fflush(stderr);
}

/// Copies the running state out.  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    const ConvStateSizes z = conv_state_sizes(g);
    c.gdn.resize(z.gdn);
    c.ple.resize(ss.ple_hist != nullptr ? z.ple : 0);
    c.tails.resize(z.tail * (size_t) g.n_qsa_layers());
    if (cudaMemcpy(c.gdn.data(), ss.gdn_state, z.gdn, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (!c.ple.empty() && cudaMemcpy(c.ple.data(), ss.ple_hist, z.ple, cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        if (cudaMemcpy(c.tails.data() + (size_t) i * z.tail, ss.qsa_states[i].idx_tail, z.tail, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return false;
    return true;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    const ConvStateSizes z = conv_state_sizes(g);
    if (c.gdn.size() != z.gdn || c.tails.size() != z.tail * (size_t) g.n_qsa_layers()) return false;
    if (cudaMemcpy(ss.gdn_state, c.gdn.data(), z.gdn, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (!c.ple.empty() && cudaMemcpy(ss.ple_hist, c.ple.data(), z.ple, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        if (cudaMemcpy(ss.qsa_states[i].idx_tail, c.tails.data() + (size_t) i * z.tail, z.tail, cudaMemcpyHostToDevice) !=
            cudaSuccess)
            return false;
    // the PLE's token window is the last two tokens, OLDEST FIRST, -1 where there is none (as session_zero leaves it)
    const size_t L = c.ids.size();
    ss.ple_prev[0] = L >= 2 ? c.ids[L - 2] : -1;
    ss.ple_prev[1] = L >= 1 ? c.ids[L - 1] : -1;
    return cudaDeviceSynchronize() == cudaSuccess;
}

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports.
bool load_control_vectors(const Options& o, const strata::core::ModelGeometry& g, std::string& summary, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                std::fprintf(stderr, "strata generate: %s was made for '%s', not qwen4exp\n", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!strata::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    std::fprintf(stderr, "strata generate: control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
double probe_pcie_h2d_gbps() {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kIters = 4;
    void* h = nullptr;
    void* d = nullptr;
    cudaEvent_t ev0, ev1;
    if (cudaMallocHost(&h, kBytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d, kBytes) != cudaSuccess || cudaEventCreate(&ev0) != cudaSuccess ||
        cudaEventCreate(&ev1) != cudaSuccess) {
        if (d != nullptr) cudaFree(d);
        cudaFreeHost(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    cudaEventRecord(ev0);
    for (int i = 0; i < kIters; ++i) cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(ev1);
    const bool ok = cudaEventSynchronize(ev1) == cudaSuccess;
    float ms = 0.f;
    const bool timed = ok && cudaEventElapsedTime(&ms, ev0, ev1) == cudaSuccess && ms > 0.01f;
    const double bw = timed ? ((double) kIters * (double) kBytes / (ms * 1e-3)) / 1e9 : -1.0;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d);
    cudaFreeHost(h);
    return bw;
}

}  // namespace

int main(int argc, char** argv) {
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
    Options o;
    bool have_tokens = false;
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { std::fprintf(stderr, "--logits-stride must be supplied only once\n"); return 2; }
            if (!strata::program::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                std::fprintf(stderr, "--logits-stride requires a positive decimal int64\n"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--ple-fp8") o.ple_fp8 = next("--ple-fp8");
        else if (a == "--ple-fp8-scale") o.ple_fp8_scale = std::atof(next("--ple-fp8-scale"));
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--check-logits") o.check_logits = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--sync-every-layer") o.sync_every_layer = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--no-ple-prefetch") o.no_ple_prefetch = true;
        else if (a == "--no-ple-ahead") o.no_ple_ahead = true;
        else if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--expert-cache-device1") o.expert_cache_remote[0] = std::atoi(next("--expert-cache-device1"));
        else if (a == "--expert-cache-device2") o.expert_cache_remote[1] = std::atoi(next("--expert-cache-device2"));
        else if (a == "--expert-cache-device3") o.expert_cache_remote[2] = std::atoi(next("--expert-cache-device3"));
        else if (a == "--expert-cache-remote-placement")
            o.expert_cache_remote_placement = next("--expert-cache-remote-placement");
        else if (a == "--remote-rdma") o.remote_rdma = next("--remote-rdma");
        else if (a == "--remote-rdma-port") o.remote_rdma_port = std::atoi(next("--remote-rdma-port"));
        else if (a == "--remote-rdma-device") o.remote_rdma_device = next("--remote-rdma-device");
        else if (a == "--remote-rdma-gid") o.remote_rdma_gid = std::atoi(next("--remote-rdma-gid"));
        else if (a == "--remote-rdma-slots") o.remote_rdma_slots = std::atoi(next("--remote-rdma-slots"));
        else if (a == "--remote-rdma-slot-bytes") o.remote_rdma_slot_bytes = std::atoll(next("--remote-rdma-slot-bytes"));
        else if (a == "--remote-rdma-wait-ms") o.remote_rdma_wait_ms = std::atoi(next("--remote-rdma-wait-ms"));
        else if (a == "--no-remote-rdma-push") o.remote_rdma_push = false;
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_auto = v == "auto";
            o.prefill_chunk = o.prefill_auto ? 8192 : std::atoll(v.c_str());
        }
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--expert-resident-all") o.expert_resident_all = true;
        else if (a == "--cold-expert-inproc") o.cold_expert_inproc = true;
        else if (a == "--cold-expert-device") o.cold_expert_device = next("--cold-expert-device");
        else if (a == "--cold-expert-port") o.cold_expert_port = std::atoi(next("--cold-expert-port"));
        else if (a == "--cold-expert-gid") o.cold_expert_gid = std::atoi(next("--cold-expert-gid"));
        else if (a == "--cold-expert-slots") o.cold_expert_slots = std::atoi(next("--cold-expert-slots"));
        else if (a == "--cold-expert-slot-bytes") o.cold_expert_slot_bytes = std::atoll(next("--cold-expert-slot-bytes"));
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--dflash") o.dflash = next("--dflash");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--split-device") o.split_device = next("--split-device");
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return 2; }
        }
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--hot-learn") o.hot_learn = true;
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-format") o.expert_format = next("--expert-format");
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = true;
        else if (a == "--stats") o.stats = true;
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (const char* ple_ahead_env = std::getenv("STRATA_PLE_AHEAD")) {
        if (std::strcmp(ple_ahead_env, "0") == 0 || std::strcmp(ple_ahead_env, "off") == 0 ||
            std::strcmp(ple_ahead_env, "false") == 0)
            o.no_ple_ahead = true;
    }
    if (!o.hot_learn) {
        if (const char* hot_env = std::getenv("STRATA_HOT_LEARN")) {
            if (std::strcmp(hot_env, "1") == 0 || std::strcmp(hot_env, "on") == 0 ||
                std::strcmp(hot_env, "true") == 0)
                o.hot_learn = true;
        }
    }
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  Across
    // GPUs, not yet: KV streaming, images, control vectors, the helper caches (--expert-cache-remote), and lending
    // cache slots to the prompt path (each stage's prompt path has its own buffers).
    const bool pcie_given = o.pcie_frac >= 0.0;
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    bool split_auto = false, split_same = false;
    if (!o.layer_split.empty()) {
        int n_dev = 1;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) n_dev = 1;
        cudaGetLastError();
        auto ints = [](const std::string& str, auto& out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = o.serve && (split_auto || ints(o.layer_split, split_at));
        if (ok && !o.split_device.empty()) ok = ints(o.split_device, split_devs);
        else if (ok)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto;
        if (ok && split_auto && split_devs.empty()) {
            std::fprintf(stderr, "strata generate: --layer-split auto: one GPU visible, so no split\n");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = split_devs[i] > 0 && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            std::fprintf(stderr, "strata generate: --layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU)\n", n_dev - 1);
            return 2;
        }
    }
    const bool multi_gpu = !split_devs.empty() && !split_same;
    // P4c: the NVFP4 expert tier.  Constraints are the ones P2/P3's acceptance was built under: a static
    // profile (the runtime admit path has no planar fill - the profile must cover every slot so admission
    // always refuses), a cache to hold the hot tier, and no helper-GPU/layer-split experiments on top.
    if (o.expert_format != "q2" && o.expert_format != "nvfp4") {
        std::fprintf(stderr, "strata generate: --expert-format must be q2 or nvfp4\n");
        return 2;
    }
    const bool nvfp4_experts = o.expert_format == "nvfp4";
    if (o.expert_resident_all && !nvfp4_experts) {
        std::fprintf(stderr, "strata generate: --expert-resident-all needs --expert-format nvfp4\n");
        return 2;
    }
    if (o.cold_expert_inproc && !o.expert_resident_all) {
        std::fprintf(stderr, "strata generate: --cold-expert-inproc needs --expert-resident-all "
                             "(the doorbell shares the resident marlin plane)\n");
        return 2;
    }
    if (nvfp4_experts) {
        if (o.expert_cache <= 0 || (o.expert_profile.empty() && !o.expert_resident_all) || o.pack.empty() ||
            multi_gpu || !o.layer_split.empty() || o.expert_cache_remote[0] > 0) {
            std::fprintf(stderr, "strata generate: --expert-format nvfp4 needs --pack <pack-hot dir>, "
                                 "--expert-cache N, a static --expert-profile (or --expert-resident-all), "
                                 "and no GPU splits\n");
            return 2;
        }
        // The prompt path cannot borrow planar slots (device_slot refuses a planar cache): its buffers are
        // allocated for the whole session instead, and the cache auto-sizing accounts them (prefill_mib).
        o.no_prefill_borrow = true;
#if !defined(STRATA_ENABLE_NVFP4)
        std::fprintf(stderr, "strata generate: --expert-format nvfp4 needs a build with STRATA_ENABLE_NVFP4=ON\n");
        return 2;
#endif
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty() || o.adapt_every != 0)) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts requires --mmap-experts, a static --expert-profile and --adapt-every 0\n");
        return 2;
    }
    if (o.resident_cpu_experts &&
        (!o.layer_split.empty() || o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 ||
         o.expert_cache_remote[2] > 0)) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts does not support layer splits or remote expert caches\n");
        return 2;
    }
    // the helper-GPU expert caches (--expert-cache-remote, docs/SECOND_GPU.md): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            std::fprintf(stderr, "strata generate: a layer split across GPUs needs --expert-profile\n");
            return 2;
        }
        o.no_prefill_borrow = true;   // each stage's prompt path has its own buffers
        int n_vis = 1;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || n_vis < 1) n_vis = 1;
        cudaGetLastError();
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                std::fprintf(stderr, "strata generate: --expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)\n", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        std::fprintf(stderr, "strata generate: layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)\n",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
    if (o.prefill_auto && (o.no_prefill_borrow || o.expert_profile.empty())) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        std::fprintf(stderr, "strata generate: --tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)\n");
        usage();
        return 2;
    }

    if ((o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "rdma") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        std::fprintf(stderr, "strata generate: invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us\n");
        return 2;
    }
    if (o.ple_io == "rdma" && o.remote_rdma.empty()) {
        std::fprintf(stderr, "strata generate: --ple-io rdma needs --remote-rdma (the table is the worker's RAM)\n");
        return 2;
    }
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "fp8" && o.kv != "q4_0" && o.kv != "k8v4") {
        std::fprintf(stderr, "strata generate: --kv must be fp16, int8, fp8, q4_0 or k8v4\n");
        return 2;
    }
    strata::core::qsa_set_kv_int8(o.kv == "int8");
    strata::core::qsa_set_kv_fp8(o.kv == "fp8");
    strata::core::qsa_set_kv_q4(o.kv == "q4_0");   // PR #21: 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    strata::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (o.kv_resident < 0) {
        std::fprintf(stderr, "strata generate: --kv-resident must be >= 0\n");
        return 2;
    }
    if (o.kv == "k8v4" && o.kv_resident > 0) {
        std::fprintf(stderr, "strata generate: --kv k8v4 does not support --kv-resident streaming (yet)\n");
        return 2;
    }
    strata::core::qsa_set_kv_resident(o.kv_resident);
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer; the draft policy (strata/spec/draft_policy.hpp) takes one only where it pays. Code
    // edits +6-11%, ordinary text unchanged (bench/results/2026-09-27-spec). --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
    if (!o.dflash.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --dflash needs --spec T (T >= 2)\n");
            return 2;
        }
        if (!o.mtp.empty()) {
            std::fprintf(stderr, "strata generate: --dflash and --mtp are mutually exclusive\n");
            return 2;
        }
        if (!o.layer_split.empty()) {
            std::fprintf(stderr, "strata generate: --dflash does not support --layer-split\n");
            return 2;
        }
        if (o.spec != 8)
            std::fprintf(stderr, "strata generate: --dflash verifies a window of 8 (anchor + 7 drafts)\n");
        o.spec = 8;
        o.mtp_max_t = 0;
    }
    strata::core::layer_set_shared_early(!o.shared_late);
    if (!o.native_preset.empty()) {
        if (o.no_ple || o.ple_gguf.empty()) {
            std::fprintf(stderr, "strata generate: --native requires --ple-gguf (the PLE key is native too)\n");
            return 2;
        }
        o.stream_token = true;
        o.gr_native_mmvf = true;
        o.native_bf16 = o.native_bf16_extra = true;
        o.native_ple_key = o.native_moe_combine = o.native_gdn = o.native_router = true;
        o.native_qsa = o.native_qsa_indexer = o.native_rope = o.native_ple_postops = true;
        if (o.native_head_gguf.empty()) o.native_head_gguf = o.native_preset;
        if (o.native_dense_gguf.empty()) {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native), then the PLE shard: a split
            // may put any layer in any shard (Swift's GGUFs: layers 13-47 in shard 2, the PLE table in shard 1)
            o.native_dense_gguf = model_shards(o.native_preset);
            if (std::find(o.native_dense_gguf.begin(), o.native_dense_gguf.end(), o.ple_gguf) == o.native_dense_gguf.end())
                o.native_dense_gguf.push_back(o.ple_gguf);
        }
        // Plan v0.3 (24 Sep): the CPU experts stay on the VNNI kernel.  The llama.cpp-CPU-exact q8_0 contract
        // cost 27.0 vs 17.2 ms/token of pool time and G-C does not need it; `--cpu-oracle-q8-0` still selects it.
    }
    if (o.logits_stride > 1 && (o.max_new != 1 || o.dump_logits.empty())) {
        std::fprintf(stderr, "strata generate: --logits-stride > 1 requires --max-new 1 and --dump-logits\n");
        return 2;
    }
    if (o.no_ple && (!o.ple_gguf.empty() || !o.ple_fp8.empty())) {
        std::fprintf(stderr, "strata generate: --no-ple and --ple-gguf/--ple-fp8 are mutually exclusive\n");
        return 2;
    }
    if (!o.ple_fp8.empty() && (o.ple_fp8_scale <= 0.0 || o.ple_io == "rdma")) {
        std::fprintf(stderr, "strata generate: --ple-fp8 needs --ple-fp8-scale and is incompatible with --ple-io rdma (the table is a local NVMe file, not the worker's RAM)\n");
        return 2;
    }
    if (o.native_ple_postops && o.no_ple) {
        std::fprintf(stderr, "strata generate: --native-ple-postops requires PLE enabled\n");
        return 2;
    }
    if (!o.no_ple && o.ple_gguf.empty() && o.ple_fp8.empty()) {
        std::fprintf(stderr, "strata generate: --ple-gguf or --ple-fp8 is required; --no-ple explicitly enables a diagnostic ablation\n");
        return 2;
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        std::fprintf(stderr, "strata generate: --max-context must be below 2^31\n");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        std::fprintf(stderr, "strata generate: positive --max-new and --max-context must fit the prompt and generation\n");
        return 2;
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        std::fprintf(stderr, "strata generate: invalid sampling or resource parameter\n");
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        std::fprintf(stderr, "strata generate: --expert-cache-remote-placement must be stripe or layer\n");
        return 2;
    }

    if (o.native_flash_attn_short && o.max_context > 256) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires --max-context <=256\n");
        return 2;
    }
    if (o.native_flash_attn_short && (o.gpu_only_full || o.graph_only || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires the normal decode loop for status validation\n");
        return 2;
    }
    if (o.native_ple_key && (o.native_dense_gguf.empty() || o.no_ple)) {
        std::fprintf(stderr, "strata generate: --native-ple-key requires PLE and --native-dense-gguf\n");
        return 2;
    }
    if (o.cpu_oracle_q8_0 && (o.expert_cache != 0 || !o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --cpu-oracle-q8-0 cannot be combined with --expert-cache or --expert-profile until the GPU expert contract matches\n");
        return 2;
    }

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI) and its translation unit is
    // compiled `/arch:AVX512`, so on a CPU without those features it does not fail - it executes an illegal
    // instruction at some unpredictable token.  Refusing at second zero is the whole point of P2.S3's check.
    strata::kernels::cpu::expert_set_oracle_q8_0(o.cpu_oracle_q8_0);

    std::string err;
    if (!o.native_head_gguf.empty() && !o.stream_token) {
        std::fprintf(stderr, "--native-head-gguf requires --stream-token\n");
        return 2;
    }
    // Plan v0.3 P6: where the experts live.  A native pack (tools/iq_pack.py: the IQ2_XS / IQ3_XXS files) keeps
    // every quantized tensor in its GGUF form, so it needs --native (the dense projections, head and embedding
    // come from the model file) and runs its experts in verify windows only (--spec).
    {
        const strata::core::ModelGeometry g0;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
    }
    const bool native_pack = strata::kernels::cpu::expert_layout().native;
    if (!o.remote_rdma.empty()) {
        if (native_pack) {
            std::fprintf(stderr, "strata generate: --remote-rdma serves canonical Q2_0 packs (experts.bin) only; "
                                 "%s is a native (IQ) pack\n", o.pack.c_str());
            return 2;
        }
        if (o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0) {
            std::fprintf(stderr, "strata generate: --remote-rdma cannot be combined with --expert-cache-device1..3 "
                                 "(both claim the same miss rows)\n");
            return 2;
        }
        if (o.remote_rdma_slots <= 0 || o.remote_rdma_slot_bytes < (1400000) || o.remote_rdma_wait_ms <= 0) {
            std::fprintf(stderr, "strata generate: --remote-rdma needs slots >= 1 and slot-bytes >= ~1.4 MB "
                                 "(a full 8x10 verify window's rows are 800 KiB; the protocol cap is 128 entries)\n");
            return 2;
        }
    }
    if (o.expert_cache_remote[0] > 0) {
        // Keep CUDA1's proven startup order: initialise its context before
        // allocating GPU0 weights or mapping the large host expert arena.
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[0], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[0], free_gib);
    }
    // plan v0.3 P6: the PCIe share of the missed experts, measured per kind of pack (the paper, finding on PCIe).
    // PR #44: a x8 link carries half of what the native default assumes - the GPU's SMs read that share over the
    // link (the copy kernel, since 0.1.14), so on a slower link it must shrink or the window waits for it.  The
    // real H2D bandwidth is probed once; from 20 GB/s up (x16 PCIe 4/5) the measured default stays.  The canonical
    // pack's 0.2 was never measured against the link, so it is left alone.  `--calibrate` measures it outright.
    if (o.pcie_frac < 0.0) {
        const double base = native_pack ? 0.55 : 0.2;
        const double bw = native_pack ? probe_pcie_h2d_gbps() : -1.0;
        if (!native_pack) {
            o.pcie_frac = base;
        } else if (bw > 0.0) {
            // below ~4 GB/s (an x1 link: ~0.9 GB/s) a missed expert's 1.4 MB takes longer to cross than the CPU
            // pool takes to compute it, so none of them go over the link
            o.pcie_frac = bw >= 20.0 ? base : bw < 4.0 ? 0.0 : std::min(base, std::max(0.05, base * (bw / 26.0)));
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device -> pcie_frac %.2f (default %.2f)\n",
                         bw, o.pcie_frac, base);
        } else {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe failed -> pcie_frac default %.2f\n", base);
        }
    }
    // the canonical Q2_0 pack's CPU kernels are AVX-512 only; a native pack runs on AVX2 CPUs as well.
    // With --remote-rdma every miss is served by the remote worker and the CPU expert kernel never runs, so
    // its AVX-512 requirement does not apply (P2.S3's check exists to stop an illegal instruction at token
    // 4000 - there is no instruction to execute when the pool has no jobs).
    if (!native_pack && o.remote_rdma.empty()) strata::kernels::cpu::cpu_require_expert_support();
    else if (!native_pack)
        std::fprintf(stderr, "strata generate: --remote-rdma %s: cold experts served remotely; the CPU expert "
                             "kernel (AVX-512) is not required\n", o.remote_rdma.c_str());
    else if (!strata::kernels::cpu::cpu_avx512_ok())
        std::fprintf(stderr, "strata generate: this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)\n",
                     std::getenv("STRATA_NO_IQ256") == nullptr ? "AVX-2" : "ggml-cpu vec_dot (STRATA_NO_IQ256 set)");
    strata::core::NativeEmbed native_embed;
    if (native_pack) {
        if (o.native_preset.empty() || o.spec < 2 || o.keep_canonical ||
            (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            std::fprintf(stderr, "strata generate: %s is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) "
                                 "and --prefill CHUNK\n", o.pack.c_str());
            return 2;
        }
        const strata::core::ModelGeometry g0;
        if (!native_embed.load(o.native_preset, g0.n_embd, 248320, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        strata::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "strata generate: native pack: %s experts (largest blob %.2f MB), token embedding "
                             "type %d in mapped host memory (%.0f MiB)\n",
                     o.pack.c_str(), (double) strata::kernels::cpu::expert_layout().max_blob / 1e6,
                     native_embed.type(), (double) native_embed.bytes() / 1048576.0);
    }
    // Plan v0.3 P1: tensors served in native form are not also loaded in canonical form (~2.7 GB of VRAM back
    // to the expert cache with --native).  `--keep-canonical` loads both, as before.
    std::set<std::string> skip;
    if (!o.keep_canonical) {
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::served_names(o.native_dense_gguf, o.native_ple_key, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.native_head_gguf.empty()) skip.insert("output.weight");
        // the PLE module validates its canonical key at construction (8 MB); a native pack has none to load
        if (!native_pack) skip.erase("blk.1.ple_key.weight");
        if (native_pack) skip.insert("token_embd.weight");
    }
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool_bytes) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: cudaMalloc(%llu) for the weight arena failed\n",
                     (unsigned long long) pool_bytes);
        return 1;
    }
    strata::core::WeightTable wt;
    if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata generate: %llu MiB of weights loaded from %s (%zu canonical tensors skipped: "
                         "served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), o.pack.c_str(), skip.size());

    strata::core::NativeDense native_dense;
    if (!o.native_dense_gguf.empty()) {
        if (!native_dense.load(o.native_dense_gguf, wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: native dense projections: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: %zu native projection matrices, %.2f MiB of weights\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0));
    }

    strata::kernels::gr_set_fp32_activations(o.gr_fp32_activations);
    strata::kernels::gr_set_native_mmvf(o.gr_native_mmvf);
    // Plan v0.3 P3: the fused hyper-connection read rides the native (FP32-activation) contract; the per-stage
    // and dump measurements need the unfused layout of R, so they keep the old kernels.
    strata::core::layer_set_fast_attn(!o.no_fast_attn);
    strata::core::layer_set_publish_kernel(!o.no_publish_kernel);
    strata::core::layer_set_fused_gdn(!o.no_fused_gdn);
    strata::core::layer_set_fast_select(!o.no_fast_select);
    strata::core::layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr && !o.gpu_stages && o.dump_layers.empty() &&
                                     o.dump_halves.empty() && !o.stage_timing);
    strata::core::layer_set_native_bf16(o.native_bf16);
    strata::core::layer_set_native_flash_attn_short(o.native_flash_attn_short);
    strata::kernels::ple_set_native_bf16(o.native_bf16_extra);
    strata::kernels::shared_expert_set_native_bf16(o.native_bf16_extra);
    strata::kernels::native_moe_combine_set_enabled(o.native_moe_combine);
    strata::kernels::native_gdn_set_enabled(o.native_gdn);
    strata::kernels::native_router_set_enabled(o.native_router);
    strata::kernels::native_qsa_set_enabled(o.native_qsa);
    strata::kernels::native_qsa_indexer_set_enabled(o.native_qsa_indexer);
    strata::kernels::native_rope_set_enabled(o.native_rope);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (strata/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (cudaMalloc(&d_mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot allocate the image position table\n");
            return 1;
        }
        strata::kernels::mrope_table_set(d_mrope);
    }
    strata::kernels::ple_set_native_postops(o.native_ple_postops);
    strata::core::ModelGeometry g;   // canonical defaults; the model file overrides the MoE shape below
    int64_t K = 10;
    if (!o.native_preset.empty()) {
        // a pruned variant (GSQ-RCO Coder) ships fewer experts than the canonical 512x10; the model file
        // is the authority on its own MoE shape - everything else in the geometry is unchanged
        try {
            strata::GgufFile model_gguf(o.native_preset);
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) g.n_expert = (int64_t) v->u;
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) K = (int64_t) v->u;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: reading the model's expert shape from %s: %s\n",
                         o.native_preset.c_str(), e.what());
            return 1;
        }
    }
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    std::string cvec_summary = "0";
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!load_control_vectors(o, g, cvec_summary, ce)) {
            std::fprintf(stderr, "strata generate: control vector: %s\n", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        std::fprintf(stderr, "strata generate: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    // Batch ladder (phase 1 of the N=2..8 decode batching): sessions are an indexed array and sequence i's
    // arena is carved from sbuf + i * kSessionBytes (each slot is a full KV namespace of its own - the QSA
    // pools and page tables live inside the arena).  Only slot 0 is allocated and live today; every use
    // below goes through the `ss` reference, so behavior is byte-identical until a later phase lights up
    // more slots.
    constexpr int MAX_BATCH = 8;
    const uint64_t kSessionBytes = strata::core::session_bytes(g, o.max_context, K);
    void* sbuf = nullptr;
    if (cudaMalloc(&sbuf, kSessionBytes) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: session state allocation failed\n");
        return 1;
    }
    strata::core::SessionState ss_all[MAX_BATCH];
    strata::core::SessionState& ss = ss_all[0];
    // **THE ENGINE RAN ON THE LEGACY DEFAULT STREAM, WHICH ON WDDM IS THE SLOW PATH.**  All four session
    // calls - `session_capture`, `session_replay`, `session_token` and `session_loop` - were handed `nullptr`,
    // i.e. stream 0.  `bench/micro/kernel_costs.cu` measures what that costs: EVERY kernel it launches through
    // a wrapper comes back at 28-31 us REGARDLESS OF SIZE, `scale_inplace` on 2,048 floats and `silu_inplace`
    // on 10,240 floats being indistinguishable, which is a fixed per-launch cost and not execution.
    // `bench/micro/graph_node_cost.cu` measures the same kernels on a real stream at 3.63 us ungrapped and
    // 0.805 us inside a graph.  **That is an ~8x penalty on every launch in the engine.**
    cudaStream_t main_stream = nullptr;
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: cannot create the main stream\n");
        return 1;
    }
    void* const main_cs = (void*) main_stream;
    if (strata::core::session_init(g, o.max_context, K, sbuf, ss) == 0) {
        std::fprintf(stderr, "strata generate: session_init failed\n");
        return 1;
    }
    if (g.n_qsa_layers() > 0 && ss.qsa_states[0].kv_mode == 1)
        std::fprintf(stderr, "strata generate: KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                             "%.2f GiB of pinned RAM\n", (long long) (ss.qsa_states[0].n_slots * 4),
                     (long long) o.max_context, (double) strata::core::qsa_kv_host_bytes() / 1073741824.0);

    // ---- **THE HALF-LEVEL DUMP HAS TO BE ARMED BEFORE `session_capture`, AND THE LADDER MUST NOT BE.**  The
    // half copies are issued from inside `block_layer_pre`/`block_layer_post`, so they are only ever enqueued
    // while a graph is being CAPTURED - arming `ss.block.dump` afterwards would produce a file of zeros that
    // reads exactly like a wrong answer.  The ladder is the opposite: `session_loop` enqueues it per token on
    // the replay stream, so it must be armed after capture to stay out of the graph.
    const uint64_t half_stride = (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc +
                                 (uint64_t) g.n_head * g.head_dim +
                                 (uint64_t) 5 * g.n_head_kv * g.head_dim + 8;
    std::FILE* half_dump = nullptr;
    float* half_stage = nullptr;
    if (!o.dump_halves.empty()) {
        half_dump = std::fopen(o.dump_halves.c_str(), "wb");
        if (half_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_halves.c_str());
            return 1;
        }
        const size_t n = (size_t) g.n_layers * (size_t) half_stride;
        if (cudaHostAlloc((void**) &half_stage, n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot pin the half-dump staging buffer\n");
            return 1;
        }
        ss.block.dump = half_stage;
    }

    strata::core::Doorbell db;
    if (strata::core::doorbell_init(g, K, db) == 0) {
        std::fprintf(stderr, "strata generate: doorbell_init failed\n");
        return 1;
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    strata::kernels::PleTable ple_table;
    std::unique_ptr<strata::ngram::PleRemoteRows> ple_remote;   // rdma mode: built once the tier exists
    std::vector<float> ple_emb_host((size_t) strata::kernels::NG_N_EMBD * 2);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    if (!o.ple_gguf.empty() || !o.ple_fp8.empty()) {
        strata::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap"  ? strata::kernels::PleIo::Mmap
                 : o.ple_io == "rdma" ? strata::kernels::PleIo::Rdma
                                      : strata::kernels::PleIo::Direct;
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        pio.io_thread = !o.ple_sync_submit;
        if (!o.ple_fp8.empty()) {
            // The bare FP8 table (--ple-fp8) is a local NVMe file: Direct reads of FP8 E4M3 rows with
            // the checkpoint's one global scale - no GGUF header, no worker, no RDMA in the PLE path.
            pio.mode = strata::kernels::PleIo::Direct;
            pio.format = strata::kernels::PleRowFormat::Fp8E4m3;
            pio.fp8_scale = (float) o.ple_fp8_scale;
            pio.bare_table = true;
        }
        // RDMA mode defers the open: the row source comes from the worker's handshake, after the tier opens.
        const std::string& ple_path = o.ple_fp8.empty() ? o.ple_gguf : o.ple_fp8;
        if (pio.mode != strata::kernels::PleIo::Rdma && !ple_table.open(ple_path, err, pio)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const strata::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const strata::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const strata::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const strata::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const strata::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const strata::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            std::fprintf(stderr, "strata generate: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                                 "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (o.native_ple_key && wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23) || !wk->native_q8_1) {
                std::fprintf(stderr, "strata generate: native PLE key is absent or incompatible\n");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = strata::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        ss.ple.hist = ss.ple_hist;
        ss.ple.emb_host = ple_emb_host.data();
        ss.ple.emb_host_alt = ple_emb_host.data() + (size_t) strata::kernels::NG_N_EMBD;
        ss.ple.ahead = !o.no_ple_ahead;
        if (cudaMalloc((void**) &ple_emb_dev, (size_t) strata::kernels::NG_N_EMBD * 4) != cudaSuccess ||
            cudaMalloc((void**) &ple_scratch, strata::core::ple_run_scratch_bytes()) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the PLE buffers failed\n");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
        if (pio.mode != strata::kernels::PleIo::Rdma && !ss.ple.ready()) {
            // RDMA mode opens the table after the tier's handshake; ready() is re-evaluated per token anyway.
            std::fprintf(stderr, "strata generate: the PLE run is not ready after construction\n");
            return 1;
        }
        if (pio.mode != strata::kernels::PleIo::Rdma)
            std::fprintf(stderr, "strata generate: PLE on, table %llu rows of %s%s\n",
                         (unsigned long long) ple_table.rows(), ple_path.c_str(),
                         ss.ple.ahead ? " (issue one token ahead)" : " (issue at token start)");
        else
            std::fprintf(stderr, "strata generate: PLE on, table opens over RDMA once the worker publishes it\n");
    } else {
        std::fprintf(stderr,
                     "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                     "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n");
    }

    float* d_parts = nullptr;
    if (cudaMalloc(&d_parts, (size_t) K * g.n_embd * 4) != cudaSuccess ||
        cudaMemset(d_parts, 0, (size_t) K * g.n_embd * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the parts buffer failed\n");
        return 1;
    }

    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped (as the drafter below, for the same WDDM reason)
    std::vector<std::unique_ptr<GpuStage>> stages;
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        const bool last = i + 1 == split_devs.size();
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const strata::core::OnDevice on(st.dev);
        void* arena_s = nullptr;
        if (cudaMalloc(&arena_s, pool_bytes) != cudaSuccess ||
            !st.wt.load(o.pack, arena_s, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d weights: %s\n", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str());
            return 1;
        }
        if (!o.native_dense_gguf.empty() && !st.dense.load(o.native_dense_gguf, st.wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d native dense projections: %s\n", st.dev,
                         err.c_str());
            return 1;
        }
        void* sbuf_s = nullptr;
        if (cudaMalloc(&sbuf_s, strata::core::session_bytes(g, o.max_context, K)) != cudaSuccess ||
            strata::core::session_init(g, o.max_context, K, sbuf_s, st.ss) == 0 ||
            cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st.adapt_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&st.adapt_ev, cudaEventDisableTiming) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: the session state failed\n", st.dev);
            return 1;
        }
        const strata::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !o.native_head_gguf.empty() && !st.head.load(o.native_head_gguf, g.n_embd, wo_s->ne1, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d head: %s\n", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str());
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!strata::kernels::cvec_replicate(err)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %s\n", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (cudaMalloc(&st.mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(st.mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d: the image position table failed\n", st.dev);
                return 1;
            }
            strata::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed)
        st.pcie_frac = o.pcie_frac;
        if (!pcie_given && native_pack) {
            const double bw = probe_pcie_h2d_gbps();
            if (bw > 0.0) st.pcie_frac = bw >= 20.0 ? 0.55 : bw < 4.0 ? 0.0 : std::min(0.55, std::max(0.05, 0.55 * (bw / 26.0)));
            std::fprintf(stderr, "strata generate: layer split: CUDA%d PCIe probe %.1f GB/s -> pcie_frac %.2f\n", st.dev,
                         bw, st.pcie_frac);
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights, session%s; %.2f GiB free\n",
                     st.dev, last ? " and head" : "", (double) fb / 1073741824.0);
    }
    GpuStage* const last_st = stages.empty() ? nullptr : stages.back().get();

    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts.
    // In particular WDDM can refuse the draft weights after mapping tens of GiB of host pages.
    strata::core::MtpDrafter mtp;
    strata::core::DFlashDrafter dflash;
    if (!o.dflash.empty() && !dflash.load(o.dflash, o.max_context, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    if (!o.mtp.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        static const strata::core::ModelGeometry draft_geometry{};
        // with a layer split across GPUs the drafter reads the last stage's residual: it lives on that device
        const strata::core::OnDevice on_mtp(last_st ? last_st->dev : -1);
        if (!o.mtp.empty() && !mtp.load(o.mtp, draft_geometry, last_st ? last_st->ss : ss, o.spec, err, o.mtp_window)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
    }

    // ---- the CPU expert pool
    //
    // R2.1: the experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind
    // `--mmap-experts` because it is the A/B arm, not because it is competitive.
    //
    // The reasoning is the review's C1 and it is now measured on both sides.  `FileExpertSource` maps the 34 GB
    // file, and mapped file pages are the first thing the OS reclaims; the engine's rate then depends on whether
    // the standby list happens to hold `experts.bin`, which is why two consecutive runs of the SAME BINARY with
    // the SAME FLAGS measured 71.97 and 34.78 ms/token in the pool (7.54 vs 12.18 tok/s).  The arena is
    // anonymous memory the engine owns, and the pool runs at 19.41 ms/token - 1.79x better than the warm mmap
    // and 3.7x better than the cold one.
    //
    // IT IS NOT PINNED, and that is reported rather than hidden: `cudaHostRegister` on 31.64 GiB fails with
    // "out of memory" (you cannot pin 34 of 63 GB) and the arena falls back to 4 KB anonymous pages.  That is
    // fine for the CPU pool - which is all that exists today - and NOT fine for Phase 3, whose cache fills and
    // CPU/PCIe miss split need the GPU to DMA out of this arena.  Read `note()` when that lands.
    //
    // The earlier "the arena does not fit" conclusion was WRONG and is worth recording: the failure was a stale
    // CUDA error left set by the failed `cudaHostRegister` and read later by `gr_read`'s launch check.  See the
    // note in `pinned.cu`.
    strata::core::FileExpertSource src;
    strata::core::ArenaExpertSource arena_src;
    Nvfp4HotSource nv4_src;
    strata::core::ExpertSource* srcp = nullptr;
    if (nvfp4_experts) {
        // P4c: --pack names the pack-hot directory; the marlin tier maps one blob array per layer.
        if (!nv4_src.open(o.pack, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: NVFP4 hot experts via mmap (%s, %d layers x %d x %d B)\n",
                     o.pack.c_str(), (int) g.n_layers, (int) g.n_expert, (int) strata::kernels::NV4_BLOB);
        srcp = &nv4_src;
    } else if (o.mmap_experts) {
        if (native_pack) {   // FileExpertSource maps the canonical pack's experts.bin; a native pack has none
            std::fprintf(stderr, "strata generate: --mmap-experts needs a canonical pack (experts.bin); %s is a native "
                                 "(IQ) pack, whose experts are loaded into the arena\n", o.pack.c_str());
            return 2;
        }
        if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts via mmap (--mmap-experts; the A/B arm of R2.1)\n");
        srcp = &src;
    } else {
        arena_src.set_gguf(o.native_preset);   // plan v0.3 P6: a native pack may take its experts from shard 1
        // On the multi-GPU Windows experiment, start with at most 8 GiB of mapped host pages.
        // Unregistered layers remain in the resident arena and use the CPU expert path.
        // (a layer split across GPUs too: pinning all of it into two contexts leaves WDDM refusing every later
        // allocation - measured on the 5080 + 3090 rig: cudaMemGetInfo and the next cudaMalloc fail)
        const uint64_t pin_limit = (o.expert_cache_remote[0] > 0 || multi_gpu) ? (8ull << 30) : 0;
        if (!arena_src.open(o.pack, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: expert arena: %s\n", arena_src.note().c_str());
        std::fprintf(stderr, "strata generate: loaded %.2f GiB at %.2f GiB/s\n",
                     (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        // A rate under ~0.2 GiB/s is not the hardware.  Task Scheduler / service contexts throttle this
        // read+fill about 24x (measured 0.05 vs 1.42 GiB/s for the same binary, args and cache state; the
        // scheduler's defaults - Below normal priority and a least-privilege token - were the only
        // difference between the runs).  Say so instead of letting the user blame the disk; see
        // docs/DETAILS.md, "Running it at startup (Task Scheduler)".
#ifdef _WIN32   // a Windows launch context; elsewhere a load this slow is the disk
        if (arena_src.load_gib_per_second() > 0.0 && arena_src.load_gib_per_second() < 0.2) {
            std::fprintf(stderr,
                         "strata generate: hint: ~24x below what this hardware streams from a normal "
                         "launch. If Strata is started by Task Scheduler or a service, register the task "
                         "with Priority 4 (Normal) and 'Run with highest privileges' - the scheduler's "
                         "defaults (Below normal + a least-privilege token) throttle the load. See "
                         "docs/DETAILS.md ('Running it at startup').\n");
        }
#endif
        srcp = &arena_src;
    }
    strata::kernels::cpu::ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker);
    if (o.no_ple_prefetch) strata::kernels::ple_prefetch_enable(false);
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session and the drafter");
    strata::core::ExpertCache xcache;
    // P4c: the NVFP4 marlin arena and its per-slot global scales (built by the profile fill below; the
    // device gm arrays live as long as the cache does).
    strata::kernels::Nvfp4HotArena nv4arena;
    std::vector<uint16_t> nv4_gm13, nv4_gm2;
    void* nv4_scratch = nullptr;
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the
        // file).  `--expert-cache 0` used to take the count the profile was built for; the profile now ranks
        // every pair (issue #46: a card that holds more than the old 8,000 used to stop there), so it means auto.
        if (o.expert_cache == 0) o.expert_cache = -1;
        std::fprintf(stderr, "strata generate: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    if (o.expert_resident_all) {
        // Standalone full residency: every (layer, expert) pair, layer-major, so the planar profile fill
        // below loads ALL experts at startup and the runtime never misses (no remote tier needed).  The
        // cache must hold them all exactly - a short cache would pass the "cover every slot" check yet
        // leave the tail non-resident, so pin the count here instead of discovering it at runtime.
        profile.clear();
        const int64_t total = (int64_t) g.n_layers * (int64_t) g.n_expert;
        profile.reserve((size_t) total);
        for (int32_t l = 0; l < g.n_layers; ++l)
            for (int32_t e = 0; e < g.n_expert; ++e)
                profile.emplace_back(l, e);
        if (o.expert_cache > 0 && o.expert_cache != (int) total) {
            std::fprintf(stderr, "strata generate: --expert-resident-all needs --expert-cache %lld "
                                 "(every expert), got %d\n", (long long) total, o.expert_cache);
            return 2;
        }
        o.expert_cache = (int) total;
        std::fprintf(stderr, "strata generate: resident-all: %lld ranked pairs synthesized (every expert)\n",
                     (long long) total);
    }
    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window, measured on
    // the 5080 + 3090 rig (bench/results/2026-09-29-layer-split):
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (0.33 ms on an RTX 5080, 0.50 on a
    //     3090: the per-layer round trip and kernels, not the bytes - both cards have ~950 GB/s);
    //   - an expert no cache holds costs ~190 ms per unit of routed mass: the CPU pool in decode and the PCIe stream
    //     in prompts (fitted: the sweep's best K, 26-28, is where one more layer on the faster card stops paying
    //     for the ~0.1% of the mass it pushes out of its cache);
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, 1 GiB for its windows and the drafter) is used;
    //     the routed mass of rank r is taken as (r+1)^-1.2 (fits the sweep's hit rates: K=24/26/28 predicted
    //     99.53/99.34/99.15%, measured 99.5/99.4/99.0%).
    // Up to 3 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
    // STRATA_SPLIT_MISS_MS tunes the miss cost (a slower CPU: higher).
    const int64_t split_pf_mib = o.prefill_chunk > 0 ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
    auto stage_room = [&](int dev, bool later) -> int64_t {
        const strata::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        if (const cudaError_t e = cudaMemGetInfo(&fb, &tb); e != cudaSuccess)
            std::fprintf(stderr, "strata generate: layer split: CUDA%d free memory: %s\n", dev < 0 ? 0 : dev,
                         cudaGetErrorString(e));
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + split_pf_mib + (later ? 1024 : 0)) << 20;
        return std::max<int64_t>((int64_t) fb - reserve, 0);
    };
    if (multi_gpu && split_auto) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int ns = (int) stages.size() + 1;
        std::vector<int64_t> cap((size_t) ns), used((size_t) ns);
        std::vector<double> layer_ms((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
            cap[(size_t) i] = stage_room(i == 0 ? -1 : dev, i > 0);
            int sms = 0, khz = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0) khz = 1800000;
            cudaGetLastError();
            const double speed = std::max(1.0, (double) sms * (double) khz / 1e6);   // SMs x GHz
            layer_ms[(size_t) i] = 0.33 * (84.0 * 2.617) / speed;
            std::fprintf(stderr, "strata generate: layer split auto: CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, "
                                 "%.2f GiB for experts\n", dev, sms, khz / 1e6, layer_ms[(size_t) i],
                         (double) cap[(size_t) i] / 1073741824.0);
        }
        const double miss_ms = std::getenv("STRATA_SPLIT_MISS_MS") ? std::atof(std::getenv("STRATA_SPLIT_MISS_MS")) : 190.0;
        std::vector<double> mass(profile.size());
        double total_mass = 0;
        for (size_t r = 0; r < profile.size(); ++r) total_mass += (mass[r] = std::pow((double) r + 1.0, -1.2));
        auto cost = [&](int64_t l) -> int64_t {
            return native_pack ? ((int64_t) lay.blob_bytes(l) + 255) / 256 * 256 : (int64_t) lay.max_blob;
        };
        // the predicted window time (ms) of a placement, and the routed mass its caches hold
        auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
            std::fill(used.begin(), used.end(), 0);
            held_mass = 0;
            held = 0;
            std::vector<bool> full((size_t) ns, false);
            for (size_t r = 0; r < profile.size(); ++r) {
                const int64_t l = profile[r].first;
                int st = 0;
                while (st + 1 < ns && l >= at[(size_t) st]) ++st;
                if (full[(size_t) st]) continue;
                if (used[(size_t) st] + cost(l) > cap[(size_t) st]) { full[(size_t) st] = true; continue; }   // as the fill
                used[(size_t) st] += cost(l);
                held_mass += mass[r];
                ++held;
            }
            held_mass /= std::max(total_mass, 1e-9);
            double ms = miss_ms * (1.0 - held_mass);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1], le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                ms += (double) (le - lb) * layer_ms[(size_t) i];
            }
            return ms;
        };
        std::vector<int64_t> best, at((size_t) ns - 1);
        double best_ms = 1e30, best_mass = 0;
        int64_t best_held = 0;
        auto consider = [&]() {
            double hm = 0;
            int64_t held = 0;
            const double ms = predict(at, hm, held);
            if (ms < best_ms) { best = at; best_ms = ms; best_mass = hm; best_held = held; }
        };
        const int64_t L = g.n_layers;
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else {
            double total = 0;
            for (const double c : layer_ms) total += 1.0 / c;
            double acc = 0;
            for (int i = 0; i + 1 < ns; ++i) {
                acc += 1.0 / layer_ms[(size_t) i];
                at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                    i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
            }
            consider();
        }
        split_at = best;
        std::string ks;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        std::fprintf(stderr, "strata generate: layer split auto: K=%s - predicted %.1f ms per decode window; the caches "
                             "hold %lld of %zu profiled pairs (~%.1f%% of the routed mass)\n", ks.c_str(), best_ms,
                     (long long) best_held, profile.size(), 100.0 * best_mass);
    }
    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            std::fprintf(stderr, "strata generate: --layer-split: layer %lld is past the last (%lld)\n",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the stage that runs a layer (0: CUDA0's)
    auto stage_of = [&](int64_t l) -> int {
        int st = 0;
        while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
        return st;
    };
    if (multi_gpu) {
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }
    // THE HEAD BEFORE THE CACHE.  The expert cache takes what is free minus the reserve, so everything allocated
    // after it comes out of the reserve.  The native head (~0.5 GB with IQ3_S) was loaded after it and ate most of
    // the 700 MiB: 128K IQ3_S ended with 30 MiB free, the driver paged, and a request stalled for good at its first
    // verify window.  Loaded first, the cache is sized around it.
    const strata::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { std::fprintf(stderr, "strata generate: output.weight is missing\n"); return 1; }
    const int64_t n_vocab = wo->ne1;
    strata::core::NativeHead native_head;
    if (!o.native_head_gguf.empty() && !multi_gpu) {   // a layer split's head is on its last stage
        if (!native_head.load(o.native_head_gguf, g.n_embd, n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experimental native Q5_K head, %llu bytes\n",
                     (unsigned long long) native_head.weight_bytes());
    }
    if (dflash.loaded() &&
        !dflash.bind(native_embed.bytes() ? &native_embed : nullptr, native_head.loaded() ? &native_head : nullptr,
                     &wt, n_vocab, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, (size_t) n_vocab * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the logits buffer failed\n");
        return 1;
    }
    const bool auto_cache = o.expert_cache < 0;
    if (o.expert_cache < 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // Plan v0.3 P5: the batched prompt path's chunk buffers are allocated later, so they are reserved here -
        // under WDDM an over-subscribed allocation does not fail, it pages to system memory and crawls.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead)
        const bool borrow = !o.no_prefill_borrow && !o.expert_profile.empty();
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        int64_t slots = ((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "strata generate: expert cache auto: %.2f GiB free, %d MiB reserved -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, o.expert_cache);
    } else if (multi_gpu && o.expert_cache > 0) {
        // a layer split's prompt path has its own buffers (it borrows no slots): an explicit cache size leaves room
        // for them and the reserve, or the first prompt fails with "device buffers ... do not fit"
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t prefill_mib = o.prefill_chunk > 0 ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob, 0);
        if (o.expert_cache > fit) {
            std::fprintf(stderr, "strata generate: layer split: --expert-cache %d leaves no room for the prompt path's "
                                 "buffers (%lld MiB) on CUDA0: %lld slots\n", o.expert_cache, (long long) prefill_mib,
                         (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so with a profile its slots are sized per pair: the
    // same VRAM holds ~30% more IQ3_XXS experts than slots of the largest blob would
    std::vector<int64_t> sized_slots;
    if (native_pack && o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const auto& lay = strata::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * lay.max_blob;   // what the uniform sizing granted
        uint64_t used = 0;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        o.expert_cache = (int) sized_slots.size();
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() && used + (sized_slots[keep] + 255) / 256 * 256 <= keep_bytes)
                    used += (sized_slots[keep++] + 255) / 256 * 256;
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) strata::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += (s + 255) / 256 * 256;
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: under WDDM an
        // allocation is not resident until it is touched, and the free figure read before it can be ~1 GB too
        // high.  A cache sized from it filled the card to 0 MiB, the driver then paged, and a request that needed a
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
        int failed = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            if (fake_fails > 0) {
                --fake_fails;
                err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
                if (nvfp4_experts) {
                    // P2: marlin needs every plane contiguous across slots -> the planar cache layout.
                    const int64_t planes[4] = {(int64_t) strata::kernels::NV4_W13_BYTES,
                                               (int64_t) strata::kernels::NV4_W2_BYTES,
                                               (int64_t) strata::kernels::NV4_W13S_BYTES,
                                               (int64_t) strata::kernels::NV4_W2S_BYTES};
                    ok = xcache.open_planar(o.expert_cache, g.n_layers, g.n_expert, planes, 4, err);
                } else {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
                }
            }
            if (!ok) {
                // Issue #60: on Windows a device allocation is also charged to the system commit (RAM + page file),
                // so with a small page file the cache's one big cudaMalloc fails while the VRAM is free.  An auto
                // cache then tries three quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
#if defined(_WIN32)
                MEMORYSTATUSEX ms{};
                ms.dwLength = sizeof ms;
                if (GlobalMemoryStatusEx(&ms))
                    std::snprintf(commit, sizeof commit, " (Windows has %.1f GiB of commit left: RAM + page file)",
                                  (double) ms.ullAvailPageFile / 1073741824.0);
#endif
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    std::fprintf(stderr, "strata generate: %s%s; trying a smaller expert cache: %d slots\n", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
#if defined(_WIN32)
                std::fprintf(stderr, "strata generate: on Windows the graphics card's memory also needs room in the page "
                                     "file: set it to \"System managed\" (System > About > Advanced system settings > "
                                     "Performance > Advanced > Virtual memory), or lower --expert-cache\n");
#endif
                return 1;
            }
            if (!auto_cache || attempt - failed >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least", so then give back a quarter as well
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20)) give = std::max<int64_t>(give, xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "strata generate: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            std::fprintf(stderr, "strata generate: expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM\n",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "strata generate: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        xcache.set_per_layer_admission(o.expert_cache_per_layer);
        // Round 328 warned here that the GPU hit path was wrong (tokens diverged from a cache-off run from
        // token 0). That fault was fixed long since (native_expert_parity, expert_parity, the grouped kernels'
        // tests), and the warning outlived it (issue #23). What remains is rounding: a GPU expert and the CPU's
        // compute the same quantized expert with different float order, so a near-tie can flip. Measured teacher-
        // forced on 2,557 tokens (bench/results/2026-09-27-cache-parity): 95-98% same top-1, and perplexity equal
        // (on - off = -0.005 +- 0.005 nats). Neither output is more correct than the other.
        std::fprintf(stderr,
                     "strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality:\n"
                     "                 bench/results/2026-09-27-cache-parity).\n");
        if (o.expert_cache_per_layer) {
            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));
        } else if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
#if defined(STRATA_ENABLE_NVFP4)
    if (!profile.empty() && srcp != nullptr && nvfp4_experts) {
        // P4c: the planar profile fill.  Every admitted slot gets its four planes copied from the hot blob
        // and its two marlin globals folded on the host (the same steps P2's harness ran, evidence
        // nvfp4-p2-hotpath-20261002.json); the gm arrays go up once, after the loop.  The runtime admit
        // path never fills a planar slot: the profile covers every slot, so admission always refuses and
        // a non-profiled expert stays a remote/CPU miss by construction.
        const int64_t want = std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        nv4_gm13.assign((size_t) xcache.slots(), 0);
        nv4_gm2.assign((size_t) xcache.slots(), 0);
        const uint8_t* planes[4] = {};
        size_t plane_sizes[4] = {};
        const uint8_t* first_blob = nullptr;
        int32_t first_slot = -1;
        uint8_t first_w13_head[4096] = {};
        // Resident-all bulk reads (the Spark single-machine stack): the synthesized profile visits every
        // layer's 512 blobs in order, so read each layer file ONCE with pread into a bounce buffer and
        // point the fill at it.  The mmap path pages the pack in fault-by-fault (~12 MB/s measured on
        // GB10) and the page cache it creates counts AGAINST the unified-memory driver's pool - the
        // planar arena only fits when the cache is cold.  DONTNEED each layer after reading so later
        // allocations (the prompt path's buffers) do not hit the same wall.
        std::vector<uint8_t> bulk;
        size_t bulk_bytes = 0, blob_bytes = 0;
        int32_t bulk_layer = -1;
        if (o.expert_resident_all) {
            char p0[4096];
            std::snprintf(p0, sizeof p0, "%s/layer-%05d-hot-nvfp4.bin", o.pack.c_str(), 0);
            struct stat st;
            if (stat(p0, &st) != 0 || st.st_size <= 0 || st.st_size % g.n_expert != 0) {
                std::fprintf(stderr, "strata generate: resident-all: cannot size %s\n", p0);
                return 1;
            }
            blob_bytes = (size_t) st.st_size / (size_t) g.n_expert;
            bulk_bytes = (size_t) st.st_size;
            try {
                bulk.resize(bulk_bytes);
            } catch (const std::bad_alloc&) {
                std::fprintf(stderr, "strata generate: resident-all: the %.2f GiB layer bounce "
                                     "does not fit\n", bulk_bytes / 1073741824.0);
                return 1;
            }
        }
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = nullptr;
            if (o.expert_resident_all) {
                if (profile[(size_t) i].first != bulk_layer) {
                    char lp[4096];
                    std::snprintf(lp, sizeof lp, "%s/layer-%05d-hot-nvfp4.bin", o.pack.c_str(),
                                  (int) profile[(size_t) i].first);
                    int fd = open(lp, O_RDONLY);
                    size_t got = 0;
                    if (fd >= 0) {
                        while (got < bulk_bytes) {
                            const ssize_t r = pread(fd, bulk.data() + got, bulk_bytes - got, (off_t) got);
                            if (r <= 0) break;
                            got += (size_t) r;
                        }
                        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                        close(fd);
                    }
                    if (got != bulk_bytes) {
                        std::fprintf(stderr, "strata generate: resident-all: read %zu of %zu bytes "
                                             "from %s\n", got, bulk_bytes, lp);
                        return 1;
                    }
                    bulk_layer = profile[(size_t) i].first;
                    if ((bulk_layer & 7) == 0)
                        std::fprintf(stderr, "strata generate: resident-all: layer %d staged\n",
                                     (int) bulk_layer);
                }
                b = bulk.data() + (size_t) profile[(size_t) i].second * blob_bytes;
            } else {
                b = srcp->blob(profile[(size_t) i].first, profile[(size_t) i].second);
            }
            if (b == nullptr ||
                !strata::kernels::nvfp4_hot_plane_views(b, planes, plane_sizes,
                                                        profile[(size_t) i].first, profile[(size_t) i].second) ||
                !xcache.fill_slot_planar(slot, planes, nullptr, err) ||
                !strata::kernels::nvfp4_process_globals(strata::kernels::nvfp4_blob_globals(b),
                                                        nv4_gm13[(size_t) slot], nv4_gm2[(size_t) slot])) {
                std::fprintf(stderr, "strata generate: the NVFP4 profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            if (first_blob == nullptr) {
                first_blob = b; first_slot = slot;
                // the bounce buffer's content turns over per layer, so the readback below compares
                // against this stash, not the pointer
                std::memcpy(first_w13_head, b + strata::kernels::NV4_O_W13, sizeof first_w13_head);
            }
            ++prefilled;
        }
        // The same falsifiability rule as the Q2 fill's verify_slot: read back plane 0 of the first slot
        // (the marlin w13 plane's head) and compare against the blob it came from.
        if (prefilled > 0) {
            std::vector<uint8_t> back(sizeof first_w13_head);
            if (cudaMemcpy(back.data(), xcache.device_plane_slot(0, first_slot), back.size(),
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                std::memcmp(back.data(), first_w13_head, back.size()) != 0) {
                std::fprintf(stderr, "strata generate: the NVFP4 planar fill failed its slot-0 readback\n");
                return 1;
            }
        }
        uint16_t* gm13_dev = nullptr;
        uint16_t* gm2_dev = nullptr;
        if (cudaMalloc((void**) &gm13_dev, nv4_gm13.size() * sizeof(uint16_t)) != cudaSuccess ||
            cudaMalloc((void**) &gm2_dev, nv4_gm2.size() * sizeof(uint16_t)) != cudaSuccess ||
            cudaMemcpy(gm13_dev, nv4_gm13.data(), nv4_gm13.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(gm2_dev, nv4_gm2.data(), nv4_gm2.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the NVFP4 global scales could not reach the device\n");
            return 1;
        }
        nv4arena.w13 = xcache.device_plane_slot(0, 0);
        nv4arena.w2 = xcache.device_plane_slot(1, 0);
        nv4arena.w13s = xcache.device_plane_slot(2, 0);
        nv4arena.w2s = xcache.device_plane_slot(3, 0);
        nv4arena.gm13 = gm13_dev;
        nv4arena.gm2 = gm2_dev;
        nv4arena.n_slots = xcache.slots();
        mem_mark("the NVFP4 profile fill");
        std::fprintf(stderr, "strata generate: NVFP4 planar tier: pre-filled %lld of %lld slots "
                             "(%d planes), slot 0 verified, gm scales uploaded\n",
                     (long long) prefilled, (long long) want, xcache.plane_count());
        if (prefilled < xcache.slots()) {
            // A free slot would be admitted at runtime and the planar cache has no runtime fill path
            // (fill_slot refuses it) - so a short profile is a startup error, not a runtime surprise.
            std::fprintf(stderr, "strata generate: --expert-format nvfp4 needs the profile to cover every "
                                 "slot (%lld profiled < %lld slots)\n",
                         (long long) prefilled, (long long) xcache.slots());
            return 1;
        }
    } else
#endif
    if (!profile.empty() && srcp != nullptr) {
        const int64_t want = std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = srcp->blob(profile[(size_t) i].first, profile[(size_t) i].second);
            if (b == nullptr || !xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first))) {
                std::fprintf(stderr, "strata generate: the profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            ++prefilled;
        }
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile; slot 0 verified\n",
                     (long long) prefilled, (long long) want);
    }

    // ---- P1②: the in-process cold-expert doorbell (the standalone worker's role, same marlin plane) ----
    std::vector<int32_t> cold_slot_of;
    std::unique_ptr<strata::core::ColdExpertHost> cold_host;
    if (o.cold_expert_inproc) {
        if (!nv4arena.on()) {
            std::fprintf(stderr, "strata generate: --cold-expert-inproc: the resident marlin plane is "
                                 "not live\n");
            return 1;
        }
        cold_slot_of.assign((size_t) g.n_layers * (size_t) g.n_expert, -1);
        for (size_t i = 0; i < profile.size(); ++i)
            cold_slot_of[(size_t) profile[i].first * (size_t) g.n_expert + (size_t) profile[i].second] =
                (int32_t) i;
        cold_host.reset(new strata::core::ColdExpertHost(&nv4arena, cold_slot_of.data(), g.n_layers,
                                                         g.n_expert));
        strata::core::ColdExpertHostConfig cc;
        cc.device = o.cold_expert_device;
        cc.port = o.cold_expert_port;
        cc.gid = o.cold_expert_gid;
        cc.slots = o.cold_expert_slots;
        cc.slot_bytes = o.cold_expert_slot_bytes;
        if (!cold_host->start(cc, err)) {
            std::fprintf(stderr, "strata generate: --cold-expert-inproc: %s\n", err.c_str());
            return 1;
        }
    }

    if (o.resident_cpu_experts) {
        if (!src.pin_cache_complement(xcache, err, /*pin=*/false)) {
            std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: steady-state CPU cache misses are resident in ordinary RAM; borrowed cache entries may use mmap during prompt prefill\n");
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int64_t room = stage_room(st.dev, true);
        const strata::core::OnDevice on(st.dev);
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        if (sized.empty() ||
            !(native_pack ? st.cache.open_sized(sized, g.n_layers, g.n_expert, err)
                          : st.cache.open((int64_t) sized.size(), g.n_layers, g.n_expert, (int64_t) lay.max_blob, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         sized.empty() ? "no room" : err.c_str());
            return 1;
        }
        int64_t filled = 0;
        for (const auto& pr : st.profile) {
            if (filled >= st.cache.slots()) break;
            const int32_t slot = st.cache.admit(pr.first, pr.second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = srcp->blob(pr.first, pr.second);
            if (b == nullptr || !st.cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pr.first))) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d profile fill failed at pair %lld: %s\n",
                             st.dev, (long long) filled, err.c_str());
                return 1;
            }
            ++filled;
        }
        if (filled == 0 || !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                 srcp->blob(st.profile[0].first, st.profile[0].second), err,
                                                 (int64_t) lay.blob_bytes(st.profile[0].first))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         filled == 0 ? "nothing filled" : err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                             "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified\n",
                     st.dev, (long long) st.lb, (long long) (st.le - 1), (long long) st.cache.slots(), st.cache.gib(),
                     (long long) filled, st.profile.size());
    }
    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\n", (long long) (split_at[0] - 1));

    std::array<strata::core::RemoteExperts, 3> remote_experts;
    const bool multi_remote = o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
    if (o.expert_cache_remote[0] > 0) {
        if (o.expert_cache <= 0 || profile.empty() || o.no_pool) {
            std::fprintf(stderr, "strata generate: remote experts need --expert-profile, "
                                 "a CUDA0 expert cache and the expert pool\n");
            return 2;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked = profile;
        if (!stages.empty()) {   // a layer split: CUDA0's share of the profile, then the later stages' pairs no cache holds
            for (auto& st : stages)
                for (const auto& pr : st->profile)
                    if (st->cache.slot_of(pr.first, pr.second) < 0) ranked.push_back(pr);
        }
        if (multi_remote) {
            // The shipped frequency profile names only 8000 of 24576 experts. Once exhausted,
            // fill remaining VRAM from unranked pairs in expert-then-layer order: this spreads
            // the tail across all layers instead of concentrating it on layer zero.
            std::vector<uint8_t> seen((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (const auto& pair : ranked)
                if (pair.first >= 0 && pair.first < g.n_layers && pair.second >= 0 && pair.second < g.n_expert)
                    seen[(size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second] = 1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = 0; l < g.n_layers; ++l)
                    if (!seen[(size_t) l * (size_t) g.n_expert + (size_t) e])
                        ranked.emplace_back((int32_t) l, (int32_t) e);
            std::fprintf(stderr, "strata generate: remote ranking: %zu profiled pairs, "
                                 "%zu other pairs to fill CUDA1..3\n", profile.size(), ranked.size() - profile.size());
        }
        std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
        if (multi_remote) {
            // Either stripe experts for parallel GPU work, or give each layer one
            // secondary GPU to reduce switches and transfers over shared USB4.
            std::vector<uint8_t> assigned((size_t) g.n_layers * (size_t) g.n_expert, 0);
            const int devices = 1 + (o.expert_cache_remote[1] > 0) + (o.expert_cache_remote[2] > 0);
            int next = 0;
            for (const auto& pair : ranked) {
                if (pair.first < 0 || pair.first >= g.n_layers || pair.second < 0 || pair.second >= g.n_expert ||
                    xcache.slot_of(pair.first, pair.second) >= 0) continue;
                const size_t index = (size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second;
                if (assigned[index]) continue;
                int target = -1;
                if (o.expert_cache_remote_placement == "layer") {
                    target = pair.first % devices;
                    // Other layers' owners may still have room: keep scanning ranks.
                    if (by_device[(size_t) target].size() >=
                        (size_t) o.expert_cache_remote[(size_t) target]) continue;
                } else {
                    for (int i = 0; i < devices; ++i) {
                        const int r = (next + i) % devices;
                        if (by_device[(size_t) r].size() < (size_t) o.expert_cache_remote[(size_t) r]) {
                            target = r;
                            break;
                        }
                    }
                    if (target < 0) break;
                }
                assigned[index] = 1;
                by_device[(size_t) target].push_back(pair);
                next = (target + 1) % devices;
            }
            std::fprintf(stderr, "strata generate: remote ranks %s across %d CUDA devices\n",
                         o.expert_cache_remote_placement == "layer" ? "grouped by layer" : "striped", devices);
        } else {
            by_device[0] = std::move(ranked);
        }
        std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
        for (auto& st : stages)   // a layer split: what a stage's cache holds is no helper's
            for (const auto& pr : st->profile)
                if (st->cache.slot_of(pr.first, pr.second) >= 0)
                    claimed[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
            if (!remote_experts[(size_t) r].open(remote_dev[r], o.expert_cache_remote[(size_t) r],
                     g.n_layers, g.n_expert, by_device[(size_t) r], xcache, *srcp, claimed, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: CUDA%d: %lld additional experts, %.2f GiB; "
                                 "results return through pinned host rows\n", remote_dev[r],
                         (long long) remote_experts[(size_t) r].resident(), remote_experts[(size_t) r].gib());
        }
    }

    Drive drive;
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    // The RDMA cold-expert tier attaches LAST: it claims every miss row the CUDA tiers (and the CUDA0 cache)
    // did not, so it must see their claims in `kind`/primary_res before deciding its own.
    strata::core::RDMAExpertTier rdma_tier;
    if (!o.remote_rdma.empty()) {
        strata::core::RdmaExpertTierConfig rc;
        rc.peer_host = o.remote_rdma;
        rc.port = (uint16_t) o.remote_rdma_port;
        rc.device = o.remote_rdma_device;
        rc.gid_index = o.remote_rdma_gid;
        rc.slots = (uint32_t) o.remote_rdma_slots;
        rc.slot_bytes = (size_t) o.remote_rdma_slot_bytes;
        rc.wait_ms = o.remote_rdma_wait_ms;
        rc.push = o.remote_rdma_push;
        rc.expert_format = nvfp4_experts ? 1u : 0u;   // P4c: the worker refuses a format mismatch
        if (!rdma_tier.open(rc, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.ple_gguf.empty() && o.ple_io == "rdma") {
            // The PLE table lives in the worker's RAM: pick up the region record it published in ring slot 0
            // (the worker loads 28.8 GB first, so allow minutes), then open the table on top of it.
            strata::core::RDMAExpertTier::PleInfo pi;
            if (!rdma_tier.fetch_ple_info(pi, 300000, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            auto src = std::make_unique<PleRowsOverRdma>(rdma_tier.stage(), pi.addr, pi.rkey, pi.row_bytes);
            if (!src->open(err)) {
                std::fprintf(stderr, "strata generate: ple rdma source: %s\n", err.c_str());
                return 1;
            }
            strata::kernels::PleIoOptions pio;
            pio.mode = strata::kernels::PleIo::Rdma;
            pio.max_inflight = (uint32_t) o.ple_inflight;
            pio.cache_rows = (uint64_t) o.ple_row_cache;
            pio.remote = src.get();
            // v2 records carry the remote table's encoding (the NVFP4 stack serves FP8 E4M3 rows);
            // v1 is the original IQ4_NL-only record.
            pio.format = pi.format == strata::core::RDMAExpertTier::kPleFmtFp8E4m3
                             ? strata::kernels::PleRowFormat::Fp8E4m3
                             : strata::kernels::PleRowFormat::Iq4Nl;
            pio.fp8_scale = pi.fp8_scale;
            if (!ple_table.open(o.ple_gguf, err, pio)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (pi.rows != ple_table.rows()) {
                std::fprintf(stderr, "strata generate: the worker's PLE table has %llu rows, the local shard "
                                     "says %llu - refusing to mix them\n",
                             (unsigned long long) pi.rows, (unsigned long long) ple_table.rows());
                return 1;
            }
            ple_remote = std::move(src);
            std::fprintf(stderr, "strata generate: PLE table over RDMA: %llu rows x %u B (%s) at the worker's "
                                 "%#llx (rkey %#x)\n",
                         (unsigned long long) pi.rows, pi.row_bytes,
                         pio.format == strata::kernels::PleRowFormat::Fp8E4m3 ? "FP8 E4M3" : "IQ4_NL",
                         (unsigned long long) pi.addr, pi.rkey);
        }
        drive.d.remote[drive.d.remote_count++] = &rdma_tier;
        std::fprintf(stderr, "strata generate: RDMA cold-expert tier: %s:%d rdma=true slots=%d slot_bytes=%lld "
                             "- every expert miss crosses the flag ring\n",
                     o.remote_rdma.c_str(), o.remote_rdma_port, o.remote_rdma_slots,
                     (long long) o.remote_rdma_slot_bytes);
    }
    drive.d.hit_cpu_order = o.expert_cache_cpu_order;
    drive.d.split_rows = !o.no_split_rows;
    drive.d.pool = &pool;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // ---- R4.2c: THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< R4.2h: the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
        if (cudaMalloc(&hit_scratch, (size_t) sb) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_slot, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_dst, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34) != cudaSuccess ||
            // R4.2h: the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a 4.761e-04 relative disagreement on
            // every chunk, and the reason enabling the cache changed the tokens.
            cudaMalloc((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_out, (size_t) K * g.n_embd * 4) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the R4 hit path could not allocate its device buffers\n");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
#if defined(STRATA_ENABLE_NVFP4)
        if (nvfp4_experts) {
            // P2/P4c: the marlin hit path's scratch (locks, routing prep, topk_weights, marlin workspace),
            // allocated and initialised OUTSIDE any graph capture.
            int sms = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
            const uint64_t nsb = strata::kernels::moe_hit_nvfp4_scratch_bytes(K, sms);
            if (cudaMalloc(&nv4_scratch, (size_t) nsb) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: the NVFP4 hit scratch could not be allocated\n");
                return 1;
            }
            strata::kernels::moe_hit_nvfp4_scratch_init(nv4_scratch, K, sms, main_cs);
            drive.d.nvfp4_arena = &nv4arena;
            drive.d.nvfp4_scratch = nv4_scratch;
        }
#endif
        drive.d.h_slot.resize((size_t) K);
        cudaEvent_t hit_done = nullptr;
        if (cudaEventCreate(&hit_done) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the hit path could not create its probe event\n");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.hit_poke = !o.no_hit_poke;
        drive.d.h_dst.resize((size_t) K);
        mem_mark("the R4 hit path");
        std::fprintf(stderr, "strata generate: R4 hit path ON - resident experts are computed on the GPU\n");
    }
    // ---- P0.S8: the routing trace.  Only meaningful with the pool running, because the ids arrive through
    // the doorbell that the pool consumes - so `--no-pool` is refused rather than silently producing an empty
    // file that would read as "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently
    // dropped, so a captured graph cannot carry these events and the numbers would be zeros that read as
    // "every stage is free".  Refusing is the fix.
    if (o.stage_timing) {
        if (!o.no_capture) {
            std::fprintf(stderr, "strata generate: --stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.\n");
            return 2;
        }
        if (!strata::core::stage_timing_enable()) {
            std::fprintf(stderr, "strata generate: stage_timing_enable failed\n");
            return 1;
        }
        strata::core::stage_timing_name(0, "gr_read (attn)");
        strata::core::stage_timing_name(1, "attention block");
        strata::core::stage_timing_name(2, "gr_write (attn)");
        strata::core::stage_timing_name(3, "gr_read (ffn)");
        strata::core::stage_timing_name(4, "moe_route");
        strata::core::stage_timing_name(5, "moe_finish");
        strata::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals.  It is 36 of the 48 layers, 0.96 ms each, and its entire weight traffic
        // is ~26 MB - so ~0.11 ms at the measured read rate.  ~13 tiny latency-bound launches live in it and
        // a single "attention block" number cannot say which one costs anything.
        strata::core::stage_timing_name(8, "  gdn: quantize x");
        strata::core::stage_timing_name(9, "  gdn: qkv gemv");
        strata::core::stage_timing_name(10, "  gdn: conv+silu");
        strata::core::stage_timing_name(11, "  gdn: l2 norms");
        strata::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        strata::core::stage_timing_name(13, "  gdn: gdn_step");
        strata::core::stage_timing_name(14, "  gdn: z + out_norm");
        strata::core::stage_timing_name(15, "  gdn: out gemv");
    }
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            std::fprintf(stderr, "strata generate: --dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.\n");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    if (o.hot_learn) {
        g_hot.on = true;
        g_hot.n_layers = g.n_layers;
        g_hot.n_expert = g.n_expert;
        g_hot.pin.assign((size_t) g.n_layers * (size_t) g.n_expert, 0);
        std::fprintf(stderr, "strata generate: hot-learn ON (single-token usage, planar slot refill, "
                             "request pin). Default residency is unchanged without this switch.\n");
    }
    strata::core::PoolFn pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    strata::core::HitFn hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run;
    void* pool_user = o.no_pool ? nullptr : (void*) &drive;
    std::fprintf(stderr, "strata generate: %d expert-pool workers%s%s\n", pool.workers(),
                 pool.host_works() ? " + the host thread" : "",
                 o.no_pool ? " (UNUSED: --no-pool)" : "");

    // **THE MISALIGNMENT WARNING THAT STOOD HERE IS GONE, BECAUSE THE MISALIGNMENT IS FIXED.**
    //
    // It said the tokens were not the model's, and it was true: `session_loop` handed layer `l`'s expert
    // outputs to layer `l+1`, which multiplied them by layer `l+1`'s router weights (LEDGER L100).  The loop
    // now runs a captured PAIR per layer - `pre[l]` ending with the router and the doorbell, then the CPU
    // pool, then `post[l]` which combines those experts with THAT layer's weights - so layer `l`'s experts meet
    // layer `l`'s routing.  The generated ids changed the moment it landed, which is what a correctness fix
    // looks like from the outside.
    //
    // The cost is real and is recorded rather than hidden: the window for the CPU pool is now whatever GPU
    // work follows the ring inside `pre[l]`, which is the shared expert and nothing else - 0.038 ms against
    // 0.514 ms of CPU work per layer.  A per-layer CPU expert pool cannot be hidden behind a strictly serial
    // residual chain; the CPU term is answered by Phase 3's VRAM expert cache, not by this pipeline.

    // ---- the graphs
    strata::core::SessionGraphs gr;
    if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only
        if (!strata::core::session_capture(wt, g, ss, d_parts, gr, err, /*split=*/o.gpu_stages)) {
            std::fprintf(stderr, "strata generate: session_capture: %s\n", err.c_str());
            return 1;
        }
    }

    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        std::fprintf(stderr,
                     "strata generate: --no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.\n");
        return 2;
    }
    // The ladder is written by `session_loop`, and `session_token` does not touch the staging buffer at all - so
    // accepting the flag there would produce a file of uninitialised memory, which reads as a wrong answer rather
    // than as a mistake.  `--no-capture` without `--no-pool` is already refused above, so this catches the pair.
    if (o.no_capture && !o.dump_layers.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-layers is written by `session_loop`; `--no-capture` runs "
                     "`session_token` instead, which never fills the staging buffer. Drop one of the two.\n");
        return 2;
    }
    if (o.no_capture && !o.dump_halves.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-halves is CAPTURED into the layer graphs, so it needs the "
                     "captured path; `--no-capture` never records it. Drop one of the two.\n");
        return 2;
    }

    mem_mark("the expert cache and the graphs");
    std::fprintf(stderr, "strata generate: session is up (engine %s)\n", STRATA_VERSION);
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return strata::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return strata::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (cudaMalloc(&d_emb, (size_t) g.n_embd * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the embedding buffer failed\n");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (cudaMalloc(&d_next, sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the sampler output buffer failed\n");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    void* token_stream = o.stream_token ? main_cs : nullptr;
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        if (!strata::core::embed_row(wt, g, tok, d_emb, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return false;
        }
        if (pos == 0) {
            strata::core::session_zero(ss, g, d_emb, token_stream);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                if (cudaMemcpyAsync(ss.R + (size_t) c * g.n_embd, d_emb, (size_t) g.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, (cudaStream_t) token_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: the residual broadcast failed\n");
                    return false;
                }
        }
        return o.stream_token || cudaDeviceSynchronize() == cudaSuccess;
    };

    strata::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            std::fprintf(stderr, "strata generate: sampling greedy\n");
        else
            std::fprintf(stderr, "strata generate: sampling temperature=%g top_k=%d top_p=%g seed=%llu\n",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }

    std::FILE* dump = nullptr;
    const int64_t dump_positions = (int64_t) o.tokens.size() - 1 + o.max_new;
    if (!o.dump_logits.empty()) {
        if (dump_positions > INT32_MAX || n_vocab > INT32_MAX) {
            std::fprintf(stderr, "strata generate: logits dump dimensions exceed int32\n");
            return 2;
        }
        dump = std::fopen(o.dump_logits.c_str(), "wb");
        if (dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_logits.c_str());
            return 1;
        }
        // **THE COUNT IS `n_prompt - 1 + max_new`, NOT `n_prompt + max_new`.**  The loop writes one row per
        // position from 0, and it stops once `produced` holds `max_new` tokens - and `produced` only starts
        // receiving at position `n_prompt - 1`.  So a 5-token prompt with `--max-new 6` writes 10 rows, and the
        // header used to claim 11.  A header that describes a different file from the one written is the same
        // class of defect as a self-check that verifies the wrong invariant: anything reading the count instead
        // of the size gets a wrong answer that looks authoritative.  `tools/logits_identical.py` caught it by
        // parsing the header and refusing the file.
        const int32_t n_rows = (int32_t) strata::program::logits_selection::row_count(dump_positions, o.logits_stride);
        const int32_t hdr[2] = {(int32_t) n_vocab, n_rows};
        if (std::fwrite(hdr, sizeof hdr, 1, dump) != 1) {
            std::fprintf(stderr, "strata generate: cannot write logits header\n");
            std::fclose(dump);
            return 1;
        }
    }

    // ---- THE C1 ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    std::FILE* layer_dump = nullptr;
    float* layer_stage = nullptr;
    const size_t layer_floats = (size_t) (g.n_layers + 1) * (size_t) g.hc * (size_t) g.n_embd;
    if (!o.dump_layers.empty()) {
        layer_dump = std::fopen(o.dump_layers.c_str(), "wb");
        if (layer_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_layers.c_str());
            return 1;
        }
        if (cudaHostAlloc((void**) &layer_stage, layer_floats * sizeof(float), cudaHostAllocDefault) !=
            cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot pin the layer-dump staging buffer\n");
            return 1;
        }
    }

    // ---- prefill is the DECODE PATH ONE TOKEN AT A TIME, which `phase-2-correct-engine.md:12-13` says is
    // fine here: "process the prompt through the decode-style graphs in small batches; a 19K-token prompt will
    // take minutes".  A real batched prefill is P2.S6's other half and is not this.
    //
    // **THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE
    // LAST *PROMPT* POSITION.**  The first version sampled only on the decode positions, so `produced` was
    // still empty when the first generated position asked for `produced.back()` - an out-of-bounds read on an
    // empty vector.  Teacher forcing below is what makes the distinction unnecessary to special-case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- R0.9: THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        strata::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 are GDN and 12 are QSA and a total cannot
        // separate them.  Round 309's uncaptured table put GDN at 10.88 ms for 36 layers against QSA's 4.56 for
        // 12, which would make the recurrence the largest single R3 target - and that table had `moe_finish`
        // wrong by 4x, so the ratio is re-derived here from the captured graph rather than inherited.
        {
            // **ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED.**  The first version called the
            // per-layer replay a single time and printed `gdn / reps`, which reported GDN at 0.480 ms/token
            // against a mixer total of 14.039 - a factor of exactly `reps`, and the tell was that
            // 0.480 + 0.220 = 0.700 = 14.039 / 20.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (strata::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ R0.11: THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // the same independent-restatement test that caught round 320's divide-by-reps bug, and it is the only
        // reason to believe a table built out of differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f   <- R3.3's target is <= 3 ms for all four passes\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ R3.5c: THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!strata::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                            return 1;
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        strata::core::session_graphs_free(gr);
        strata::core::doorbell_free(db);
        cudaFree(d_next);
        return 0;
    }

    if (o.graph_only) {
        strata::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            std::fprintf(stderr, "strata generate: session_replay warm: %s\n", err.c_str());
            return 1;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session_replay warm faulted\n");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay: %s\n", err.c_str());
                return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session_replay faulted: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms per layer\n", ms / (double) g.n_layers);
        return 0;
    }

    // ---- R0.3: THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the
    // shared expert, the combine and the second `gr_write` - and the LM head.  Everything this project published
    // as "39.8 ms pure GPU" came from that loop while being described as the whole GPU, which also made the
    // "host's share = 49.4 - 39.8 = 9.6 ms" figure wrong by however much the missing work costs.  See
    // Memory/ERRORS.md A4/A5.
    //
    // This flag is what "pure GPU" has to mean, and it replaces that number everywhere.  No pool runs, so
    // `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        strata::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!strata::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay_full: %s\n", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: gpu-only-full layers faulted: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full lm_head: %s\n", err.c_str());
                return 1;
            }
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: gpu-only-full head faulted: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
    }

    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // `session_loop` used to allocate its pinned staging buffer, its probe event and its host pin ON EVERY
    // TOKEN, and `cudaFreeHost` at the end of each call implicitly synchronises the device - so every token
    // finished with a device-wide sync nobody asked for.  The scratch is created once here and reused; it also
    // owns the host pin for the whole session rather than taking and releasing it per token.
    strata::core::SessionLoopScratch loop_scratch;
    struct ScratchFree {
        strata::core::SessionLoopScratch* p;
        ~ScratchFree() { if (p != nullptr) p->free(); }
    } scratch_free{&loop_scratch};
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at setup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    // Plan v0.3 P3: the whole token as ONE graph whenever nothing needs a host step between the ring and post[l]
    // (the VRAM expert tier and the per-layer dumps do).  `--no-token-graph` keeps two graphs per layer.
    strata::core::TokenGraph tgraph;
    struct TokenGraphFree {
        strata::core::TokenGraph* p;
        ~TokenGraphFree() { strata::core::token_graph_free(*p); }
    } tgraph_free{&tgraph};
    // Plan v0.3 P4: with a PROFILE-filled cache the residency is static, so the hit decision moves onto the
    // device and the token graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    std::vector<int32_t> host_res;
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    strata::core::TokenHits thits;
    const bool graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), strata::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != strata::core::kNotResident) ++resident;
            }
        if (cudaMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_count, sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the device residency table could not be staged\n");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const strata::core::OnDevice on(st->dev);
            if (cudaMalloc((void**) &st->d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(st->d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                std::fprintf(stderr, "strata generate: layer split: CUDA%d residency table failed\n", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        if (nvfp4_experts) {
            // P2: non-null switches the captured graph's hit kernel to the marlin path (on_nvfp4()).
            thits.nvfp4_arena = &nv4arena;
            thits.nvfp4_scratch = nv4_scratch;
        }
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "strata generate: token graph hit path: %lld resident experts, decided on the device\n",
                     (long long) resident);
    }
    if (!o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr &&
        (hit_fn == nullptr || thits.on() || thits.on_nvfp4()) && !native_pack) {
        // W1: when push mode will arm below, the graph scatters the worker's contiguous result block on the
        // device instead of copying y_miss whole (the two must agree, so the condition matches arm_push's).
        const float* y_contig = (rdma_tier.is_open() && o.remote_rdma_push) ? loop_scratch.y_contig : nullptr;
        if (!strata::core::session_capture_token(wt, g, ss, d_parts, loop_scratch.y_miss, loop_scratch.parts_bytes,
                                                 tgraph, err, (thits.on() || thits.on_nvfp4()) ? &thits : nullptr,
                                                 y_contig)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: token graph captured (48 layers, one launch per token)\n");
    }

    // RDMA push mode arms AFTER the session exists: the doorbell activation, y_miss and the doorbell flag
    // are the buffers the worker will one-sided-read and RDMA-write, and they only exist now.  From here on
    // the host's verbs are out of the layer loop (the pull path stays for verify windows).
    if (rdma_tier.is_open() && o.remote_rdma_push) {
        if (!tgraph.captured) {
            std::fprintf(stderr, "strata generate: --remote-rdma push mode needs the token graph (drop "
                                 "--no-capture/--no-token-graph) or pass --no-remote-rdma-push\n");
            return 1;
        }
        if (!rdma_tier.arm_push(db.h_x_f, (size_t) g.n_embd * sizeof(float), loop_scratch.y_miss,
                                loop_scratch.parts_bytes, db.h_flag, loop_scratch.y_contig,
                                256 + loop_scratch.parts_bytes, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        db.external_flag = true;
        std::fprintf(stderr, "strata generate: RDMA push mode: the worker writes y_miss and the doorbell "
                             "flag - the host's verbs are out of the layer loop\n");
    }

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // **`--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.**  It never enters
    // this loop, so it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the
    // NaN scan or the sampler - and `--no-pool --stats` against that floor was being read as "the per-layer
    // round trip costs 12.4 ms" when an unknown part of it is per TOKEN, not per layer.  That is the same error
    // the review catalogued as A4/A5, one level down: a difference between two measurements attributed to a
    // mechanism that neither of them isolates.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  **The
    // boundary after the layer loop is the one that matters**: without it the head's interval swallows all 48
    // layers and the report reads as "head = 50 ms", which is not a thing that can happen to 1.4 ms of GPU
    // work.  That is not hypothetical - it is what the first version of this printed.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused>
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk, and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's streamed ring and its lend cap)
    if (srcp != nullptr && o.prefill_chunk > 0) {
        uint64_t pinned = 0, total = 0;
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        strata::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    }
    auto lend_slots = [&](int64_t c) -> int64_t {
        const uint64_t need = strata::prefill::Prefill::bytes_needed(g, ss, c);
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t k = (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        if (xcache.slot_offsets() != nullptr) {   // sized slots: take slots from the end until they hold `need`
            k = 0;
            while (k < xcache.slots() &&
                   (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[xcache.slots() - k]) < need) ++k;
        }
        return k;
    };
    auto lend_bytes = [&](int32_t first) -> uint64_t {
        return xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                     : (uint64_t) (xcache.slots() - first) *
                                           (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
    };
    auto request_chunk = [](int64_t tokens, int64_t max_chunk) -> int64_t {
        if (tokens <= 0 || max_chunk <= 0) return 0;
        const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                    ? tokens
                                    : ((tokens + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
    // or with --prefill auto the largest of kAutoChunks whose buffers take at most kAutoLendPct % of the slots (a
    // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
    // prompt: 4096 791 tok/s, 6144 878, 8192 973 with 69% of the slots lent).  A request lends only what its own
    // prompt needs (Prefill::relayout), so a big chunk costs short prompts nothing.  0 = none fits.
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        static constexpr int64_t kAutoChunks[] = {8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
        // at 8192-token chunks nearly every expert streams anyway, so a lent slot costs little: 90% when the
        // copies are DMA from pinned RAM (Q2_0 8192 + a 384-slot ring: 1283 tok/s), 85% when host copies are the
        // limit (lending more only streams more through them).  STRATA_PREFILL_LEND_PCT overrides (tuning).
        const int64_t kAutoLendPct = [] {
            const char* v = std::getenv("STRATA_PREFILL_LEND_PCT");
            return v ? (int64_t) std::atoi(v)
                     : (int64_t) (strata::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85);
        }();
        auto slots_for = lend_slots;
        if (o.prefill_auto) {
            for (const int64_t c : kAutoChunks) {
                const int64_t k = slots_for(c);
                if (k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots()) { chunk = c; return k; }
            }
            return 0;
        }
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = slots_for(c);
            if (k + 128 <= xcache.slots()) { chunk = c; return k; }
        }
        return 0;
    };
    if (o.serve) {
        // non-speculative serve: without --spec the request loop below decodes on the plain per-token path
        // (session_run_token/session_loop + head + sampler, as `strata generate` does) instead of verify windows
        const bool spec_serve = o.spec >= 2 && (!o.mtp.empty() || !o.dflash.empty());
        if ((o.spec > 0 && !spec_serve) || o.prefill_chunk <= 0 ||
            (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
            std::fprintf(stderr, "strata serve: needs --spec T, --mtp DIR or --dflash PATH, and --prefill CHUNK (and a fillable "
                                 "--expert-cache; the graphed hit path additionally needs --expert-profile P)\n");
            return 2;
        }
        if (!spec_serve)
            std::fprintf(stderr, "strata serve: no --spec: decoding without speculation (the plain token path)\n");
        // Batch ladder phase 2 (the EAGER half): the second decode session's doorbell, its pinned handoff and
        // its push channel exist from startup, because the worker's bind records only flow at session setup.
        // The VRAM-heavy half (session arena, token graph, prompt path, PLE buffers) stays lazy on the first
        // GEN2.  Fail-soft: a failure here only disables GEN2, never the single-request path.
        bool g2_arm_ok = false;
        strata::core::Doorbell g2_db1;
        strata::core::SessionLoopScratch g2_scratch1;
        cudaStream_t g2_cs1 = nullptr;
        {
            std::string g2e;
            bool ok = strata::core::doorbell_init(g, K, g2_db1) != 0;
            if (ok) ss_all[1].db = &g2_db1;
            if (ok) ok = g2_scratch1.init((size_t) K * g.n_embd * 4, g2e);
            if (ok && cudaStreamCreateWithFlags(&g2_cs1, cudaStreamNonBlocking) != cudaSuccess) ok = false;
            g2_arm_ok = ok;
        }
        strata::prefill::Prefill sp;
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
        int32_t lend_first_now = -1;      // where its buffers are laid out now
        // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card the
        // cache already filled to its reserve, that is the over-subscription the auto sizing avoids - so the
        // prompt chunk is halved until its buffers fit in the lendable slots (a smaller chunk only reads slower)
        if (!o.no_prefill_borrow && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            if (const int64_t k = plan_lend(chunk); k > 0) {
                if (o.prefill_auto)
                    std::fprintf(stderr, "strata serve: prompt chunk auto: %lld tokens\n", (long long) chunk);
                else if (chunk != o.prefill_chunk)
                    std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens so its buffers fit in the "
                                         "expert cache\n", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
                lend_first = (int32_t) (xcache.slots() - k);
                lend_first_now = lend_first;
                borrow = xcache.device_slot(lend_first);
                borrow_bytes = xcache.slot_offsets()
                                   ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[lend_first])
                                   : (uint64_t) k * (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
            } else if (o.prefill_auto) {
                o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
            }
        } else if (o.prefill_auto && d_res == nullptr) {
            o.prefill_chunk = 1024;       // #85: no expert cache at all (a full 8 GB card): small buffers of its own
        }
        if (borrow != nullptr)
            std::fprintf(stderr, "strata serve: the prompt path borrows %lld cache slots (%.2f GiB)\n",
                         (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
        else
            std::fprintf(stderr, "strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
        // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next
        for (size_t i = 0; i < stages.size(); ++i) {
            GpuStage& st = *stages[i];
            st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
            const strata::core::OnDevice on(st.dev);
            if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream, err)) {
                std::fprintf(stderr, "strata serve: layer split, CUDA%d prompt path: %s\n", st.dev, err.c_str());
                return 1;
            }
        }
        if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
        // RDMA push channel 1: one window covers a whole chunk of one layer, capped at kMaxPushTok tokens.
        if (rdma_tier.push_armed() && o.prefill_chunk > strata::core::RDMAExpertTier::kMaxPushTok) {
            o.prefill_chunk = strata::core::RDMAExpertTier::kMaxPushTok;
            std::fprintf(stderr, "strata serve: remote prefill windows cap the chunk at %lld tokens\n",
                         (long long) strata::core::RDMAExpertTier::kMaxPushTok);
        }
        sp.set_nvfp4_hot(nv4arena.on() ? &nv4arena : nullptr);
        if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
                     rdma_tier.push_armed() ? &rdma_tier : nullptr)) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            if (err.find("fit") != std::string::npos)   // #85: say what frees VRAM
                std::fprintf(stderr, "strata serve: the GPU has too little free VRAM for the prompt path: turn images "
                                     "off (setup: --vision no), close other programs using the GPU, use a shorter "
                                     "context, or read prompts in smaller chunks (--prefill 512)\n");
            return 1;
        }
        // The channel-3 bind goes out here, AFTER sp.init: the inbox is one mailbox and the older binds
        // were always spaced by seconds of setup work, so the second decode session's bind takes the same
        // spacing (plus its own explicit beat inside arm_push_ch3).
        if (g2_arm_ok && rdma_tier.is_open() && o.remote_rdma_push) {
            std::string g2e;
            if (rdma_tier.arm_push_ch3(g2_db1.h_x_f, (size_t) g.n_embd * sizeof(float), g2_scratch1.y_contig,
                                       256 + g2_scratch1.parts_bytes, g2_db1.h_flag, g2e)) {
                g2_db1.external_flag = true;   // the worker raises session B's flag
                std::fprintf(stderr, "strata serve: GEN2 slot-1 push channel armed\n");
            } else {
                std::fprintf(stderr, "strata serve: GEN2 push channel 3 not armed: %s\n", g2e.c_str());
                g2_arm_ok = false;
            }
        }
        mem_mark("the head and the prompt path");
        // ================================ BATCH LADDER PHASE 2: THE GEN2 SLOT-1 RESOURCES ================================
        //
        // Everything a SECOND decode slot needs beyond the eager arm above, allocated lazily on the first
        // GEN2 request so production single-stream runs never pay for it: the slot-1 session arena, its
        // parts/logits/sampler/embed buffers, a private set of hit-path outputs (two graphs in flight cannot
        // share hit_out / nvfp4 scratch / the hit counters), its PLE I/O buffers, and its token graph.  Slot
        // A keeps its production graph and its channel 0 untouched; slot B pushes on channel 3.
        bool g2_ready = false;
        void* g2_sbuf1 = nullptr;
        strata::core::TokenGraph g2_tg1;
        float* g2_d_parts1 = nullptr;
        float* g2_d_logits1 = nullptr;
        int* g2_d_next1 = nullptr;
        float* g2_d_emb1 = nullptr;
        float* g2_hit_out1 = nullptr;
        int32_t* g2_d_slot1 = nullptr;
        int32_t* g2_d_dst1 = nullptr;
        int32_t* g2_d_count1 = nullptr;
        uint8_t* g2_d_hit_q8_1 = nullptr;
        float* g2_d_hit_q8_scale_1 = nullptr;
        void* g2_hit_scratch1 = nullptr;
        void* g2_nv4_scratch1 = nullptr;
        float* g2_ple_emb_dev1 = nullptr;
        float* g2_ple_scratch1 = nullptr;
        std::vector<float> g2_ple_emb_host1;
        strata::core::TokenHits g2_thits1;
        auto g2_ensure = [&]() -> bool {
            if (g2_ready) return true;
            if (!tgraph.captured) {
                err = "GEN2 needs the token graph (drop --no-capture/--no-token-graph)";
                return false;
            }
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            if (cudaMalloc(&g2_sbuf1, kSessionBytes) != cudaSuccess) {
                err = "GEN2: the slot-1 session state (" + std::to_string((long long) (kSessionBytes >> 20)) +
                      " MiB) does not fit (" + std::to_string((long long) (free_b >> 20)) +
                      " MiB free) - restart with a smaller --expert-cache";
                return false;
            }
            if (strata::core::session_init(g, o.max_context, K, g2_sbuf1, ss_all[1]) == 0) {
                err = "GEN2: slot-1 session_init failed";
                return false;
            }
            if (!g2_arm_ok) {
                err = "GEN2 is not armed (the slot-1 push channel failed at startup - see the serve log)";
                return false;
            }
            if (cudaMalloc(&g2_d_parts1, (size_t) K * g.n_embd * 4) != cudaSuccess ||
                cudaMemset(g2_d_parts1, 0, (size_t) K * g.n_embd * 4) != cudaSuccess ||
                cudaMalloc(&g2_d_logits1, (size_t) n_vocab * 4) != cudaSuccess ||
                cudaMalloc(&g2_d_next1, sizeof(int)) != cudaSuccess ||
                cudaMalloc(&g2_d_emb1, (size_t) g.n_embd * 4) != cudaSuccess) {
                err = "GEN2: slot-1 stream or device buffers failed";
                return false;
            }
            g2_thits1 = thits;   // the read-only halves stay shared: residency table, arena, cache planes
            if (thits.on() || thits.on_nvfp4()) {
                if (cudaMalloc((void**) &g2_d_slot1, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
                    cudaMalloc((void**) &g2_d_dst1, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
                    cudaMalloc((void**) &g2_d_count1, sizeof(int32_t)) != cudaSuccess ||
                    cudaMalloc((void**) &g2_hit_out1, (size_t) K * g.n_embd * 4) != cudaSuccess) {
                    err = "GEN2: slot-1 hit-path buffers failed";
                    return false;
                }
                g2_thits1.d_slot = g2_d_slot1;
                g2_thits1.d_dst = g2_d_dst1;
                g2_thits1.d_count = g2_d_count1;
                g2_thits1.hit_out = g2_hit_out1;
#if defined(STRATA_ENABLE_NVFP4)
                if (nvfp4_experts) {
                    int sms = 0;
                    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
                    const uint64_t nsb = strata::kernels::moe_hit_nvfp4_scratch_bytes(K, sms);
                    if (cudaMalloc(&g2_nv4_scratch1, (size_t) nsb) != cudaSuccess) {
                        err = "GEN2: slot-1 NVFP4 hit scratch failed";
                        return false;
                    }
                    strata::kernels::moe_hit_nvfp4_scratch_init(g2_nv4_scratch1, K, sms, main_cs);
                    g2_thits1.nvfp4_scratch = g2_nv4_scratch1;
                } else
#endif
                {
                    const uint64_t sb =
                        strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
                    if (cudaMalloc(&g2_hit_scratch1, (size_t) sb) != cudaSuccess ||
                        cudaMalloc((void**) &g2_d_hit_q8_1, (size_t) (g.n_embd / 32) * 34) != cudaSuccess ||
                        cudaMalloc((void**) &g2_d_hit_q8_scale_1, (size_t) (g.n_embd / 32) * sizeof(float)) !=
                            cudaSuccess) {
                        err = "GEN2: slot-1 grouped hit buffers failed";
                        return false;
                    }
                    g2_thits1.scratch = g2_hit_scratch1;
                    g2_thits1.x_q8 = g2_d_hit_q8_1;
                    g2_thits1.x_scale = g2_d_hit_q8_scale_1;
                }
            }
            if (ss.ple.ready() && std::getenv("STRATA_G2_B_NOPLE") == nullptr) {
                g2_ple_emb_host1.assign((size_t) strata::kernels::NG_N_EMBD * 2, 0.0f);
                if (cudaMalloc((void**) &g2_ple_emb_dev1, (size_t) strata::kernels::NG_N_EMBD * 4) != cudaSuccess ||
                    cudaMalloc((void**) &g2_ple_scratch1, strata::core::ple_run_scratch_bytes()) != cudaSuccess) {
                    err = "GEN2: slot-1 PLE buffers failed";
                    return false;
                }
                ss_all[1].ple.w = ss.ple.w;   // weights and table are read-only and shared
                ss_all[1].ple.consts = ss.ple.consts;
                ss_all[1].ple.table = ss.ple.table;
                ss_all[1].ple.ahead = ss.ple.ahead;
                ss_all[1].ple.token = &ss_all[1].ple_token;
                ss_all[1].ple.prev = ss_all[1].ple_prev;
                ss_all[1].ple.hist = ss_all[1].ple_hist;
                ss_all[1].ple.emb_host = g2_ple_emb_host1.data();
                ss_all[1].ple.emb_host_alt = g2_ple_emb_host1.data() + (size_t) strata::kernels::NG_N_EMBD;
                ss_all[1].ple.emb_dev = g2_ple_emb_dev1;
                ss_all[1].ple.scratch = g2_ple_scratch1;
                if (!ss_all[1].ple.ready()) {
                    err = "GEN2: slot-1 PLE did not arm";
                    return false;
                }
            }
            // Slot A keeps its PRODUCTION graph (channel 0, unchanged).  Slot B's graph scatters the
            // channel-3 contiguous result block when that channel is armed - the same W1 shape as slot A's,
            // only the regions differ (the capture condition must match the arm, as for arm_push).
            // STRATA_G2_B_NOPUSH=1 (debug): capture B pull-shaped instead (copy_from_mapped from its own
            // y_miss) - with paB.pull it bypasses the push request block and the channel-3 regions entirely.
            const float* g2_y_contig =
                (rdma_tier.is_open() && o.remote_rdma_push && rdma_tier.push_ch3_armed() &&
                 std::getenv("STRATA_G2_B_NOPUSH") == nullptr)
                    ? g2_scratch1.y_contig
                    : nullptr;
            // STRATA_G2_B_NOHITS=1 (debug): capture B without the GPU hit path - B's tokens are then wrong
            // by construction; slot A's tokens are the oracle for whether the marlin kernels on the shared
            // NVFP4 planes are what collides between the two graphs.
            const strata::core::TokenHits* g2_hits_arg =
                (g2_thits1.on() || g2_thits1.on_nvfp4()) && std::getenv("STRATA_G2_B_NOHITS") == nullptr
                    ? &g2_thits1
                    : nullptr;
            if (!strata::core::session_capture_token(wt, g, ss_all[1], g2_d_parts1, g2_scratch1.y_miss,
                                                     g2_scratch1.parts_bytes, g2_tg1, err, g2_hits_arg,
                                                     g2_y_contig)) {
                err = "GEN2: slot-B graph: " + err;
                return false;
            }
            // NO second Prefill object: arm_push_prefill is once-per-process, so channel 1 stays bound to
            // sp's regions; GEN2's slot-1 prompts go through sp with set_session (see g2_prefill).
            g2_ready = true;
            std::fprintf(stderr, "strata serve: GEN2 dual-slot decode is live (slot-1 arena %lld MiB, %.1f GiB free)\n",
                         (long long) (kSessionBytes >> 20), (double) free_b / 1073741824.0);
            return true;
        };
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
        // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
        // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
        // neutral path (no upload, no buffer handed to the sampler).
        constexpr int kPenaltyWindowCap = 4096;
        constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) strata::kernels::kVerifyMaxT;
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
        if (const strata::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: the penalty-history allocation failed\n");
            return 1;
        }
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
        // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
        // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
        strata::core::Verifier ver_same;
        SplitDrive split_drive;
        auto stage_ver = [&](int st) -> strata::core::Verifier& {
            return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
        };
        auto pcie_num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
        const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
        if (n_stages > 1) {
            const size_t hb = (size_t) strata::kernels::kVerifyMaxT *
                              (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
            std::vector<float*> hand((size_t) n_stages - 1, nullptr);
            for (float*& h : hand) {
                float* hh = nullptr;
                if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                    cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: the layer-split hand-off allocation failed\n");
                    return 1;
                }
                std::memset(hh, 0, hb);
            }
            split_drive.base = &drive;
            split_drive.n = n_stages;
            for (int st = 0; st < n_stages; ++st) {
                stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                        st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
                split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
                split_drive.cache_base[st] = drive.d.cache_base;
                split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
                split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
            }
            for (int st = 1; st < n_stages; ++st) {
                bool ok_s = false;
                if (split_same) {
                    ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err);
                } else {
                    GpuStage& gs = *stages[(size_t) st - 1];
                    const strata::core::OnDevice on(gs.dev);
                    strata::core::VerifyHits vs;
                    vs.d_res = gs.d_res;
                    vs.cache_base = gs.cache.device_slot(0);
                    vs.blob = thits.blob;
                    vs.slot_off = gs.cache.slot_offsets();
                    vs.n_slots = gs.cache.slots();
                    ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, o.spec, err);
                    split_drive.cache_base[st] = gs.cache.device_slot(0);
                    split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                    split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
                }
                if (!ok_s) {
                    std::fprintf(stderr, "strata serve: layer split, stage %d: %s\n", st + 1, err.c_str());
                    return 1;
                }
            }
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
            std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
            for (int st = 1; st < n_stages; ++st)
                plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                          " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
            std::fprintf(stderr, "strata serve: layer split: layers %s, one hand-off per window\n", plan_s.c_str());
        }
        if (spec_serve &&
            (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err) ||
             (!o.mtp.empty() && !mtp.bind(last_st ? last_st->wt : wt, last_st ? &last_st->head : &native_head,
                                          ver.final_R_all(), err)))) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            return 1;
        }
        if (spec_serve && dflash.loaded()) ver.set_dflash_taps(dflash.tap_buffer(), dflash.tap_stride());
        // RDMA push channel 2: the verify window's misses stop driving the wire from this host too - the
        // worker one-sided-reads the window's activation segment and RDMA-writes the rows straight into the
        // window's y_miss segment, then raises the window's own flag.  Without it a verify window's
        // begin/finish is a verbs round trip per layer, which is what made --spec collapse on this host.
        if (spec_serve && rdma_tier.push_armed()) {
            int64_t vx_bytes = 0, vy_bytes = 0;
            const float* vx = ver.push_x(vx_bytes);
            float* vy = ver.push_y(vy_bytes);
            if (!rdma_tier.arm_push_verify(vx, (size_t) vx_bytes, vy, (size_t) vy_bytes, ver.push_flag(), err)) {
                std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata serve: RDMA push channel 2 armed: verify windows are push windows "
                                 "(%lld tokens, %lld rows)\n", (long long) (vx_bytes / ((int64_t) g.n_embd * 4)),
                         (long long) (vy_bytes / ((int64_t) g.n_embd * 4)));
        }
        for (int st = 0; spec_serve && st < n_stages && n_stages > 1; ++st) {
            split_drive.plan[st] = stage_ver(st).plan_sink();
            if (st > 0) {
                stage_ver(st).set_split(o.spec_split);
                stage_ver(st).set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
            }
        }
        // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
        const strata::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
        void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
        mem_mark("the verifier and the drafter's binding");
        if (spec_serve) ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        if (spec_serve) ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        std::vector<int64_t> cur;
        // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens
        // it has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the
        // saved points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain -
        // the radix cache's tree collapsed onto the one branch of history whose cells the session holds.  The
        // chain's root is the deepest point every request so far shared (the end of the system prompt, in
        // practice); the retention policy pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that
        // shares that prefix mounts through it instead of reading it again.
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs, req_imgs;
        bool live_ok = false;
        std::vector<ConvCheckpoint> checks;
        uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
        Clock::time_point pp_t0 = Clock::now();
        auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
            std::vector<ImgKey> v;
            for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
            return v;
        };
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
        // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
        // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
        // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
        std::mutex part_mu;
        std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
        std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
        // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
        auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr) -> bool {
            if (o.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) { c.used = ++check_clock; return true; }
            ConvCheckpoint c;
            c.ids.assign(cur.begin(), cur.begin() + L);
            c.imgs = imgs_below(req_imgs, L);
            if (parts != nullptr) {
                if (parts->size() != stages.size() + 1) return false;
                c.gdn = std::move((*parts)[0].gdn);
                c.ple = std::move((*parts)[0].ple);
                c.tails = std::move((*parts)[0].tails);
                for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
            } else {
                if (cudaDeviceSynchronize() != cudaSuccess || !checkpoint_save(c, ss, g)) return false;
                for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                    const strata::core::OnDevice on(st->dev);
                    ConvCheckpoint part;
                    if (cudaDeviceSynchronize() != cudaSuccess || !checkpoint_save(part, st->ss, g)) return false;
                    c.stage_parts.push_back(std::move(part));
                }
            }
            c.used = ++check_clock;
            checks.push_back(std::move(c));
            while ((int) checks.size() > o.prompt_cache) {
                std::vector<uint64_t> stamps;
                stamps.reserve(checks.size());
                for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
                const size_t victim = strata::program::conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                                   o.prompt_cache);
                checks.erase(checks.begin() + (std::ptrdiff_t) victim);
            }
            return true;
        };
        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            std::vector<int32_t> nxt((size_t) T);
            for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
            if (spec_serve && o.dflash.empty() && !mtp.prefill(R_rows, nxt.data(), T, p0, e)) return false;
            // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
            const int64_t done = p0 + T;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
            strata::core::progress_at("reading the prompt (batched), done up to token", done);
            strata::core::progress_beat();
            std::fflush(stdout);
            if (o.prompt_cache_every > 0 && done >= pp_next_check) {
                bool saved = false;
                if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                    std::vector<ConvCheckpoint> parts;
                    {
                        std::lock_guard<std::mutex> lk(part_mu);
                        auto it = part_at.find(done);
                        if (it != part_at.end()) parts = std::move(it->second);
                        part_at.erase(part_at.begin(), part_at.upper_bound(done));
                    }
                    bool complete = parts.size() == stages.size() + 1;
                    for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                    saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
                } else {
                    saved = checkpoint_at(done);
                }
                if (!saved) { e = "saving a conversation checkpoint failed"; return false; }
                pp_next_check = done + o.prompt_cache_every;
            }
            return true;
        };
        if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
            stages.back()->sp.on_chunk = std::move(sp.on_chunk);
            sp.on_chunk = nullptr;
            for (size_t i = 0; i <= stages.size(); ++i) {
                strata::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
                strata::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
                stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                    part_next[i] = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                    if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto& v = part_at[done];
                    v.resize(stages.size() + 1);
                    v[i] = std::move(part);
                    return true;
                };
            }
        }
        if (spec_serve) drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        cudaStream_t adapt_stream = nullptr;
        if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
        auto res_upload = [&]() {
            if (d_res != nullptr)
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                cudaMemcpy(st->d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            }
        };
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
            for (auto& st : stages)
                if (st->adapt_live) {
                    if (wait) cudaEventSynchronize(st->adapt_ev);
                    else if (cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
                }
            for (auto& st : stages) st->adapt_live = false;
            for (const auto& [i, slot] : pending) host_res[(size_t) i] = slot;
            pending.clear();
            res_upload();
        };
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
        auto adapt = [&]() -> bool {
#if defined(STRATA_ENABLE_NVFP4)
            // Between windows only (the caller does not overlap this with a decode-graph replay).
            // The planar fill is synchronous before return, so the next graph sees the new slot.
            if (o.hot_learn && xcache.planar()) {
                std::string e;
                int64_t nswap = 0;
                if (!hot_learn_planar_adapt(xcache, drive.d.usage, host_res, d_res, nv4_gm13, nv4_gm2,
                                            const_cast<uint16_t*>(nv4arena.gm13),
                                            const_cast<uint16_t*>(nv4arena.gm2), cold_slot_of, srcp,
                                            g.n_layers, g.n_expert, o.adapt_swaps, adapt_stream, nswap, e)) {
                    std::fprintf(stderr, "strata serve: %s\n", e.c_str());
                    return false;
                }
                (void) nswap;
                return true;
            }
#endif
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                    else if (!hot_pinned(l, e)) vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            bool main_live = false;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const strata::core::OnDevice on(gs ? gs->dev : -1);
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,
                                    (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, gs ? gs->adapt_stream : adapt_stream) != cudaSuccess)
                    return false;
                if (gs) gs->adapt_live = true;
                else main_live = true;
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            (void) main_live;
            for (auto& st : stages)
                if (st->adapt_live) {
                    const strata::core::OnDevice on(st->dev);
                    cudaEventRecord(st->adapt_ev, st->adapt_stream);
                }
            for (float& v : drive.d.usage) v *= 0.7f;
            return true;
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        std::atomic<bool> stop_req{false};
        std::mutex in_mu;
        std::condition_variable in_cv;
        std::deque<std::string> in_lines;
        bool in_eof = false;
        std::thread([&] {
            // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
            // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
            // std::exit) would hang in exit() on Linux, and the server would wait for it forever
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop_req.store(true); continue; }
                std::lock_guard<std::mutex> lk(in_mu);
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(in_mu);
            in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        sp.should_stop = [&] { return stop_req.load(); };
        // STRATA_TRACE=1: one stderr line per step of a request (the log shows where a request stops)
        const bool trace = std::getenv("STRATA_TRACE") != nullptr;
        auto tr = [&](const char* what, long long a = -1, long long b = -1) {
            if (!trace) return;
            std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
            std::fflush(stderr);
        };
        {
            // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
            // and a page-in while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
            // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
            const int64_t free_mib = (int64_t) (free_b >> 20);
            if (free_mib >= 256) {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded\n", (long long) free_mib);
            } else {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded - LOW: requests may stall;"
                                     " add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                             (long long) free_mib, (long long) (o.vram_reserve_mib + 512 - free_mib));
            }
        }
        // what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            std::printf("INFO context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "spec_min_p=%.2f engine=" STRATA_VERSION "\n",
                        (long long) o.max_context, o.kv.c_str(),
                        (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[0].kv_mode == 1
                                         ? ss.qsa_states[0].n_slots * 4 : 0),
                        (long long) xcache.slots(), (long long) (xcache.bytes() >> 20), o.spec, o.mtp_max_t,
                        o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                        (long long) (strata::kernels::cpu::expert_layout().total >> 20), pool.workers(), o.pcie_frac,
                        o.spec_min_p);
        }
        // issue #29: a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on
        // a flag nobody will raise - end the engine with where it was, so the server starts it again instead of the
        // GPU spinning forever.  STRATA_WATCHDOG_S=0 turns it off.  Issue #31: before it does, it reports what every
        // part was doing (stall_report), so one occurrence says where the wait is.
        {
            const char* ws = std::getenv("STRATA_WATCHDOG_S");
            const int limit = ws ? std::atoi(ws) : 60;   // one step (a prompt layer, a verify window) takes seconds
            if (limit > 0)
                std::thread([limit] {
                    strata::core::Progress& p = strata::core::progress();
                    uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                    auto since = std::chrono::steady_clock::now();
                    for (;;) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        const auto now = std::chrono::steady_clock::now();
                        const uint64_t b = p.beats.load();
                        if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                        if (now - since < std::chrono::seconds(limit)) continue;
                        std::fprintf(stderr, "strata serve: no progress for %d s during a request (%s %lld) - stopping "
                                             "the engine so the server starts it again (issue #29)\n",
                                     limit, p.where.load(), (long long) p.detail.load());
                        stall_report(stderr, p.ticks.load() - ticks_at);
                        std::fflush(stderr);
                        std::abort();
                    }
                }).detach();
        }
        std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
        std::fflush(stdout);
        std::string line;
        int64_t rounds = 0;
        const int S = o.spec;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
        if (S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(S);   // MTP or lookup window, learned over the whole process
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        while (next_line(line)) {
            if (line == "QUIT") break;
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() { strata::core::progress().busy.store(true); strata::core::progress_at("request"); }
                ~BusyScope() { strata::core::progress().busy.store(false); strata::core::progress_at("idle"); }
            } busy_scope;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            if (line.rfind("GEN2 ", 0) == 0) {
                // Batch ladder phase 2: `GEN2 <max_new> <idsA> | <idsB>` — two prompts read in turn (slot 0,
                // then slot 1), then decoded TOGETHER: both token graphs stay in flight on their own streams
                // and the host serves the two doorbells in arrival order (session_run_token_dual).  Slot A
                // pushes on channel 0, slot B on channel 3; greedy sampling, no prompt cache, no images, no
                // speculation.  Output: `T0 <tok>` / `T1 <tok>` lines as they are drawn, then one
                // `DONE2 <madeA> <madeB> <ms>` line.
                if (spec_serve) {
                    std::printf("ERR GEN2 needs the non-speculative serve (drop --spec)\n");
                    std::fflush(stdout);
                    continue;
                }
                hot_learn_clear_pin();
                char* end2 = nullptr;
                const long long g2_max_new = std::strtoll(line.c_str() + 5, &end2, 10);
                std::vector<int64_t> idsA, idsB;
                {
                    const char* bar = std::strchr(line.c_str() + 5, '|');
                    std::string pe2;
                    if (bar == nullptr || end2 == line.c_str() + 5) {
                        std::printf("ERR bad GEN2: expected `GEN2 <max_new> <idsA> | <idsB>`\n");
                        std::fflush(stdout);
                        continue;
                    }
                    const std::string listA(end2, (size_t) (bar - end2)), listB(bar + 1);
                    if (g2_max_new < 1 || !parse_i64_list(listA.c_str(), idsA, pe2) ||
                        !parse_i64_list(listB.c_str(), idsB, pe2)) {
                        std::printf("ERR bad GEN2: %s\n", pe2.empty() ? "max_new" : pe2.c_str());
                        std::fflush(stdout);
                        continue;
                    }
                }
                const int64_t nA = (int64_t) idsA.size(), nB = (int64_t) idsB.size();
                if (nA < 1 || nB < 1 || nA - 1 + g2_max_new > o.max_context ||
                    nB - 1 + g2_max_new > o.max_context) {
                    std::printf("ERR bad GEN2: a prompt plus %lld new tokens exceeds --max-context %lld\n",
                                (long long) g2_max_new, (long long) o.max_context);
                    std::fflush(stdout);
                    continue;
                }
                if (!g2_ensure()) {
                    std::printf("ERR GEN2: %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                const Clock::time_point g2_t0 = Clock::now();
                // Self-contained lend/refill (the request path's lend/refill lambdas live further down and are
                // slot-0 only): lend the prompt path its slots, read the prompt, give the slots back.
                std::vector<std::pair<int32_t, int32_t>> g2_lent;
                auto g2_refill = [&]() -> bool {
                    for (const auto& [i, slot] : g2_lent) {
                        const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                        const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                        if (b == nullptr || !xcache.fill_slot_queued(slot, b, err, nb)) return false;
                        host_res[(size_t) i] = slot;
                    }
                    if (!xcache.sync_queued(err)) return false;
                    if (!g2_lent.empty())
                        cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice);
                    g2_lent.clear();
                    return true;
                };
                // STRATA_G2_STATE=1 (debug): fingerprint each slot's session state between the two prefills and
                // after them - identical prompts must give identical hashes, and slot 1's prefill must not move
                // slot 0's.  The implementation is shared with the single-request path (state_hash_dump).
                auto g2_state_hash = [&](strata::core::SessionState& s, const char* tag, int64_t L) {
                    state_hash_dump(g, s, tag, L);
                };
                int32_t g2_first_now_unused = 0;   // layout tracking lives in the shared lend_first_now now
                (void) g2_first_now_unused;
                auto g2_prefill = [&](int slot_i, const int64_t* pids, int64_t cnt) -> bool {
                    // ONE prompt path shared with the single-request path: the RDMA push channel-1 bind is
                    // once-per-process (it stays registered to sp's buffers), so a second Prefill would push
                    // into the first one's regions.  set_session retargets the session between prompts; the
                    // layout tracker is the production lend_first_now, so GEN and GEN2 cannot go stale on
                    // each other either.
                    sp.set_session(ss_all[slot_i]);
                    strata::prefill::Prefill& spx = sp;
                    if (cnt <= 0) {   // a one-token prompt reads nothing; the decode's first token covers it
                        strata::core::session_zero(ss_all[slot_i], g, nullptr, main_cs);
                        cudaStreamSynchronize(main_stream);
                        return true;
                    }
                    if (lend_first >= 0) {
                        const int64_t want = request_chunk(cnt, o.prefill_chunk);
                        if (want <= 0) {
                            err = "GEN2: cannot lend buffers for an empty prompt";
                            return false;
                        }
                        const int64_t slots = lend_slots(want);
                        if (slots <= 0 || slots > xcache.slots() - lend_first) {
                            err = "GEN2: prompt buffers exceed the configured lend region";
                            return false;
                        }
                        const int32_t first =
                            std::max<int32_t>(lend_first, (int32_t) (xcache.slots() - slots));
                        if (want != spx.chunk() || first != lend_first_now) {
                            if (!spx.relayout(want, xcache.device_slot(first), lend_bytes(first), err))
                                return false;
                            lend_first_now = first;
                        }
                        for (size_t i = 0; i < host_res.size(); ++i)
                            if (host_res[i] >= first) {
                                g2_lent.emplace_back((int32_t) i, host_res[i]);
                                host_res[i] = strata::core::kNotResident;
                            }
                        cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice);
                    }
                    strata::core::session_zero(ss_all[slot_i], g, nullptr, main_cs);
                    cudaStreamSynchronize(main_stream);
                    return spx.run(pids, cnt, 0, err);
                };
                if (!g2_prefill(0, idsA.data(), nA - 1) || !g2_refill()) {
                    std::printf("ERR GEN2 prefill: %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                g2_state_hash(ss_all[0], "S0-after-prefill0", nA - 1);
                if (!g2_prefill(1, idsB.data(), nB - 1) || !g2_refill()) {
                    std::printf("ERR GEN2 prefill: %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                g2_state_hash(ss_all[0], "S0-after-prefill1", nA - 1);
                g2_state_hash(ss_all[1], "S1-after-prefill1", nB - 1);
                // Both slots push: slot A on channel 0 (its production graph and flag, untouched), slot B
                // on channel 3 (the RXE5 regions; the worker raises g2_db1's flag).  The dual runner's
                // publish-ack spin keeps the shared request block safe.
                PoolAt paA{&drive, rdma_tier.is_open() ? &rdma_tier : nullptr, false, 0};
                PoolAt paB{&drive, rdma_tier.is_open() ? &rdma_tier : nullptr, false, 3};
                // STRATA_G2_PULL=1 (A/B): force the legacy pull for both slots - y then lands in each slot's
                // own y_miss via RDMA read, nothing shared but the connection (the push request block and
                // the channel-3 regions are bypassed entirely).
                if (std::getenv("STRATA_G2_PULL") != nullptr) {
                    paA.pull = true; paB.pull = true;
                    // pull is synchronous on the pool call - the worker never raises a flag, so the host
                    // must (the dual runner does when external_flag is false).
                    db.external_flag = false;
                    g2_db1.external_flag = false;
                }
                if (std::getenv("STRATA_G2_B_NOPUSH") != nullptr) {
                    // B alone pulls (its graph was captured pull-shaped); A keeps the production push.
                    paB.pull = true;
                    g2_db1.external_flag = false;
                }
                struct G2Slot {
                    strata::core::SessionState* s;
                    strata::core::TokenGraph* tg;
                    float* y_miss;
                    void* cs;
                    float* d_logits;
                    int* d_next;
                    float* d_emb;
                    const std::vector<int64_t>* ids;
                    int64_t p;
                    int32_t x;
                    int64_t made;
                    bool done;
                    strata::kernels::SamplerParams smp;
                    const char* tag;
                };
                G2Slot SA{&ss_all[0], &tgraph, loop_scratch.y_miss, main_cs, d_logits, d_next, d_emb,
                          &idsA, nA - 1, (int32_t) idsA[(size_t) (nA - 1)], 0, false,
                          strata::kernels::SamplerParams{}, "T0"};
                G2Slot SB{&ss_all[1], &g2_tg1, g2_scratch1.y_miss, (void*) g2_cs1, g2_d_logits1, g2_d_next1,
                          g2_d_emb1, &idsB, nB - 1, (int32_t) idsB[(size_t) (nB - 1)], 0, false,
                          strata::kernels::SamplerParams{}, "T1"};
                for (G2Slot* slp : {&SA, &SB}) {
                    slp->smp.greedy = true;
                    slp->s->ple_prev[0] = slp->p >= 2 ? (int32_t) (*slp->ids)[(size_t) (slp->p - 2)] : -1;
                    slp->s->ple_prev[1] = slp->p >= 1 ? (int32_t) (*slp->ids)[(size_t) (slp->p - 1)] : -1;
                }
                auto g2_put_input = [&](G2Slot& sl) -> bool {
                    if (!strata::core::embed_row(wt, g, sl.x, sl.d_emb, sl.cs, err)) return false;
                    if (sl.p == 0) {
                        strata::core::session_zero(*sl.s, g, sl.d_emb, sl.cs);
                        return true;
                    }
                    for (int64_t c = 0; c < g.hc; ++c)
                        if (cudaMemcpyAsync(sl.s->R + (size_t) c * g.n_embd, sl.d_emb, (size_t) g.n_embd * 4,
                                            cudaMemcpyDeviceToDevice, (cudaStream_t) sl.cs) != cudaSuccess) {
                            err = "GEN2: the residual broadcast failed";
                            return false;
                        }
                    return true;
                };
                auto g2_run_head = [&](G2Slot& sl) -> bool {
                    if (!native_head.loaded())
                        return strata::core::lm_head(wt, g, sl.s->block, sl.d_logits, sl.cs, err);
                    return strata::core::lm_head_mix(wt, g, sl.s->block, sl.cs, err) &&
                           native_head.run(sl.s->block.mixed, sl.d_logits, sl.cs, err);
                };
                // STRATA_G2_DEBUG=1: first-round logits fingerprint per slot, for the divergence bisect.
                static const bool g2_debug = std::getenv("STRATA_G2_DEBUG") != nullptr;
                auto g2_dbg = [&](const char* tag, int64_t round, const float* dev, void* stream) {
                    if (!g2_debug) return;
                    float h[16];
                    if (cudaMemcpyAsync(h, dev, sizeof h, cudaMemcpyDeviceToHost, (cudaStream_t) stream) !=
                            cudaSuccess ||
                        cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess) return;
                    int am = 0;
                    float mx[4] = {h[0], h[1], h[2], h[3]};
                    double sum = 0;
                    for (float v : h) sum += v;
                    std::fprintf(stderr,
                                 "G2DBG %s round %lld logits[0..3] %.6f %.6f %.6f %.6f sum16 %.6f\n", tag,
                                 (long long) round, mx[0], mx[1], mx[2], mx[3], sum);
                    (void) am;
                };
                bool g2_ok = true;
                // The PLE TABLE holds one read ticket at a time, so only one slot may have an ahead issue
                // outstanding: the flag gates the end-of-round ahead issues, and each round's finish phase
                // collects whatever was outstanding before any new issue.
                bool g2_ticket = false;
                while (g2_ok && ((!SA.done && SA.made < g2_max_new) || (!SB.done && SB.made < g2_max_new))) {
                    const bool a_on = !SA.done && SA.made < g2_max_new && SA.p + 1 <= o.max_context;
                    const bool b_on = !SB.done && SB.made < g2_max_new && SB.p + 1 <= o.max_context;
                    if (!a_on && !b_on) break;
                    g2_ticket = false;
                    for (G2Slot* slp : {&SA, &SB}) {
                        G2Slot& sl = *slp;
                        if (!(slp == &SA ? a_on : b_on)) continue;
                        if (!sl.s->ple.inflight) sl.s->ple_token = sl.x;
                        else if (sl.s->ple_token != sl.x) {
                            err = "GEN2: ple ahead prefetched a different token";
                            g2_ok = false;
                            break;
                        }
                        if (sl.s->ple.ready() && !sl.s->ple.inflight &&
                            !strata::core::ple_issue_token(sl.s->ple, err)) {
                            g2_ok = false;
                            break;
                        }
                        if (!g2_put_input(sl)) {
                            g2_ok = false;
                            break;
                        }
                        if (sl.s->ple.ready() && !strata::core::ple_finish_token(sl.s->ple, sl.cs, err)) {
                            g2_ok = false;
                            break;
                        }
                    }
                    if (!g2_ok) break;
                    drive.d.failed = false;
                    err.clear();
                    // STRATA_G2_SERIAL=1 (debug): serve the two slots one after the other through the same
                    // dual runner (single-sided calls).  If the tokens are then right, the corruption comes
                    // from the two graphs flying concurrently, not from the dual path's own bookkeeping.
                    const bool g2_serial = std::getenv("STRATA_G2_SERIAL") != nullptr;
                    const bool g2_step =
                        g2_serial
                            ? ((!a_on || strata::core::session_run_token_dual(
                                   g, SA.p, 0, ss_all[0], tgraph, SB.p, 0, ss_all[1], g2_tg1, &drive_pool_at,
                                   &paA, &paB, loop_scratch.y_miss, g2_scratch1.y_miss, true, false, main_cs,
                                   (void*) g2_cs1, err)) &&
                               (!b_on || strata::core::session_run_token_dual(
                                   g, SA.p, 0, ss_all[0], tgraph, SB.p, 0, ss_all[1], g2_tg1, &drive_pool_at,
                                   &paA, &paB, loop_scratch.y_miss, g2_scratch1.y_miss, false, true, main_cs,
                                   (void*) g2_cs1, err)))
                            : strata::core::session_run_token_dual(
                                  g, SA.p, /*pos_base=*/0, ss_all[0], tgraph, SB.p, /*pos_base=*/0, ss_all[1],
                                  g2_tg1, &drive_pool_at, &paA, &paB, loop_scratch.y_miss, g2_scratch1.y_miss,
                                  a_on, b_on, main_cs, (void*) g2_cs1, err);
                    if (!g2_step || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail != nullptr) err = drive.d.fail;
                        g2_ok = false;
                        break;
                    }
                    for (G2Slot* slp : {&SA, &SB}) {
                        G2Slot& sl = *slp;
                        if (!(slp == &SA ? a_on : b_on)) continue;
                        if (!g2_run_head(sl)) {
                            g2_ok = false;
                            break;
                        }
                        g2_dbg(sl.tag, sl.made, sl.d_logits, sl.cs);
                        sl.smp.counter = (uint64_t) sl.p;
                        strata::kernels::sample_tokens(sl.d_logits, 1, (int) n_vocab, nullptr, 0, sl.smp,
                                                       sl.d_next, sl.cs);
                        int next = -1;
                        if (cudaMemcpyAsync(&next, sl.d_next, sizeof(int), cudaMemcpyDeviceToHost,
                                            (cudaStream_t) sl.cs) != cudaSuccess ||
                            cudaStreamSynchronize((cudaStream_t) sl.cs) != cudaSuccess || next < 0 ||
                            next >= n_vocab) {
                            err = "GEN2: the sampler readback failed";
                            g2_ok = false;
                            break;
                        }
                        std::printf("%s %d\n", sl.tag, next);
                        std::fflush(stdout);
                        ++sl.made;
                        if (std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end())
                            sl.done = true;
                        sl.s->ple_prev[0] = sl.s->ple_prev[1];
                        sl.s->ple_prev[1] = sl.x;
                        sl.x = next;
                        ++sl.p;
                        if (!sl.done && !g2_ticket && sl.s->ple.ahead && sl.s->ple.ready() &&
                            !sl.s->ple.inflight && sl.made < g2_max_new && sl.p < o.max_context) {
                            sl.s->ple_token = sl.x;
                            if (!strata::core::ple_issue_token(sl.s->ple, err)) {
                                g2_ok = false;
                                break;
                            }
                            g2_ticket = true;
                        }
                    }
                }
                // STRATA_G2_ROUNDS=1 (debug): per-round state fingerprints.  The dual runner already syncs
                // both streams at the round boundary, so dumping here does not change the race inside it.
                if (std::getenv("STRATA_G2_ROUNDS") != nullptr) {
                    char rt[32];
                    std::snprintf(rt, sizeof rt, "S0-round-%lld", (long long) SA.p);
                    state_hash_dump(g, ss_all[0], rt, SA.p);
                    std::snprintf(rt, sizeof rt, "S1-round-%lld", (long long) SB.p);
                    state_hash_dump(g, ss_all[1], rt, SB.p);
                }
                if (rdma_tier.is_open()) rdma_tier.set_decode_channel(0);
                // Drain any ahead PLE issue the round loop left outstanding: the table ticket is one per
                // process, and the next single GEN's first finish would otherwise collect a row read that
                // was issued for THIS sequence's token (the ST_REFA2 contamination).
                for (G2Slot* slp : {&SA, &SB}) {
                    if (slp->s->ple.ready() && slp->s->ple.inflight) {
                        std::string ple_e;
                        strata::core::ple_finish_token(slp->s->ple, slp->cs, ple_e);
                    }
                }
                if (!g2_ok) {
                    std::printf("ERR GEN2: %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                const double g2_ms = std::chrono::duration<double, std::milli>(Clock::now() - g2_t0).count();
                g2_state_hash(ss_all[0], "S0-end", SA.p);
                g2_state_hash(ss_all[1], "S1-end", SB.p);
                std::printf("DONE2 %lld %lld %.1f\n", (long long) SA.made, (long long) SB.made, g2_ms);
                std::fflush(stdout);
                continue;
            }
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) {
                std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n");
                continue;
            }
            hot_learn_clear_pin();
            char* endp = nullptr;
            const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
            // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
            // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
            // only).  Absent keys keep today's behavior: greedy, no penalties.
            float req_temperature = 0.0f, req_top_p = 1.0f;
            int req_top_k = 20;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64
            unsigned long long req_seed = 0;
            float req_min_p = 0.0f, req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_penalty_last_n = 0;
            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
            int req_seq = 0;   // batch ladder: the session slot this request runs on; only slot 0 exists today
            // tuning keys (setup's calibration measures settings without restarting the engine): the PCIe share of
            // the missed experts and the draft-probability floor, for this request only
            double req_pcie_frac = o.pcie_frac, req_spec_min_p = o.spec_min_p;
            if (endp != nullptr) {   // GENI takes the same keys (#75: image requests were always greedy); its
                                     // embedding file path is the first token without an =
                for (;;) {
                    while (*endp == ' ') ++endp;
                    const char* start = endp;
                    while (*endp != '\0' && *endp != ' ') ++endp;
                    if (endp == start) break;
                    const std::string tok(start, (size_t) (endp - start));
                    const size_t eq = tok.find('=');
                    if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                    const std::string key = tok.substr(0, eq);
                    const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
                    if (key == "cvec") req_cvec = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "seq") req_seq = std::clamp(std::atoi(tok.c_str() + eq + 1), 0, MAX_BATCH - 1);
                    else if (key == "temperature") req_temperature = fv;
                    else if (key == "top_p") req_top_p = fv;
                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "min_p") req_min_p = fv;
                    else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "penalty_repeat") req_penalty_repeat = fv;
                    else if (key == "penalty_freq") req_penalty_freq = fv;
                    else if (key == "penalty_present") req_penalty_present = fv;
                    else if (key == "seed") req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
                    else if (key == "pcie_frac") req_pcie_frac = std::clamp((double) fv, 0.0, 1.0);
                    else if (key == "spec_min_p") req_spec_min_p = std::clamp((double) fv, 0.0, 1.0);
                    // unknown keys are skipped: the ids start at the first token without '='
                }
            }
            if (req_seq != 0) {   // batch ladder: slots 1..MAX_BATCH-1 arrive with the decode-interleave phase
                std::printf("ERR seq=%d is not live yet (only session slot 0)\n", req_seq);
                continue;
            }
            std::string emb_path;
            if (geni && endp != nullptr) {
                while (*endp == ' ') ++endp;
                char* gap = std::strchr(endp, ' ');
                if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
            }
            std::vector<int64_t> ids;
            std::string pe;
            if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !parse_i64_list(endp, ids, pe)) {
                std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
                continue;
            }
            const int64_t n = (int64_t) ids.size();
            req_imgs.clear();
            if (geni && !o.vision) { std::printf("ERR this engine was started without --vision\n"); continue; }
            if (geni || !mrope_identity) {
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    struct Img { int64_t n, nx, ny; size_t off; };
                    std::vector<Img> imgs;
                    img_rows.clear();
                    std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                    if (!f) ve = "cannot open " + emb_path;
                    while (f && ve.empty()) {
                        int32_t hdr[5];
                        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                        if (got == 0) break;
                        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                            ve = "bad embeddings file (expected strata-vision records of width " +
                                 std::to_string((long long) g.n_embd) + ")";
                            break;
                        }
                        const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                        img_rows.resize(off + cnt);
                        if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                    }
                    if (f) std::fclose(f);
                    int64_t p = 0, i = 0;
                    size_t k = 0;
                    while (ve.empty() && i < n) {
                        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                        if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                        const Img& im = imgs[k++];
                        {   // what the conversation cache compares: a picture is its grid and its embeddings
                            const int64_t grid[3] = {im.n, im.nx, im.ny};
                            uint64_t h = fnv1a(grid, sizeof grid);
                            h = fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * sizeof(float), h);
                            req_imgs.push_back({i, h});
                        }
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                     " rows but fewer <|image_pad|> tokens";
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                            const int64_t y = j / im.nx, x = j % im.nx;
                            put(i + j, p, p + y, p + x);
                            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                        }
                        i += im.n;
                        p += std::max(im.nx, im.ny);
                    }
                    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                    for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
                }
                tr("positions built", (long long) img_rows.size());
                cudaDeviceSynchronize();
                tr("device idle");
                // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
                auto upload_mrope = [&]() -> bool {
                    bool ok = cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                         cudaMemcpyHostToDevice) == cudaSuccess;
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        cudaDeviceSynchronize();
                        ok = ok && cudaMemcpy(st->mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                              cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    return ok;
                };
                if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    upload_mrope();
                    mrope_identity = true;
                    std::printf("ERR %s\n", ve.c_str());
                    std::fflush(stdout);
                    continue;
                }
                mrope_identity = !geni;
            }
            sp.embd_rows = geni ? row_ptr.data() : nullptr;
            if (n + max_new + 8 > o.max_context) {
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            (long long) max_new, (long long) o.max_context);
                continue;
            }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); continue; }
            std::array<int64_t, 3> remote_before{};
            std::array<int64_t, 3> launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            {
                remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
                launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
                compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
                full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
            }
            cur = ids;
            const Clock::time_point r0 = Clock::now();
            // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
            // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
            auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
                const int64_t L = (int64_t) pre.size();
                if (L < 1 || L > n - 1) return false;
                for (int64_t i = 0; i < L; ++i)
                    if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
                return imgs_below(req_imgs, L) == pre_imgs;
            };
            // the control vector for this request.  The live session and the checkpoints were read one way, so a
            // switch reads the prompt again from the start
            if (strata::kernels::cvec().loaded()) {
                const bool want = req_cvec != 0;
                if (want != cvec_cached) {
                    live_ok = false;
                    checks.clear();
                    cvec_cached = want;
                }
                strata::kernels::cvec_set_enabled(want);
            }
            int64_t resume = 0;
            bool from_live = false;
            if (o.prompt_cache > 0) {
                if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                        resume = (int64_t) c.ids.size();
                        from_live = false;
                    }
            }
            // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
            // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
            checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                             return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                         }), checks.end());
            live_ok = false;   // until this request has finished, the session is in between
            int64_t reread_to = -1;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring
            if (resume == 0) {
                strata::core::session_zero(ss, g, nullptr, main_cs);
                cudaStreamSynchronize(main_stream);
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                    cudaStreamSynchronize(st->stream);
                }
                checks.clear();
            } else if (!from_live) {
                ConvCheckpoint* c = nullptr;
                for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
                if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
                static const bool reread = std::getenv("STRATA_CKPT_REREAD") != nullptr;
                if (reread && c != nullptr) {
                    // THE CHECK OF THE CHECKPOINT: instead of restoring it, read its tokens again from position 0 in
                    // one run (below, with the prompt path's slots lent like any read) - the same chunks the request
                    // that saved it read them in, when that request started at 0.  With the VRAM expert set fixed
                    // (--adapt-swaps 0) the answer must match the restored one token for token; anything the
                    // checkpoint missed shows up as a difference.
                    strata::core::session_zero(ss, g, nullptr, main_cs);
                    cudaStreamSynchronize(main_stream);
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                        cudaStreamSynchronize(st->stream);
                    }
                    reread_to = resume;
                    std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: reading %lld tokens again instead of "
                                         "restoring\n", (long long) resume);
                } else if (c == nullptr || !checkpoint_restore(*c, ss, g) || c->stage_parts.size() != stages.size() ||
                           [&] {
                               for (size_t i = 0; i < stages.size(); ++i) {
                                   const strata::core::OnDevice on(stages[i]->dev);
                                   if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                               }
                               return false;
                           }()) {
                    std::printf("ERR restoring a conversation checkpoint failed\n");
                    return 1;
                }
            }
            // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
            // host copies and slots are always current (every writer writes both), so they need nothing
            if (spec_serve && resume > 0 && reread_to <= 0) mtp.kv_restore(resume);
            tr("request", n, geni ? 1 : 0);
            if (spec_serve) mtp.set_prompt_len(n);
            const int64_t read_from = reread_to > 0 ? 0 : resume;
            pp_total = n;
            pp_from = read_from;
            pp_t0 = r0;
            pp_next_check = reread_to > 0 ? INT64_MAX : resume + o.prompt_cache_every;
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), pp_next_check);
            }
            std::printf("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
            strata::core::progress_at("reading the prompt, from token", read_from);
            std::fflush(stdout);
            // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
            // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path
            // costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is
            // not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a
            // token, with the misses on the CPU.  Each part below is decided on its own, so a long first message is
            // read batched and its header still goes through the windows.  Picture rows need the batched path.
            // STRATA_CKPT_REREAD compares a restored checkpoint with a batched re-read, so it keeps every read batched.
            static const bool no_short = std::getenv("STRATA_CKPT_REREAD") != nullptr;
            auto windows_ok = [&](int64_t a, int64_t b) -> bool {
                if (!spec_serve) return false;   // no verify windows without --spec: short reads go batched
                if (no_short || (o.dflash.empty() && b - a > o.short_read)) return false;
                if (sp.embd_rows != nullptr)
                    for (int64_t i = a; i < b; ++i)
                        if (sp.embd_rows[i] != nullptr) return false;
                return true;
            };
            // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
            auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
                // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
                struct NoHeadSampling {
                    strata::core::Verifier& v;
                    explicit NoHeadSampling(strata::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                    ~NoHeadSampling() { v.set_head_sampling(true); }
                } no_head_sampling(ver);
                std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
                for (int64_t q = a; q < b;) {
                    if (stop_req.load()) { e = "cancelled"; return false; }
                    const int T = (int) std::min<int64_t>(S, b - q);
                    for (int t = 0; t < T; ++t) {
                        win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                        nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                    }
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                        return false;
                    }
                    if (!ver.commit(T, e)) return false;
                    if (dflash.loaded()) {
                        if (!dflash.ingest_window(T, q, e)) return false;
                    } else if (!mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e)) return false;
                    q += T;
                }
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                strata::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // the batched path's slots are lent just before its first run and given back (refilled) before a window
            // reads - so the windows always see the whole expert cache - or once the prompt is read
            std::vector<std::pair<int32_t, int32_t>> lent_now;
            int64_t lent_chunk = 0;   // the chunk the lent slots hold the prompt path's buffers for
            auto refill = [&](std::string& e) -> bool {
                if (lent_now.empty()) return true;
                tr("refill start", (long long) lent_now.size());
                for (const auto& [i, slot] : lent_now) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                    const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                    const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                    if (b == nullptr || !(refill_blocking() ? xcache.fill_slot_blocking(slot, b, e, nb)
                                                            : xcache.fill_slot_queued(slot, b, e, nb)))
                        return false;
                    host_res[(size_t) i] = slot;
                }
                if (!xcache.sync_queued(e)) return false;
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
                lent_now.clear();
                lent_chunk = 0;
                return true;
            };
            // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
            // rounded up to 256), laid out in the last of the slots it may borrow
            auto lend = [&](int64_t tokens, std::string& e) -> bool {
                if (lend_first < 0) return true;                       // its own buffers: nothing to lend
                const int64_t want = request_chunk(tokens, o.prefill_chunk);
                if (want <= 0) {
                    e = "prefill: cannot lend buffers for an empty request segment";
                    return false;
                }
                if (!lent_now.empty()) {
                    if (want <= lent_chunk) return true;
                    if (!refill(e)) return false;
                }
                const int64_t slots = lend_slots(want);
                if (slots <= 0 || slots > xcache.slots() - lend_first) {
                    e = "prefill: request-sized buffers exceed the configured lend region";
                    return false;
                }
                const int32_t first = std::max<int32_t>(lend_first, (int32_t) (xcache.slots() - slots));
                if (want != sp.chunk() || first != lend_first_now) {
                    if (!sp.relayout(want, xcache.device_slot(first), lend_bytes(first), e)) return false;
                    lend_first_now = first;
                }
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent_now.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
                lent_chunk = want;
                return true;
            };
            apply_pending(true);
            sp.set_session(ss);   // GEN2's dual prefill shares this prompt path - take it back for slot 0
            // per-request sampling for the verify window's head (greedy when temperature is absent)
            strata::kernels::SamplerParams req_sp;
            req_sp.greedy = req_temperature <= 0.0f;
            req_sp.temperature = req_temperature;
            req_sp.top_p = req_top_p;
            req_sp.top_k = req_top_k;
            req_sp.seed = req_seed ? req_seed
                                   : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
            req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
            req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
            req_sp.penalty_repeat = req_penalty_repeat;
            req_sp.penalty_freq = req_penalty_freq;
            req_sp.penalty_present = req_penalty_present;
            req_sp.counter = 0;
            if (spec_serve) ver.set_sampling(req_sp);
            drive.d.pcie_num = std::max(0, std::min(256, (int) (req_pcie_frac * 256.0 + 0.5)));
            // a layer split: CUDA0's share as asked; a later GPU keeps its own (its link) unless the request sets one
            for (int st = 0; st < split_drive.n; ++st)
                split_drive.pcie_num[st] = (st == 0 || split_same || req_pcie_frac != o.pcie_frac)
                                               ? drive.d.pcie_num : pcie_num_of(stages[(size_t) st - 1]->pcie_frac);
            const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            if (spec_serve) ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            bool cancelled = false;
            tr("prompt start", n - 1);
            // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
            // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
            // chat renders the same history - but not always the same header or the thinking of this reply - so that
            // checkpoint is the one it reuses.
            int64_t turn_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0)
                for (int64_t i = n - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
            // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with
            // the tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root,
            // which the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.
            // (PR #65, code-martin.)  Only for a system prompt of --prompt-cache-root tokens or more: a small one
            // is cheaper to read again than the extra part costs (~0.3 s).
            int64_t root_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && read_from == 0)
                for (int64_t i = 1; i < turn_at; ++i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
            int64_t at = read_from;
            for (const int64_t to : {reread_to, root_at, turn_at, n - 1}) {
                if (to <= at) continue;
                const bool win = windows_ok(at, to);
                if (win && !refill(err)) {
                    std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                if (!win && !lend(to - at, err)) {
                    std::printf("ERR lending the prompt path its slots failed: %s\n", err.c_str());
                    return 1;
                }
                const auto tsp = Clock::now();
                const bool sp_ok = win ? read_windows(at, to, err) : sp.run(ids.data() + at, to - at, at, err);
                if (trace) {
                    std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\n", (long long) (to - at),
                                 win ? "windows" : "batched",
                                 std::chrono::duration<double, std::milli>(Clock::now() - tsp).count());
                    std::fflush(stderr);
                }
                if (!sp_ok) {
                    if (!stop_req.load()) {
                        std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                    break;
                }
                at = to;
                if ((to == turn_at || to == root_at) && !checkpoint_at(to)) {
                    std::printf("ERR saving a conversation checkpoint failed\n");
                    return 1;
                }
            }
            if (!refill(err)) {
                std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                return 1;
            }
            tr("prompt done (slots refilled)");
            state_hash_dump(g, ss, "GEN-after-prefill", n - 1);
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
            std::fflush(stdout);
            // the verify windows: the first holds the last prompt token alone
            int64_t p = n - 1;
            int32_t x = (int32_t) ids[(size_t) (n - 1)];
            std::vector<int32_t> drafts((size_t) S, 0), window((size_t) S), outv((size_t) S);
            std::vector<float> dprob((size_t) S, 0.0f);
            std::vector<int32_t> sbuf((size_t) S, 0);
            if (o.suffix_draft > 0) {
                sfx.reset();
                for (int64_t t : ids) sfx.append((int32_t) t);
            }
            bool first_window = true;
            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
            int64_t draft_offered = 0, draft_accepted = 0;
            // what the session holds once this request is done: the prompt read so far, then every committed token
            std::vector<int32_t> consumed;
            consumed.reserve((size_t) (n + max_new + S));
            for (int64_t i = 0; i < n - 1; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            const char* finish = "length";
            const Clock::time_point d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
            struct DecSnap {
                double wait, pool, host, plan, actq, jobs, run;
                int64_t misses, entries, hits, pcie;
            };
            auto dec_snap = [&]() {
                if (!spec_serve) return DecSnap{};
                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                               drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                               drive.d.pcie_experts};
            };
            const DecSnap ds0 = dec_snap();
            double dt_run = 0, dt_commit = 0, dt_draft = 0;
            int64_t dec_windows = 0, dec_T = 0;
            const int64_t decode_hits0 = drive.d.cache_hits;
            const int64_t decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            if (cancelled) finish = "cancel";
            if (!spec_serve) {
                // ---- non-speculative decode: the same per-token path `strata generate` runs (PLE window,
                // session_run_token/session_loop, head, sampler), emitting one T line per token.  No verify
                // windows and no drafter: everything the window path needed --spec for is absent here.
                ss.ple_prev[0] = p >= 2 ? (int32_t) ids[(size_t) (p - 2)] : -1;
                ss.ple_prev[1] = p >= 1 ? (int32_t) ids[(size_t) (p - 1)] : -1;
                while (!cancelled && produced_n < max_new) {
                    if (p + 1 > o.max_context) break;
                    // A prefetched ticket already hashed this id. The first token of the request has none.
                    if (!ss.ple.inflight) ss.ple_token = x;
                    else if (ss.ple_token != x) {
                        std::printf("ERR ple ahead prefetched %d but this token is %d\n", ss.ple_token, x);
                        return 1;
                    }
                    if (ss.ple.ready() && !ss.ple.inflight && !strata::core::ple_issue_token(ss.ple, err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    if (!put_input(x, p)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    if (ss.ple.ready() && !strata::core::ple_finish_token(ss.ple, token_stream, err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    err.clear();
                    tr("window", p, 1);
                    strata::core::doorbell_reset(db);
                    if (tgraph.captured) {
                        if (!strata::core::session_run_token(g, p, /*pos_base=*/0, ss, tgraph, pool_fn, pool_user,
                                                             loop_scratch.y_miss, main_cs, err)) {
                            std::printf("ERR %s\n", err.c_str());
                            return 1;
                        }
                    } else if (!strata::core::session_loop(g, p, /*pos_base=*/0, ss, gr, pool_fn, hit_fn, pool_user,
                                                           /*overlap=*/true, main_cs, err, layer_stage, &loop_scratch)) {
                        std::printf("ERR session_loop: %s\n", err.c_str());
                        return 1;
                    }
                    if (drive.d.failed) {
                        std::printf("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed");
                        return 1;
                    }
                    if (!run_head(token_stream)) {
                        std::printf("ERR lm_head: %s\n", err.c_str());
                        return 1;
                    }
                    if (std::getenv("STRATA_G2_DEBUG") != nullptr && produced_n == 0) {
                        float h[16];
                        if (cudaMemcpyAsync(h, d_logits, sizeof h, cudaMemcpyDeviceToHost,
                                            (cudaStream_t) token_stream) == cudaSuccess &&
                            cudaStreamSynchronize((cudaStream_t) token_stream) == cudaSuccess) {
                            double sum = 0;
                            for (float v : h) sum += v;
                            std::fprintf(stderr, "G2DBG T round 0 logits[0..3] %.6f %.6f %.6f %.6f sum16 %.6f\n",
                                         h[0], h[1], h[2], h[3], sum);
                        }
                    }
                    // the draw is Philox(seed, position), as in `strata generate` and the verify windows
                    req_sp.counter = (uint64_t) p;
                    strata::kernels::sample_tokens(d_logits, 1, (int) n_vocab, nullptr, 0, req_sp, d_next, token_stream);
                    int next = -1;
                    if (cudaMemcpyAsync(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost,
                                        (cudaStream_t) token_stream) != cudaSuccess ||
                        cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess) {
                        std::printf("ERR reading the sampled token back failed\n");
                        return 1;
                    }
                    if (next < 0 || next >= n_vocab) {
                        std::printf("ERR the sampler returned %d, outside 0..%lld\n", next, (long long) (n_vocab - 1));
                        return 1;
                    }
                    std::printf("T %d\n", next);
                    strata::core::progress_beat();
                    ++produced_n;
                    consumed.push_back(x);
                    if (o.suffix_draft > 0) sfx.append(next);
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end();
                    std::fflush(stdout);
                    ++rounds;
                    if (eos) { finish = "stop"; break; }
                    if (stop_req.load()) { finish = "cancel"; break; }
#if defined(STRATA_ENABLE_NVFP4)
                    // The token graph and its doorbell have returned (the sample sync above). Refill
                    // before the next token launches another replay.
                    if (o.hot_learn && xcache.planar() && !drive.d.usage.empty() && o.adapt_every > 0 &&
                        (rounds % o.adapt_every) == 0 && !adapt()) {
                        std::printf("ERR an adaptive refill failed\n");
                        return 1;
                    }
#endif
                    ss.ple_prev[0] = ss.ple_prev[1];
                    ss.ple_prev[1] = x;
                    x = next;
                    ++p;
                    // The next iteration's id is known and it will call ple_finish. Issue now, after this
                    // token's sample sync, so that wait is only whatever the SSD has not already finished.
                    // Skipped when the loop is about to stop: a leftover ticket breaks the next request.
                    if (ss.ple.ahead && ss.ple.ready() && !ss.ple.inflight && produced_n < max_new &&
                        p < o.max_context) {
                        ss.ple_token = x;
                        if (!strata::core::ple_issue_token(ss.ple, err)) {
                            std::printf("ERR %s\n", err.c_str());
                            return 1;
                        }
                    }
                }
            }
            while (spec_serve && !cancelled && produced_n < max_new) {
                int T = S_mtp;
                if (!dflash.loaded() && req_spec_min_p > 0.0) {
                    T = 1;
                    while (T < S_mtp && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;
                }
                if (first_window) T = 1;
                // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
                // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
                bool from_sfx = false;
                int sfx_match = 0;
                if (!dflash.loaded() && o.suffix_draft > 0 && !first_window) {
                    const int k = sfx.propose(S - 1, sbuf.data());
                    sfx_match = sfx.last_match();
                    if (k > 0 && sbuf[0] == drafts[0]) {
                        const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                        if (pk.lookup) { T = pk.t; from_sfx = true; }
                    }
                }
                const bool timed_round = !first_window;
                const Clock::time_point round0 = Clock::now();
                if (p + T > o.max_context) break;
                window[0] = x;
                for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                apply_pending(false);
                if (hist_n > 0) {
                    // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the
                    // fed-back head `x` (it joins `consumed` only after this window commits), then the drafts
                    // before that row - what plain decode would have counted there.  (Until 0.1.19 only row 0
                    // was staged, and the drafted rows read unwritten slots.)
                    strata::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                                  hist_n, hist_stage.data());
                    const strata::core::OnDevice on_h(hist_dev);
                    cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                               cudaMemcpyHostToDevice);
                }
                tr("window", p, T);
                const Clock::time_point tw0 = Clock::now();
                if (!ver.run(T, window.data(), p, win_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed) {
                    std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                    return 1;
                }
                int a = 0;
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
                const Clock::time_point tw1 = Clock::now();
                std::thread adapt_thr;   // the adaptive tier beside the commit and the draft (as in generate)
                bool adapt_ok = true;
                const bool adapt_due = !drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0;
                // Planar refills publish gm and cold_slot_of; they wait until commit/draft are done
                // so they do not overlap a decode-graph replay or an in-flight doorbell window.
                if (adapt_due && !(o.hot_learn && xcache.planar()))
                    adapt_thr = std::thread([&] { adapt_ok = adapt(); });
                if (!ver.commit(a + 1, err)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
                for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
                draft_offered += T - 1;
                draft_accepted += a;
                first_window = false;
                bool eos = false;
                for (int i = 0; i <= a && produced_n < max_new && !eos; ++i) {
                    std::printf("T %d\n", (int) outv[(size_t) i]);
                    strata::core::progress_beat();
                    ++produced_n;
                    if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                    eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                std::fflush(stdout);
                ++rounds;
                const Clock::time_point tw2 = Clock::now();
                const bool drafted = eos || produced_n >= max_new ||
                                     (dflash.loaded()
                                          ? (dflash.ingest_window(a + 1, p, err) &&
                                             dflash.draft(outv[(size_t) a], p + a + 1, drafts.data(), err))
                                          : mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(),
                                                      (float) req_spec_min_p));
                {
                    const Clock::time_point tw3 = Clock::now();
                    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                    dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                    ++dec_windows; dec_T += T;
                }
                if (adapt_thr.joinable()) adapt_thr.join();
                if (!adapt_ok) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (adapt_due && o.hot_learn && xcache.planar() && !adapt()) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (!drafted) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (timed_round && !eos)
                    policy.observe(from_sfx, T, a, sfx_match,
                                   std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                if (eos) { finish = "stop"; break; }
                if (stop_req.load()) { finish = "cancel"; break; }
                x = outv[(size_t) a];
                p += a + 1;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            if (dec_timing && dec_windows > 0) {
                const DecSnap d1 = dec_snap();
                const double w = (double) dec_windows, L = (double) g.n_layers;
                std::fprintf(stderr, "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                     "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                     "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                     "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f\n",
                             (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                             (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                             (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                             (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                             (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L));
                const std::string pr = ver.profile_report();
                if (!pr.empty()) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());
            }
            if (!cancelled) {
                // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
                // (the checkpoints taken while reading it are still good)
                live.swap(consumed);
                live_imgs = imgs_below(req_imgs, (int64_t) live.size());
                live_ok = o.prompt_cache > 0;
            }
            static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;
            if (state_hash && live_ok && spec_serve) {
                // DEBUG: a fingerprint of every part of the session over the positions it holds ([0, L)), and
                // separately of what lies past them in the last KV page (stale cells, fine unless something reads them)
                cudaDeviceSynchronize();
                const int64_t L = (int64_t) live.size();
                const strata::kernels::QsaShapes qs = [&] {
                    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
                    s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_dim = g.idx_key_dim;
                    return s;
                }();
                auto hash_dev = [&](const void* p, size_t bytes, uint64_t h) {
                    std::vector<uint8_t> b(bytes);
                    if (bytes) cudaMemcpy(b.data(), p, bytes, cudaMemcpyDefault);   // VRAM or a streamed host copy
                    return fnv1a(b.data(), b.size(), h);
                };
                // the cells [c0, c1) of one int8 K or V pool ([page][kv_head][page_size][head_dim]), bytes per value `w`
                auto hash_cells = [&](const void* pool, int64_t per_cell, int64_t c0, int64_t c1, uint64_t h) {
                    const int64_t ps = qs.page_size;
                    for (int64_t pg = c0 / ps; pg * ps < c1; ++pg)
                        for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                            const int64_t a = std::max(c0, pg * ps) - pg * ps, e = std::min(c1, (pg + 1) * ps) - pg * ps;
                            const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps + a) * per_cell);
                            h = hash_dev((const uint8_t*) pool + off, (size_t) ((e - a) * per_cell), h);
                        }
                    return h;
                };
                const ConvStateSizes z = conv_state_sizes(g);
                uint64_t h_gdn = hash_dev(ss.gdn_state, z.gdn, 1469598103934665603ull);
                if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // per GDN layer: which one differs first
                    const size_t per = z.gdn / (size_t) g.n_gdn_layers();
                    std::string s;
                    char b[8];
                    for (int64_t i = 0; i < g.n_gdn_layers(); ++i) {
                        std::snprintf(b, sizeof(b), "%04llx ", (unsigned long long) (hash_dev((const uint8_t*) ss.gdn_state + i * per, per, 1469598103934665603ull) & 0xffff));
                        s += b;
                    }
                    std::fprintf(stderr, "strata serve: STATE_HASH_GDN %s\n", s.c_str());
                }
                uint64_t h_ple = hash_dev(ss.ple_hist, z.ple, 1469598103934665603ull);
                uint64_t h_tail = 1469598103934665603ull, h_pool = h_tail, h_kv = h_tail, h_stale = h_tail;
                const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
                // a state's K/V arrays and their bytes per (cell, head) row: the host copy when it has one
                auto kv_arrays = [&](const strata::core::QsaState& st) {
                    const bool h = st.kv_mode != 0;
                    std::vector<std::pair<const void*, int64_t>> a;
                    if (st.kv_q4) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q4 : st.k_q4, q4b}, {h ? st.host.v_q4 : st.v_q4, q4b}};
                    } else if (st.kv_fp8) {
                        a = {{h ? (const void*) st.host.k_fp8 : (const void*) st.k_fp8, kvb},
                             {h ? (const void*) st.host.v_fp8 : (const void*) st.v_fp8, kvb},
                             {h ? (const void*) st.host.k_fp8_scale : (const void*) st.k_fp8_scale, scb},
                             {h ? (const void*) st.host.v_fp8_scale : (const void*) st.v_fp8_scale, scb}};
                    } else if (st.kv_hybrid) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q4 : st.v_q4, q4b},
                             {h ? st.host.k_scale : st.k_scale, scb}};
                    } else {
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q : st.v_q, kvb},
                             {h ? st.host.k_scale : st.k_scale, scb}, {h ? st.host.v_scale : st.v_scale, scb}};
                    }
                    return a;
                };
                const int64_t end_cell = std::min<int64_t>(((L + qs.page_size - 1) / qs.page_size) * qs.page_size,
                                                           ss.qsa_states[0].max_cells);
                for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
                    const strata::core::QsaState& st = ss.qsa_states[i];
                    h_tail = hash_dev(st.idx_tail, z.tail, h_tail);
                    h_pool = hash_dev(st.idx_pooled, (size_t) (L / qs.idx_block) * qs.idx_dim * 4, h_pool);
                    // KV streaming: the host copy is the identity layout and holds every cell
                    for (const auto& [pool, w] : kv_arrays(st)) {
                        h_kv = hash_cells(pool, w, 0, L, h_kv);
                        h_stale = hash_cells(pool, w, L, end_cell, h_stale);
                    }
                }
                const strata::core::QsaState& ms = mtp.kv_state();
                uint64_t h_mtp = 1469598103934665603ull;
                const int64_t mL = std::min<int64_t>(L, ms.max_cells);
                for (const auto& [pool, w] : kv_arrays(ms))
                    if (pool != nullptr) h_mtp = hash_cells(pool, w, 0, mL, h_mtp);
                std::fprintf(stderr, "strata serve: STATE_HASH L=%lld gdn=%016llx ple=%016llx tail=%016llx pooled=%016llx "
                                     "kv=%016llx mtp=%016llx stale=%016llx ple_prev=%d,%d\n", (long long) L,
                             (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                             (unsigned long long) h_pool, (unsigned long long) h_kv, (unsigned long long) h_mtp,
                             (unsigned long long) h_stale, ss.ple_prev[0], ss.ple_prev[1]);
            }
            const int64_t req_hits = drive.d.cache_hits - decode_hits0;
            const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld\n", (long long) produced_n, (long long) n, prompt_ms,
                        decode_ms, finish, (long long) draft_accepted, (long long) draft_offered, (long long) resume,
                        (long long) req_hits, (long long) req_look);
            std::fflush(stdout);
            hot_learn_clear_pin();
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
            const int64_t fresh = n - resume;
            std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), "
                                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                         (long long) n, (long long) resume, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * fresh / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                         (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
            // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU
            // to read (--pcie-frac) are in neither count
            if (req_look > 0) {
                std::fprintf(stderr, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)\n",
                             100.0 * (double) req_hits / (double) req_look,
                             (long long) req_hits, (long long) req_look);
            }
            // STRATA_SPLIT_TIMING: where each verify stage's host time went, cumulative per window since the start
            // (waiting for its GPU to ring a layer, the CPU pool and plan per layer, staging the window)
            if (static const bool st_timing = std::getenv("STRATA_SPLIT_TIMING") != nullptr; st_timing && spec_serve)
                for (int st = 0; st < n_stages; ++st) {
                    const strata::core::Verifier& v = stage_ver(st);
                    const double w = v.windows > 0 ? (double) v.windows : 1.0;
                    std::fprintf(stderr, "strata serve: stage %d: %lld windows; per window: wait for the GPU %.3f ms, "
                                         "pool + plan %.3f ms, host staging %.3f ms, commit %.3f ms\n", st,
                                 (long long) v.windows, v.ms_wait / w, v.ms_pool / w, v.ms_host / w, v.ms_commit / w);
                }
            if (g.n_qsa_layers() > 0 && ss.qsa_states[0].kv_mode == 1) {
                // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
                uint64_t miss = 0, look = 0;
                bool over = false;
                for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
                    const strata::kernels::KvStreamCounters c = strata::kernels::kv_stream_counters(ss.qsa_states[i].map);
                    miss += c.misses; look += c.lookups; over = over || c.overflow;
                }
                std::fprintf(stderr, "strata serve: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                     "from RAM%s\n", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                             (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                             over ? " - OVERFLOW (too few resident cells)" : "");
            }
            if (sfx_windows > 0)
                std::fprintf(stderr, "strata serve: suffix drafts: %lld windows, %lld of %lld drafts accepted\n",
                             (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
                std::fprintf(stderr, "strata serve: CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                     "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                     "waiting for it (since start)\n", r + 1,
                             (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                             (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                             (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                             (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                             remote_experts[(size_t) r].ms_begin(), remote_experts[(size_t) r].ms_wait());
        }
        return 0;
    }

    // ---- plan v0.3 P5: the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop
    // then starts at the last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = 0;   // plan v0.3 P6: where the speculative loop starts (0 = not used)
    strata::prefill::Prefill prefill;
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    std::vector<std::pair<int32_t, int32_t>> lent;     // (residency index, slot) lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        // RDMA push channel 1: one window covers a whole chunk of one layer, capped at kMaxPushTok tokens.
        if (rdma_tier.push_armed() && o.prefill_chunk > strata::core::RDMAExpertTier::kMaxPushTok) {
            o.prefill_chunk = strata::core::RDMAExpertTier::kMaxPushTok;
            std::fprintf(stderr, "strata generate: remote prefill windows cap the chunk at %lld tokens\n",
                         (long long) strata::core::RDMAExpertTier::kMaxPushTok);
        }
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!o.no_prefill_borrow && !host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            int64_t k = plan_lend(chunk);             // auto: the largest chunk that fits; fixed: halved to fit
            const int64_t request_sized = request_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = lend_slots(chunk);
                if (k + 128 > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : request_chunk(n_batched, 1024);
                std::fprintf(stderr, "strata generate: prompt chunk auto: %lld tokens\n", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
                borrow = xcache.device_slot(first);
                borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                                     : (uint64_t) k * (uint64_t) blob;
                std::fprintf(stderr, "strata generate: prompt path borrows %lld cache slots (%.2f GiB)\n", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            std::fprintf(stderr, "strata generate: prompt path allocates its own buffers (no cache slots to borrow)\n");
        prefill.set_nvfp4_hot(nv4arena.on() ? &nv4arena : nullptr);
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, err, borrow,
                          borrow_bytes, rdma_tier.push_armed() ? &rdma_tier : nullptr)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                return mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        const Clock::time_point tp0 = Clock::now();
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            const Clock::time_point tr = Clock::now();
            for (const auto& [i, slot] : lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(refill_blocking() ? xcache.fill_slot_blocking(slot, b, err, nb)
                                                        : xcache.fill_slot_queued(slot, b, err, nb))) {
                    std::fprintf(stderr, "strata generate: refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                host_res[(size_t) i] = slot;
            }
            if (!xcache.sync_queued(err)) {
                std::fprintf(stderr, "strata generate: refilling the lent slots failed: %s\n", err.c_str());
                return 1;
            }
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            std::fprintf(stderr, "strata generate: %zu lent slots refilled in %.1f ms\n", lent.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const strata::prefill::PrefillStats& ps = prefill.stats();
        std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld; PLE %.1f ms\n",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident, ps.ms_ple);
    }

    if (o.hot_learn && drive.d.usage.empty() && o.adapt_every > 0 && o.adapt_swaps > 0)
        drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);

    for (int64_t pos = pos_start;; ++pos) {
        // plan v0.3 P6: a native pack's last prompt token is the first verify window (T = 1)
        if (native_pack) { spec_pos = pos; break; }
        if (pos >= o.max_context) {
            std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) pos);
            return 2;
        }
        // **THE TOKEN TIMER STARTS HERE, BEFORE ANY OF THE TOKEN'S WORK (A7).**  It used to start after
        // `put_input`/`embed_row`, which excluded the embedding and the PLE window advance from the reported
        // rate, and it stopped before the NaN scan, the logits dump and the sampler.  The published tok/s
        // figure is a WALL-CLOCK rate: everything one token costs, PLE advance to sampled id.  A rate that
        // excludes real per-token work is not a rate anyone can plan against.
        const Clock::time_point t0 = Clock::now();
        // **THE PLE'S TOKEN WINDOW IS HOST-SIDE, AND `ple_issue_token` RUNS OUTSIDE THE CAPTURE.**
        // `ngram_rows` is a host hash over the last three tokens and the table gather is a host read, so
        // either one inside a captured graph would run once at capture time and replay forever.  `ple_prev`
        // is OLDEST FIRST and `-1` means "no predecessor", which `ngram_rows` treats as the EOS cut.
        // With the one-beat pipeline the rows for THIS token were issued when its id became known (the
        // previous iteration, after the sample sync, or before the graph when the next prompt id was already
        // in hand). This site only issues when nothing is in flight: the first token of the loop.
        if (!ss.ple.inflight) ss.ple_token = (int32_t) tok;
        else if (ss.ple_token != (int32_t) tok) {
            std::fprintf(stderr, "strata generate: ple ahead prefetched %d but this token is %lld\n",
                         ss.ple_token, (long long) tok);
            return 1;
        }
        Clock::time_point tp = Clock::now();
        // Plan v0.3 P2: `ms_ple` is the issue (when it still happens here) plus the time still spent WAITING
        // in `ple_finish_token`. A prefetched ticket skips the issue and the wait is whatever the SSD has left.
        if (ss.ple.ready() && !ss.ple.inflight && !strata::core::ple_issue_token(ss.ple, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (!put_input(tok, pos)) return 1;
        {
            const Clock::time_point n = Clock::now();
            ms_embed += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (ss.ple.ready() && !strata::core::ple_finish_token(ss.ple, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        // The next prompt id is already known, and its rows are not read until that token's ple_finish,
        // which is after this token's graph. Issue now so the SSD read overlaps the ~20 ms of GPU work.
        // The window is restored afterwards: the loop tail still advances ple_prev exactly once. Decode
        // ids are NOT known here (they come back from the sampler) and are issued after that readback.
        if (ss.ple.ahead && ss.ple.ready() && !ss.ple.inflight &&
            pos + 1 < n_prompt && pos + 1 < o.max_context &&
            (int64_t) produced.size() < o.max_new) {
            const Clock::time_point t_issue = Clock::now();
            const int32_t keep_tok = ss.ple_token;
            const int32_t keep0 = ss.ple_prev[0];
            const int32_t keep1 = ss.ple_prev[1];
            ss.ple_prev[0] = keep1;
            ss.ple_prev[1] = (int32_t) tok;
            ss.ple_token = (int32_t) o.tokens[(size_t) (pos + 1)];
            const bool issued = strata::core::ple_issue_token(ss.ple, err);
            ss.ple_token = keep_tok;
            ss.ple_prev[0] = keep0;
            ss.ple_prev[1] = keep1;
            if (!issued) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            ms_ple += std::chrono::duration<double, std::milli>(Clock::now() - t_issue).count();
        }

        if (pos % 256 == 0 || pos + 1 >= n_prompt - 1)
            std::fprintf(stderr, "strata generate: position %lld, token %lld%s\n", (long long) pos, (long long) tok,
                         pos < n_prompt ? " (prompt)" : "");
        // **`d.layers` IS THE BLOB'S LAYER AXIS, NOT A COUNTER.**  The adapter uses it to index
        // `experts.bin` as `layer * n_expert + expert`, so it MUST restart at 0 for every token.  Leaving it
        // running across tokens asks for layer 48 of a 48-layer file on the second token - which
        // `FileExpertSource` REFUSES rather than wrapping into layer 0's experts, and that refusal is the only
        // reason this was a clean error instead of a silently wrong second token.
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        err.clear();
        if (o.no_capture) {
            if (!strata::core::session_token(wt, g, pos, /*pos_base=*/0, ss, d_parts, main_cs,
                                             o.sync_every_layer, err)) {
                std::fprintf(stderr, "strata generate: session_token: %s\n", err.c_str());
                return 1;
            }
        } else {
            strata::core::doorbell_reset(db);
            if (tgraph.captured) {
                if (!strata::core::session_run_token(g, pos, /*pos_base=*/0, ss, tgraph, pool_fn, pool_user,
                                                     loop_scratch.y_miss, main_cs, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
            } else if (!strata::core::session_loop(g, pos, /*pos_base=*/0, ss, gr, pool_fn, hit_fn, pool_user, /*overlap=*/true, main_cs,
                                        err, layer_stage, &loop_scratch)) {
                std::fprintf(stderr, "strata generate: session_loop: %s\n", err.c_str());
                return 1;
            }
        }
        if (final_r != nullptr) {
            cudaMemcpy(final_r_host.data(), ss.R, final_r_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const int64_t posrec[2] = {pos, tok};
            std::fwrite(posrec, sizeof posrec, 1, final_r);
            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);
        }
        if (drive.d.failed) {
            std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                         (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                         drive.d.fail ? drive.d.fail : "(no message)");
            return 1;
        }
        {
            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**
            const Clock::time_point n = Clock::now();
            ms_layers += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        // ---- the ladder for THIS position, in the order `session_loop` filled it: layer 0 first.
        if (layer_dump != nullptr) std::fwrite(layer_stage, sizeof(float), layer_floats, layer_dump);
        if (half_dump != nullptr) {
            std::fwrite(half_stage, sizeof(float), (size_t) g.n_layers * (size_t) half_stride, half_dump);
        }
        if (!run_head(token_stream)) {
            std::fprintf(stderr, "strata generate: lm_head: %s\n", err.c_str());
            return 1;
        }
        // **A CHECKPOINT AFTER THE HEAD, BECAUSE AN ASYNC FAULT IS STICKY AND LIES ABOUT WHERE IT HAPPENED.**
        // Measured, and it cost an hour: without this, `embed_row`'s D2H on the NEXT token reported "an illegal
        // memory access" at a plane offset that has nothing to do with the fault, and the layer that actually
        // faulted had completed its own error checks successfully - because its kernels had not run yet.  A
        // sticky error surfaces at the next SYNCHRONISING call, which is whatever happens to come next.
        if (!o.stream_token && cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the device faulted in lm_head at position %lld: %s\n",
                         (long long) pos, cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_head += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        const bool emit_logits = dump != nullptr &&
            strata::program::logits_selection::selected(pos, dump_positions, o.logits_stride);
        const bool read_logits = !o.stream_token || o.check_logits || emit_logits;
        if (read_logits && (cudaMemcpyAsync(logits.data(), d_logits, (size_t) n_vocab * 4,
                                           cudaMemcpyDeviceToHost, (cudaStream_t) token_stream) != cudaSuccess ||
                            cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess)) {
            std::fprintf(stderr, "strata generate: reading the logits back failed\n");
            return 1;
        }
        int bad = 0;
        if (read_logits) for (float v : logits) if (!std::isfinite(v)) ++bad;
        if (bad != 0) {
            std::fprintf(stderr, "strata generate: %d of %lld logits are not finite at position %lld\n", bad,
                         (long long) n_vocab, (long long) pos);
            return 1;
        }
        if (emit_logits && std::fwrite(logits.data(), sizeof(float), (size_t) n_vocab, dump) != (size_t) n_vocab) {
            std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) pos);
            std::fclose(dump);
            return 1;
        }
        {
            // **993 KB OF SYNCHRONOUS D2H AND A 248,320-FLOAT HOST SCAN, EVERY TOKEN.**  (The review's notes
            // say 151,936 floats; the artifact's `output.weight` is 248,320 rows, so the real figure is 1.6x
            // that - a number nobody had checked because nothing measured this term.)  R2.6 asks for the dump
            // and the scan to be behind flags; round 36 did exactly that and measured it SLOWER, because on
            // this driver a large blocking readback is also what flushes the pipeline for the sampler that
            // follows.  Timed so the claim can be re-checked rather than remembered.
            const Clock::time_point n = Clock::now();
            ms_readback += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        int next = 0;
        // The draw is Philox(seed, position), as in a verify window (row t at pos0 draws pos0 + t): a seed gives
        // the same text whether a token comes from this path or from the speculative loop below.
        sp.counter = (uint64_t) pos;
        strata::kernels::sample_tokens(d_logits, 1, (int) n_vocab, nullptr, 0, sp, d_next, token_stream);
        if (cudaMemcpyAsync(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost,
                            (cudaStream_t) token_stream) != cudaSuccess ||
            cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading the sampled token back failed: %s\n",
                         cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        // The sampled-token synchronization also completes every captured QSA
        // status readback. Retain one status per layer so a later layer cannot
        // hide an earlier failure; no extra synchronization or token allocation.
        if (o.native_flash_attn_short) for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
            const int32_t status = ss.qsa_states[i].host_step[strata::kernels::kStepCount];
            if (status != 0) {
                std::fprintf(stderr, "strata generate: native attention status %d at QSA layer %lld, position %lld\n",
                             status, (long long) i, (long long) pos);
                return 1;
            }
        }
        if (next < 0 || next >= n_vocab) {
            std::fprintf(stderr, "strata generate: the sampler returned %d, outside 0..%lld\n", next,
                         (long long) (n_vocab - 1));
            return 1;
        }
        {
            // **TWO DEVICE-WIDE SYNCS FOR FOUR BYTES.**  `sample_tokens(nullptr)` ends in
            // `cudaDeviceSynchronize()` (`sampler.cu:245`) and the blocking 4-byte read below is the second.
            const Clock::time_point n = Clock::now();
            ms_sample += std::chrono::duration<double, std::milli>(n - tp).count();
            ++phase_tokens;
        }
        // CHARGED HERE, AFTER THE SAMPLER, so the wall-clock rate covers the whole token including the embedding,
        // the NaN scan, the logits readback and the sample (A7).  Only DECODE positions count; prefill is
        // measured separately.
        if (pos >= n_prompt - 1) total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        else prefill_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (pos == n_prompt - 1) ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        // **THE PREDICTION AT THE LAST PROMPT POSITION *IS* THE FIRST GENERATED TOKEN.**  Sampling on every
        // position and recording only from `n_prompt - 1` onward is what keeps the two cases from needing
        // separate handling - and the version that "obviously" only samples after the prompt loses exactly one
        // token's worth of conditioning.
        if (pos >= n_prompt - 1) produced.push_back(next);
        if ((int64_t) produced.size() >= o.max_new) break;
        if (o.stop_eos && pos >= n_prompt - 1 &&
            std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end()) break;
#if defined(STRATA_ENABLE_NVFP4)
        // Sample sync has completed this token's graph. Refill before the next replay.
        if (o.hot_learn && xcache.planar() && !drive.d.usage.empty() && o.adapt_every > 0 &&
            o.adapt_swaps > 0 && pos >= n_prompt - 1 &&
            ((int64_t) produced.size() % o.adapt_every) == 0) {
            std::string hot_err;
            int64_t nswap = 0;
            if (!hot_learn_planar_adapt(xcache, drive.d.usage, host_res, d_res, nv4_gm13, nv4_gm2,
                                        const_cast<uint16_t*>(nv4arena.gm13),
                                        const_cast<uint16_t*>(nv4arena.gm2), cold_slot_of, srcp, g.n_layers,
                                        g.n_expert, o.adapt_swaps, (cudaStream_t) token_stream, nswap, hot_err)) {
                std::fprintf(stderr, "strata generate: %s\n", hot_err.c_str());
                return 1;
            }
        }
#endif
        // TEACHER FORCING while the prompt lasts: the next input is the prompt's own next token, not the
        // model's guess.  Feeding the guess would make the run depend on the model's own errors from position
        // 1, which is a different (and worse) measurement of the same prompt.
        // and the window advances: the token just decoded becomes the newest predecessor.
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = (int32_t) tok;
        tok = (pos + 1 < n_prompt) ? o.tokens[(size_t) (pos + 1)] : next;
        // Plan v0.3 P6: from the first generated token on, the speculative loop below takes over.
        // Do not leave a PLE ticket in flight across that handoff: the verifier's gather_batch refuses
        // a table that still has a single-token issue pending.
        if (o.spec > 0 && pos >= n_prompt - 1) { spec_pos = pos + 1; break; }
        if (pos + 1 < o.max_context && ss.ple.ready() && ss.ple.ahead) {
            if (ss.ple.inflight) {
                // Prompt path already issued this id before the graph. Record it for the next iteration's check.
                ss.ple_token = (int32_t) tok;
            } else {
                const Clock::time_point t_issue = Clock::now();
                ss.ple_token = (int32_t) tok;
                if (!strata::core::ple_issue_token(ss.ple, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                ms_ple += std::chrono::duration<double, std::milli>(Clock::now() - t_issue).count();
            }
        }
    }

    // ================================ plan v0.3 P6: SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t
    // is exactly what greedy decode would emit there, so the first draft that differs ends the round and the
    // round emits (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos > 0 && (int64_t) produced.size() < o.max_new && !ended) {
        std::vector<int64_t> oracle;
        if (!o.spec_oracle.empty()) {
            std::ifstream in(o.spec_oracle);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            if (!in || !parse_i64_list(text.c_str(), oracle, e)) {
                std::fprintf(stderr, "strata generate: cannot read --spec-oracle %s\n", o.spec_oracle.c_str());
                return 2;
            }
        }
        if (thits.d_res == nullptr) {
            std::fprintf(stderr, "strata generate: --spec needs the device residency table (--expert-profile, "
                                 "--expert-cache and the token graph)\n");
            return 2;
        }
        mem_mark("the head and the prompt path");
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (dflash.loaded()) ver.set_dflash_taps(dflash.tap_buffer(), dflash.tap_stride());
        // RDMA push channel 2: the verify window's misses stop driving the wire from this host too - the
        // worker one-sided-reads the window's activation segment and RDMA-writes the rows straight into the
        // window's y_miss segment, then raises the window's own flag.  Without it a verify window's
        // begin/finish is a verbs round trip per layer, which is what made --spec collapse on this host.
        if (rdma_tier.push_armed()) {
            int64_t vx_bytes = 0, vy_bytes = 0;
            const float* vx = ver.push_x(vx_bytes);
            float* vy = ver.push_y(vy_bytes);
            if (!rdma_tier.arm_push_verify(vx, (size_t) vx_bytes, vy, (size_t) vy_bytes, ver.push_flag(), err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: RDMA push channel 2 armed: verify windows are push windows "
                                 "(%lld tokens, %lld rows)\n", (long long) (vx_bytes / ((int64_t) g.n_embd * 4)),
                         (long long) (vy_bytes / ((int64_t) g.n_embd * 4)));
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the verifier and the drafter's binding");
        ver.set_sampling(sp);   // the CLI's own sampling (until 0.1.19 this loop was always greedy); no penalties here
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        // STRATA_VERIFY_NO_GPU_PLAN=1 (diagnostic): the verify window plans NO VRAM/PCIe share, so every
        // routed row falls to the remote tier - isolates a wrong verify forward to the window's GPU hit
        // path (plan sink / grouped kernel) vs the window's own graph (activations, attention, combine).
        if (std::getenv("STRATA_VERIFY_NO_GPU_PLAN") == nullptr) drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        const int64_t pcie0 = drive.d.pcie_experts;
        // --hot-learn may already be counting from the single-token prompt; do not wipe that.
        if (o.adapt_every > 0 && o.adapt_swaps > 0 && !(o.hot_learn && !drive.d.usage.empty()))
            drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        cudaStream_t adapt_stream = nullptr;
        if (!drive.d.usage.empty() && cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
            for (const auto& [i, slot] : pending) host_res[(size_t) i] = slot;
            pending.clear();
            if (d_res != nullptr)
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        };
        // Plan v0.3 P6: the VRAM tier follows the conversation.  Candidates are missing experts routed at least
        // twice (decayed); each is paired with its layer's least-routed resident expert and swapped when it was
        // routed clearly more often.  Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
#if defined(STRATA_ENABLE_NVFP4)
            if (o.hot_learn && xcache.planar()) {
                std::string e;
                int64_t nswap = 0;
                if (!hot_learn_planar_adapt(xcache, drive.d.usage, host_res, d_res, nv4_gm13, nv4_gm2,
                                            const_cast<uint16_t*>(nv4arena.gm13),
                                            const_cast<uint16_t*>(nv4arena.gm2), cold_slot_of, srcp,
                                            g.n_layers, g.n_expert, o.adapt_swaps, adapt_stream, nswap, e)) {
                    std::fprintf(stderr, "strata generate: %s\n", e.c_str());
                    return false;
                }
                swaps_total += nswap;
                ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                return true;
            }
#endif
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                    else if (!hot_pinned(l, e)) vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                // asynchronous: the copies run while the MTP drafts; the next window waits for them
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(xcache.device_slot(slot), b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            for (float& v : drive.d.usage) v *= 0.7f;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // plan v0.3 P6: with a native pack the first window is the last prompt token alone (it produces the first
        // generated token and the MTP's first cell); otherwise the token loop already did that.
        bool first_window = native_pack || dflash.loaded();
        if (use_mtp && !first_window &&
            !mtp.draft_first(o.spec, ss.R, x, p - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(o.spec);   // MTP or lookup window (see draft_policy.hpp)
        std::vector<int32_t> sbuf((size_t) o.spec, 0);
        int64_t sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        if (o.suffix_draft > 0) {
            for (int64_t t : o.tokens) sfx.append((int32_t) t);
            for (int64_t t : produced) sfx.append((int32_t) t);
        }
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        while ((int64_t) produced.size() < o.max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = S_mtp;
            if (!dflash.loaded() && use_mtp && o.spec_min_p > 0.0) {
                T = 1;
                while (T < S_mtp && dprob[(size_t) T - 1] >= (float) o.spec_min_p) ++T;
            }
            if (first_window) T = 1;
            bool from_sfx = false;
            int sfx_match = 0;
            if (!dflash.loaded() && o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(o.spec - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = from_sfx ? sbuf[(size_t) i - 1]
                                     : (use_mtp || dflash.loaded()) ? drafts[(size_t) i - 1]
                                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            apply_pending(false);
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (drive.d.failed) {
                std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            // STRATA_VERIFY_TRACE=1 (diagnostic): window vs outv, one line per window - pins an off-by-one
            // in the window's forward (outv[t] must equal the greedy pick after window[t]).
            if (std::getenv("STRATA_VERIFY_TRACE") != nullptr) {
                std::fprintf(stderr, "vtrace pos=%lld T=%d a=%d win:", (long long) p, T, a);
                for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", window[(size_t) i]);
                std::fprintf(stderr, " outv:");
                for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", outv[(size_t) i]);
                std::fprintf(stderr, "\n");
            }
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            }
            // plan v0.3 P6: the adaptive tier's host work (ranking, copy submission) runs on its own thread while the
            // GPU commits and drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            const bool adapt_due = !drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0;
            if (adapt_due && !(o.hot_learn && xcache.planar()))
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < o.max_new && !eos; ++i) {
                produced.push_back(outv[(size_t) i]);
                if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            const bool drafted = dflash.loaded()
                                     ? ((int64_t) produced.size() >= o.max_new ||
                                        (dflash.ingest_window(a + 1, p, err) &&
                                         dflash.draft(outv[(size_t) a], p + a + 1, drafts.data(), err)))
                                     : (!use_mtp || (int64_t) produced.size() >= o.max_new ||
                                        mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(),
                                                  (float) o.spec_min_p));
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (adapt_due && o.hot_learn && xcache.planar() && !adapt()) return 1;
            if (!drafted) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (timed_round) policy.observe(from_sfx, T, a, sfx_match, round_ms);
            if (rounds % 64 == 0)
                std::fprintf(stderr, "strata generate: position %lld, %lld tokens, %lld rounds\n", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        std::printf("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        if (o.spec_min_p > 0.0) {
            std::printf("%-24s", "window sizes");
            for (size_t i = 1; i < window_hist.size(); ++i) std::printf(" T%zu:%lld", i, (long long) window_hist[i]);
            std::printf("  (min draft probability %.2f)\n", o.spec_min_p);
        }
        if (o.suffix_draft > 0)
            std::printf("%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts", (long long) sfx_windows,
                        (long long) sfx_ok, (long long) sfx_drafts);
        std::printf("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) accepted_hist[i]);
        std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0)
            std::printf("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round\n", "pool multi", pool.ms_multi_gu / rounds,
                        pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds);
        if (rounds > 0)
            std::printf("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier (every %d rounds, %.3f ms/round)\n", "adaptive tier",
                        (long long) swaps_total, o.adapt_every, ms_adapt / rounds);
        if (rounds > 0 && drive.d.pcie_num > 0)
            std::printf("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        (void) pool_ms0;
        if (use_mtp && rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, mtp.ms_prefill,
                        (double) mtp.vram_bytes() / 1048576.0);
    }

    if (dump != nullptr && std::fclose(dump) != 0) {
        std::fprintf(stderr, "strata generate: cannot finish logits dump\n");
        return 1;
    }
    if (layer_dump != nullptr) {
        std::fclose(layer_dump);
        cudaFreeHost(layer_stage);
        std::printf("%-24s %s (%lld layers + the input x %d streams x %lld per position)\n", "layers dumped",
                    o.dump_layers.c_str(), (long long) g.n_layers, (int) g.hc, (long long) g.n_embd);
    }
    if (half_dump != nullptr) {
        std::fclose(half_dump);
        cudaFreeHost(half_stage);
        std::printf("%-24s %s (%lld layers x %llu per position)\n", "halves dumped", o.dump_halves.c_str(),
                    (long long) g.n_layers, (unsigned long long) half_stride);
    }
    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        std::printf("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.calls);
    }
    if (o.stage_timing) strata::core::stage_timing_report(g.n_layers);

    const int64_t decoded = (int64_t) produced.size();
    std::printf("prompt  :");
    for (int64_t t : o.tokens) std::printf(" %lld", (long long) t);
    std::printf("\noutput  :");
    for (int64_t t : produced) std::printf(" %lld", (long long) t);
    std::printf("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);
    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (cudaMemcpy(mx.data(), ss.block.mixed, mx.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading mixed back failed\n");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (cudaMemcpy(R.data(), ss.R, R.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading R back failed\n");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    if (o.stats) {
        std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms);
        // **THE PER-TOKEN HOST TERM, WHICH `--gpu-only-full` CANNOT SEE.**  That measurement never enters the
        // token loop, so it excludes all six of these.  On the 78-token fixture + 200 generated tokens the six
        // sum to ~9 ms of non-layer work against ~1.5 ms of actual head GPU work - 16% of the token, and it is
        // not the pool.
        if (phase_tokens > 0) {
            const double pt = (double) phase_tokens;
            std::printf("%-24s PLE %.3f  embed %.3f  LAYERS %.3f  head %.3f  readback %.3f  sample %.3f  "
                        "(sum %.3f of %.3f ms)\n",
                        "  token host phases", ms_ple / pt, ms_embed / pt, ms_layers / pt, ms_head / pt,
                        ms_readback / pt, ms_sample / pt,
                        (ms_ple + ms_embed + ms_layers + ms_head + ms_readback + ms_sample) / pt, decode_ms);
        }
        if (const std::string io = ple_table.io_report(); !io.empty()) std::printf("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // **AND WHERE INSIDE `run()` IT WENT.**  Three phases per layer and they were one number, which cannot
        // tell a pool that is slow at the WORK from one that is slow at the SYNCHRONISATION - opposite fixes.
        // Wait-for-park is expected to be ~0 (the workers re-parked at the end of the previous layer); the
        // question is whether the time is in the drain or in the re-park barrier.
        if (pool_positions > 0.0) {
            double wp = 0, dr = 0, rp = 0;
            pool.phase_ms(wp, dr, rp);
            const double per = pool_positions;
            std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
                        "  pool phases", wp / per, dr / per, rp / per);
        }
        std::printf("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            std::printf("  CUDA%d experts           %lld routed entries computed\n",
                        r + 1, (long long) remote_experts[(size_t) r].computed());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t hl = drive.d.hit_ready + drive.d.hit_late;
            std::printf("%-24s %lld of %lld layers the hit work was DONE when the pool returned\n",
                        "  R4 overlap", (long long) drive.d.hit_ready, (long long) hl);
            std::printf("%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n",
                        "  R4 expert-cache hits", (long long) drive.d.cache_hits, (long long) look,
                        look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0,
                        (long long) drive.d.cache_admitted, (long long) drive.d.cache_refused,
                        100.0 * (double) (drive.d.cache_admitted + drive.d.cache_hits > 0
                                              ? (double) xcache.resident() / (double) xcache.slots()
                                              : 0.0));
        }
        if (tgraph.captured && tgraph.calls > 0) {
            const double per = (double) tgraph.calls;
            std::printf("%-24s wait for rings %.3f  pool %.3f ms/token  (%lld flushes over %lld positions)\n",
                        "  token graph", tgraph.ms_wait / per, tgraph.ms_pool / per, (long long) tgraph.flushes,
                        (long long) tgraph.calls);
        }
        if (gr.captured && gr.calls_total > 0) {
            // The counters are CUMULATIVE over every `session_loop` call, and PREFILL runs the loop too - so
            // the denominator is the number of positions, not the number of generated tokens.  Dividing by
            // `n_layers * decoded` printed "240 of 192", which is a reporting bug that looks like a ring
            // firing more often than it should.
            const int64_t positions = gr.calls_total;
            std::printf("%-24s %lld of %lld over %lld positions\n", "  rings seen MID-GRAPH",
                        (long long) gr.rings_mid_graph, (long long) (g.n_layers * positions),
                        (long long) positions);
            std::printf("%-24s %.3f ms of a %.3f ms layer\n", "  ring latency",
                        gr.ms_to_ring / (double) (g.n_layers * positions), decode_ms / (double) g.n_layers);
            // **THE ROUND TRIP, SPLIT AT THE RING.**  `ring latency` is the first half and stops when the ring
            // is seen; this is the second half - the driver calls after it, during which the GPU is IDLE
            // because `post[l]` has not been launched yet.  `--no-pool` is the arm that isolates it: 38.73
            // ms/token against a 26.32 ms pure-GPU floor is 12.4 ms of round trip with no expert work at all.
            //
            // Same denominator as the pool line above (the positions the loop actually ran on), so the two can
            // be added without one of them being inflated by prefill.
            const double perlap = (double) (g.n_layers * positions);
            std::printf("%-24s %.3f ms/token over %.0f positions (%.3f ms/layer, after the ring)\n",
                        "  host after ring", pool_positions > 0.0 ? gr.ms_host / pool_positions : 0.0,
                        pool_positions, gr.ms_host / perlap);
        }
    }

    if (dump != nullptr) std::printf("%-24s %s\n", "logits dumped", o.dump_logits.c_str());

    strata::core::session_graphs_free(gr);
    strata::core::doorbell_free(db);
    cudaFree(d_next);
    cudaFree(d_logits);
    cudaFree(d_emb);
    cudaFree(d_parts);
    cudaFree(sbuf);
    cudaFree(arena);
    return 0;
}
