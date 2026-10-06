// src/core/rdma_expert_tier.cpp - the sender half of the RDMA cold-expert tier; see the header for the
// contract and the wire format.  One window in flight at a time, one chained write per layer, one-sided
// polling on the way back.

#include "strata/core/rdma_expert_tier.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_TIER_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define STRATA_TIER_SPIN_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define STRATA_TIER_SPIN_PAUSE() ((void) 0)
#endif

namespace strata::core {

static_assert(sizeof(RDMAExpertTier::PushReqHead) == 36, "the polled request head is 36 bytes (one line with alignas(64))");
static_assert(offsetof(RDMAExpertTier::PushReq, channel) == offsetof(RDMAExpertTier::PushReqHead, channel) &&
              offsetof(RDMAExpertTier::PushReq, x_off) == offsetof(RDMAExpertTier::PushReqHead, x_off) &&
              offsetof(RDMAExpertTier::PushReq, y_off) == offsetof(RDMAExpertTier::PushReqHead, y_off),
              "PushReqHead must be a prefix of PushReq");
static_assert(sizeof(RDMAExpertTier::PushReqHead) <= sizeof(RDMAExpertTier::PushReq), "head beyond the block");

bool RDMAExpertTier::open(const RdmaExpertTierConfig& config, int64_t n_layers, int64_t n_expert,
                          std::string& err) {
    close();
    if (config.peer_host.empty() || config.slots == 0 || n_layers <= 0 || n_expert <= 0) {
        err = "rdma expert tier: peer host, ring slots and the model geometry are required";
        return false;
    }
    // The slot has to hold the largest request AND the largest response, both bounded by the dispatch's
    // own caps (n_tok * k <= 128, the kind[] array).  A mismatch here is a configuration error.
    const size_t max_request = sizeof(ReqHeader) + (size_t) kMaxTokens * (size_t) kHidden * sizeof(float) +
                               (size_t) kMaxEntries * 2 * sizeof(int32_t);
    const size_t max_response = 8 + (size_t) kMaxEntries * (size_t) kHidden * sizeof(float);
    const size_t need = max_request > max_response ? max_request : max_response;
    if (config.slot_bytes < need) {
        err = "rdma expert tier: slot_bytes " + std::to_string(config.slot_bytes) + " is below the " +
              std::to_string(need) + " the protocol needs (request " + std::to_string(max_request) +
              ", response " + std::to_string(max_response) + ")";
        return false;
    }
    RemoteStageConfig sc;
    sc.role = RemoteStageRole::Sender;
    sc.peer_host = config.peer_host;
    sc.port = config.port;
    sc.device = config.device;
    sc.gid_index = config.gid_index;
    sc.timeout_ms = config.timeout_ms;
    sc.slots = config.slots;
    sc.slot_bytes = config.slot_bytes;
    sc.require_rdma = true;
    sc.max_payload = config.slot_bytes;
    stage_ = std::make_unique<RemoteStage>(sc);
    if (!stage_->open(err)) {
        err = "rdma expert tier: " + err + " (start cold_expert_worker on " + config.peer_host + " first)";
        stage_.reset();
        return false;
    }
    if (!stage_->is_rdma()) {   // fail-closed: never a socket data path
        err = "rdma expert tier: the data plane is not RDMA";
        stage_.reset();
        return false;
    }
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    wait_ms_ = config.wait_ms;
    slots_ = config.slots;
    slot_bytes_ = config.slot_bytes;
    expert_format_ = config.expert_format;
    push_cfg_ = config.push;
    if (push_cfg_ && req_block_ == nullptr) req_block_ = new PushReq{};
    seq_ = 0;
    pending_ = false;
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = 0;
    ms_begin_ = ms_wait_ = 0;
    open_ = true;
    return true;
}

void RDMAExpertTier::close() {
    if (stage_) stage_->close();
    stage_.reset();
    open_ = false;
    pending_ = false;
    pending_v_ = false;
    push_armed_ = false;
    push_pf_armed_ = false;
    push_vf_armed_ = false;
    xv_base_ = nullptr;
    xv_tcap_ = 0;
    v_flag_ = nullptr;
    h_flag_ = nullptr;
    push_cfg_ = false;
    push_seq_ = 0;
    delete req_block_;
    req_block_ = nullptr;
}

bool RDMAExpertTier::arm_push(const void* x_f, size_t x_bytes, void* y_miss, size_t y_bytes, void* h_flag,
                              void* y_contig, size_t yc_bytes, std::string& err) {
    if (!open_ || !push_cfg_ || req_block_ == nullptr) {
        err = "rdma expert tier: push mode was not requested";
        return false;
    }
    if (push_armed_) return true;
    if (!x_f || !y_miss || !h_flag || x_bytes == 0 || y_bytes < (size_t) kHidden * sizeof(float)) {
        err = "rdma expert tier: push channel buffers are missing";
        return false;
    }
    const size_t yc_rows = yc_bytes > kPushYCDataOff
                               ? (yc_bytes - kPushYCDataOff) / ((size_t) kHidden * sizeof(float)) : 0;
    if (!y_contig || yc_rows == 0) {
        err = "rdma expert tier: push channel contiguous result block is missing";
        return false;
    }
    uint32_t req_rkey = 0, x_rkey = 0, y_rkey = 0, flag_rkey = 0, yc_rkey = 0;
    if (!stage_->expose_region(req_block_, sizeof(PushReq), true, false, req_rkey, err) ||
        !stage_->expose_region(x_f, x_bytes, true, false, x_rkey, err) ||
        !stage_->expose_region(y_miss, y_bytes, false, true, y_rkey, err) ||
        !stage_->expose_region(h_flag, sizeof(uint32_t), false, true, flag_rkey, err) ||
        !stage_->expose_region(y_contig, yc_bytes, false, true, yc_rkey, err)) {
        err = "rdma expert tier: push channel registration: " + err;
        return false;
    }
    PushBind b{};
    b.magic = kMagicBind;
    b.version = 3;
    b.expert_format = expert_format_;
    b.req_addr = reinterpret_cast<uint64_t>(req_block_);
    b.req_rkey = req_rkey;
    b.n_embd = (uint32_t) kHidden;
    b.x_addr = reinterpret_cast<uint64_t>(x_f);
    b.x_rkey = x_rkey;
    b.k_cap = (uint32_t) (y_bytes / ((size_t) kHidden * sizeof(float)));
    b.y_addr = reinterpret_cast<uint64_t>(y_miss);
    b.y_rkey = y_rkey;
    b.n_layers = (uint32_t) n_layers_;
    b.flag_addr = reinterpret_cast<uint64_t>(h_flag);
    b.flag_rkey = flag_rkey;
    b.yc_addr = reinterpret_cast<uint64_t>(y_contig);
    b.yc_rkey = yc_rkey;
    b.yc_rows = (uint32_t) yc_rows;
    if (!stage_->rdma_write_doorbell(0, &b, sizeof b, 1, err)) {
        err = "rdma expert tier: push bind: " + err;
        return false;
    }
    // The bind consumed doorbell sequence 1 on the worker; a later legacy window (a verify window's
    // begin/finish) must keep the worker's strictly-increasing sequence.
    seq_ = 1;
    h_flag_ = reinterpret_cast<volatile uint32_t*>(h_flag);
    push_armed_ = true;
    return true;
}

bool RDMAExpertTier::arm_push_ch3(const void* x_f, size_t x_bytes, void* y_contig, size_t yc_bytes,
                                  void* h_flag, std::string& err) {
    if (!push_armed_ || req_block_ == nullptr) {
        err = "rdma expert tier: push channel 3 needs arm_push first";
        return false;
    }
    if (push_ch3_armed_) return true;
    const size_t yc_rows =
        yc_bytes > kPushYCDataOff ? (yc_bytes - kPushYCDataOff) / ((size_t) kHidden * sizeof(float)) : 0;
    if (!x_f || !y_contig || !h_flag || x_bytes < (size_t) kHidden * sizeof(float) || yc_rows == 0) {
        err = "rdma expert tier: push channel 3 buffers are missing";
        return false;
    }
    uint32_t x_rkey = 0, flag_rkey = 0, yc_rkey = 0;
    if (!stage_->expose_region(x_f, x_bytes, true, false, x_rkey, err) ||
        !stage_->expose_region(h_flag, sizeof(uint32_t), false, true, flag_rkey, err) ||
        !stage_->expose_region(y_contig, yc_bytes, false, true, yc_rkey, err)) {
        err = "rdma expert tier: push channel 3 registration: " + err;
        return false;
    }
    PushBind4 b{};
    b.magic = kMagicBind4;
    b.version = 1;
    b.x_addr = reinterpret_cast<uint64_t>(x_f);
    b.x_rkey = x_rkey;
    b.k_cap = (uint32_t) yc_rows;
    b.yc_addr = reinterpret_cast<uint64_t>(y_contig);
    b.yc_rkey = yc_rkey;
    b.yc_rows = (uint32_t) yc_rows;
    b.flag_addr = reinterpret_cast<uint64_t>(h_flag);
    b.flag_rkey = flag_rkey;
    b.n_layers = (uint32_t) n_layers_;
    b.expert_format = expert_format_;
    // The next doorbell in the setup sequence: whichever binds ran before (arm_push, and the prefill/verify
    // arms when those paths exist) consumed 1..seq_, so channel 3's bind is seq_ + 1.  The worker only needs
    // the sequence strictly increasing.
    if (!stage_->rdma_write_doorbell(0, &b, sizeof b, seq_ + 1, err)) {
        err = "rdma expert tier: push channel 3 bind: " + err;
        return false;
    }
    // The inbox is ONE mailbox: a second bind that lands before the worker's next poll overwrites the
    // first, and the worker never sees it (the bind is then missing for the whole session).  The older
    // binds were spaced by seconds of setup work between them; give this one an explicit beat instead.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ++seq_;
    push_ch3_armed_ = true;
    return true;
}

bool RDMAExpertTier::arm_push_prefill(const void* x2, size_t x2_bytes, void* y2, size_t y2_bytes,
                                      uint32_t x_dtype, uint32_t y_dtype, std::string& err) {
    if (!push_armed_ || h_flag_ == nullptr) {
        err = "rdma expert tier: push channel 1 needs arm_push first";
        return false;
    }
    if (push_pf_armed_) return true;
    const size_t x_row = x_dtype == kPushXq8 ? (size_t) (kHidden / 32) * (36 + sizeof(float))
                         : x_dtype == kPushXf32 ? (size_t) kHidden * sizeof(float) : 0;
    const size_t y_row = y_dtype == kPushYf16 ? (size_t) kHidden * sizeof(uint16_t)
                         : y_dtype == kPushYf32 ? (size_t) kHidden * sizeof(float) : 0;
    if (!x2 || !y2 || x_row == 0 || y_row == 0 || x2_bytes == 0 || y2_bytes == 0 ||
        x2_bytes % x_row != 0 || y2_bytes % y_row != 0 ||
        x2_bytes > (size_t) kMaxPushTok * x_row ||
        y2_bytes > (size_t) kMaxPushEntries * y_row) {
        err = "rdma expert tier: push channel 1 buffers are missing, mistyped or beyond the protocol caps";
        return false;
    }
    uint32_t x2_rkey = 0, y2_rkey = 0;
    if (!stage_->expose_region(const_cast<void*>(x2), x2_bytes, true, false, x2_rkey, err) ||
        !stage_->expose_region(y2, y2_bytes, false, true, y2_rkey, err)) {
        err = "rdma expert tier: push channel 1 registration: " + err;
        return false;
    }
    PushBind2 b{};
    b.magic = kMagicBind2;
    b.version = 2;
    b.x2_addr = reinterpret_cast<uint64_t>(x2);
    b.x2_rkey = x2_rkey;
    b.t_cap = (uint32_t) (x2_bytes / x_row);
    b.y2_addr = reinterpret_cast<uint64_t>(y2);
    b.y2_rkey = y2_rkey;
    b.y2_rows = (uint32_t) (y2_bytes / y_row);
    b.x2_dtype = x_dtype;
    b.y2_dtype = y_dtype;
    b.x2_row_bytes = (uint32_t) x_row;
    b.y2_row_bytes = (uint32_t) y_row;
    // The next doorbell in the setup sequence (arm_push took 1, channel 3's bind may already have taken 2):
    // the worker only needs the sequence strictly increasing.
    if (!stage_->rdma_write_doorbell(0, &b, sizeof b, seq_ + 1, err)) {
        err = "rdma expert tier: push channel 1 bind: " + err;
        return false;
    }
    ++seq_;
    push_pf_armed_ = true;
    return true;
}

bool RDMAExpertTier::arm_push_verify(const void* x3, size_t x3_bytes, void* y3, size_t y3_bytes, void* v_flag,
                                     std::string& err) {
    if (!push_armed_ || h_flag_ == nullptr) {
        err = "rdma expert tier: push channel 2 needs arm_push first";
        return false;
    }
    if (push_vf_armed_) return true;
    const size_t row = (size_t) kHidden * sizeof(float);   // channel 2 is all-f32
    if (!x3 || !y3 || !v_flag || x3_bytes == 0 || y3_bytes == 0 || x3_bytes % row != 0 ||
        y3_bytes % row != 0 || x3_bytes > (size_t) kMaxTokens * row ||
        y3_bytes > (size_t) kMaxEntries * row || y3_bytes < x3_bytes) {   // at least one row per token
        err = "rdma expert tier: push channel 2 buffers are missing or beyond the protocol caps";
        return false;
    }
    uint32_t x3_rkey = 0, y3_rkey = 0, flag3_rkey = 0;
    if (!stage_->expose_region(const_cast<void*>(x3), x3_bytes, true, false, x3_rkey, err) ||
        !stage_->expose_region(y3, y3_bytes, false, true, y3_rkey, err) ||
        !stage_->expose_region(v_flag, sizeof(uint32_t), false, true, flag3_rkey, err)) {
        err = "rdma expert tier: push channel 2 registration: " + err;
        return false;
    }
    PushBind3 b{};
    b.magic = kMagicBind3;
    b.version = 1;
    b.x3_addr = reinterpret_cast<uint64_t>(x3);
    b.x3_rkey = x3_rkey;
    b.t3_cap = (uint32_t) (x3_bytes / row);
    b.y3_addr = reinterpret_cast<uint64_t>(y3);
    b.y3_rkey = y3_rkey;
    b.y3_rows = (uint32_t) (y3_bytes / row);
    b.flag3_addr = reinterpret_cast<uint64_t>(v_flag);
    b.flag3_rkey = flag3_rkey;
    if (!stage_->rdma_write_doorbell(0, &b, sizeof b, seq_ + 1, err)) {
        err = "rdma expert tier: push channel 2 bind: " + err;
        return false;
    }
    ++seq_;
    xv_base_ = reinterpret_cast<const float*>(x3);
    xv_tcap_ = (int64_t) (x3_bytes / row);
    v_flag_ = reinterpret_cast<volatile uint32_t*>(v_flag);
    push_vf_armed_ = true;
    return true;
}

bool RDMAExpertTier::push_begin_prefill(int64_t layer, int32_t n_tok, int32_t k, const int32_t* entries,
                                        int32_t n_entries, uint64_t& seq_out, std::string& err) {
    if (!push_pf_armed_) {
        err = "rdma expert tier: push channel 1 is not armed";
        return false;
    }
    if (layer < 0 || layer >= n_layers_ || n_tok < 1 || n_tok > kMaxPushTok || k < 1 || k > kMaxWidth ||
        n_entries < 0 || n_entries > n_tok * k || n_entries > kMaxPushEntries ||
        (n_entries > 0 && entries == nullptr)) {
        err = "rdma expert tier: bad prefill window geometry";
        return false;
    }
    PushReq* q = req_block_;
    q->layer = (int32_t) layer;
    q->n_tok = n_tok;
    q->k = k;
    q->n_entries = n_entries;
    q->channel = 1;
    if (n_entries > 0)
        std::memcpy(q->entries, entries, (size_t) n_entries * 2 * sizeof(int32_t));
    ++push_seq_;
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile uint64_t*>(&q->seq) = push_seq_;
    seq_out = push_seq_;
    ++launched_layers_;
    ++push_windows_;
    computed_ += n_entries;
    return true;
}

bool RDMAExpertTier::push_wait(uint64_t seq, std::string& err) {
    const uint32_t want = (uint32_t) seq;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms_);
    while (*h_flag_ != want) {
        if (std::chrono::steady_clock::now() > deadline) {
            err = "rdma expert tier: prefill window " + std::to_string(seq) + " timed out";
            return false;
        }
        STRATA_TIER_SPIN_PAUSE();
    }
    return true;
}

bool RDMAExpertTier::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                           const int32_t* kind, const int32_t* primary_res, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        double& acc;
        std::chrono::steady_clock::time_point s;
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count(); }
    } timer{ms_begin_, t0};

    const int64_t n = n_tok * k;
    pending_ = false;
    pending_v_ = false;
    owned_.assign((size_t) (n > 0 ? n : 0), 0);
    rows_.clear();
    if (!open_ || n <= 0 || n > kMaxEntries || n_tok > kMaxTokens || k > kMaxWidth || layer < 0 ||
        layer >= n_layers_) {
        if (n > 0) err = "rdma expert tier: invalid layer, routing width or window size";
        return n <= 0;   // an empty window is not an error; a malformed one is
    }
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        // Ownership is the caller's kind[] snapshot and nothing else: re-reading the residency table here
        // sampled it a SECOND time, and a mid-dispatch flip (the lent-slot refill racing the first token
        // after prefill) could leave a row with no owner at all - it fell through to the CPU pool, which
        // is fatal on hosts without the AVX-512 expert kernel (SIGILL in act_quant_q8_1).
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_) continue;
        owned_[(size_t) i] = 1;
        rows_.push_back((int32_t) i);
    }
    (void) primary_res;   // see above: the kind[] snapshot already encodes residency
    // PUSH MODE, single token: publish the miss list to the host-memory request block and nothing else.
    // No verbs, no waiting: the worker one-sided-reads this block and the doorbell activation, computes,
    // and RDMA-writes the rows into y_miss (= `out`) followed by the doorbell flag.  A zero-entry request
    // is published too - the flag is the graph's only gate, and somebody has to raise it every layer.
    if (push_armed_ && !decode_pull_ && n_tok == 1) {
        PushReq* q = req_block_;
        q->layer = (int32_t) layer;
        q->n_tok = 1;
        q->k = (int32_t) k;
        q->n_entries = (int32_t) rows_.size();
        q->channel = (int32_t) decode_channel_;   // 0 = first decode session, 3 = second (batch ladder)
        size_t j = 0;
        for (int64_t i = 0; i < n; ++i)
            if (owned_[(size_t) i]) {
                q->entries[2 * j] = (int32_t) i;
                q->entries[2 * j + 1] = ids[i];
                ++j;
            }
        ++push_seq_;
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<volatile uint64_t*>(&q->seq) = push_seq_;
        ++launched_layers_;
        ++push_windows_;
        computed_ += (int64_t) rows_.size();
        return true;
    }
    // PUSH MODE, channel 2 (verify windows): the same publish-only shape for a multi-token window.  The
    // worker one-sided-reads the window's activation segment out of the Verifier's mapped staging (x3 +
    // x_off), computes, and RDMA-writes the rows contiguous in entry order into the window's y_miss segment
    // (y3 + y_off, which IS the dispatch's `out`) followed by the window's own flag; finish() spins on it.
    // A zero-entry request is published too - finish() waits on the flag, and somebody has to raise it.
    if (push_vf_armed_ && n_tok > 1) {
        const size_t x_row = (size_t) kHidden * sizeof(float);
        const auto* xb = reinterpret_cast<const uint8_t*>(xv_base_);
        const auto* xp = reinterpret_cast<const uint8_t*>(x);
        if (xp < xb || (size_t) (xp - xb) % x_row != 0) {
            err = "rdma expert tier: a verify window's activation is outside the channel-2 block";
            return false;
        }
        const int64_t x_off = (int64_t) ((xp - xb) / x_row);
        if (x_off + n_tok > xv_tcap_ || (x_off + n_tok) * k > (int64_t) (xv_tcap_ * kMaxWidth)) {
            err = "rdma expert tier: a verify window exceeds the channel-2 block";
            return false;
        }
        PushReq* q = req_block_;
        q->layer = (int32_t) layer;
        q->n_tok = (int32_t) n_tok;
        q->k = (int32_t) k;
        q->n_entries = (int32_t) rows_.size();
        q->channel = 2;
        q->x_off = (int32_t) x_off;
        q->y_off = (int32_t) (x_off * k);
        size_t j = 0;
        for (int64_t i = 0; i < n; ++i)
            if (owned_[(size_t) i]) {
                q->entries[2 * j] = (int32_t) i;
                q->entries[2 * j + 1] = ids[i];
                ++j;
            }
        ++push_seq_;
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<volatile uint64_t*>(&q->seq) = push_seq_;
        pending_seq_ = push_seq_;
        pending_ = true;
        pending_v_ = true;
        ++launched_layers_;
        ++push_windows_;
        computed_ += (int64_t) rows_.size();
        return true;
    }
    if (rows_.empty()) return true;

    // The request: header, then ALL tokens' activations (the peer quantizes once per token and shares it
    // across that token's entries, the same split the CPU pool makes), then the {row, expert} pairs.
    const size_t x_bytes = (size_t) n_tok * (size_t) kHidden * sizeof(float);
    const size_t ent_bytes = rows_.size() * 2 * sizeof(int32_t);
    const size_t total = sizeof(ReqHeader) + x_bytes + ent_bytes;
    if (total > slot_bytes_) {
        err = "rdma expert tier: request exceeds the ring slot";
        return false;
    }
    req_.resize(total);
    ReqHeader h{};
    h.magic = kMagic;
    h.version = kVersion;
    h.layer = (int32_t) layer;
    h.n_tok = (int32_t) n_tok;
    h.k = (int32_t) k;
    h.n_entries = (int32_t) rows_.size();
    std::memcpy(req_.data(), &h, sizeof h);
    std::memcpy(req_.data() + sizeof h, x, x_bytes);
    int32_t* ent = reinterpret_cast<int32_t*>(req_.data() + sizeof h + x_bytes);
    size_t j = 0;
    for (int64_t i = 0; i < n; ++i)
        if (owned_[(size_t) i]) {
            ent[2 * j] = (int32_t) i;
            ent[2 * j + 1] = ids[i];
            ++j;
        }

    ++seq_;
    pending_seq_ = seq_;
    pending_slot_ = (uint32_t) ((seq_ - 1) % slots_);
    pending_entries_ = (int64_t) rows_.size();
    if (!stage_->rdma_write_doorbell(pending_slot_, req_.data(), total, pending_seq_, err)) {
        err = "rdma expert tier: " + err;
        return false;
    }
    pending_ = true;
    ++launched_layers_;
    computed_ += (int64_t) rows_.size();
    return true;
}

bool RDMAExpertTier::finish(float* out, std::string& err) {
    if (!pending_) return true;
    pending_ = false;
    const auto w0 = std::chrono::steady_clock::now();
    struct Timer {
        double& acc;
        std::chrono::steady_clock::time_point s;
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count(); }
    } timer{ms_wait_, w0};

    // PUSH channel 2: the worker's chained write lands the rows straight into `out` (the window's y_miss
    // segment) and then raises the window's flag with this request's seq.  Waiting for the flag is the whole
    // finish - there is nothing to pull and nothing to copy.
    if (pending_v_) {
        pending_v_ = false;
        (void) out;
        const uint32_t want = (uint32_t) pending_seq_;
        const auto deadline = w0 + std::chrono::milliseconds(wait_ms_);
        while (*v_flag_ != want) {
            if (std::chrono::steady_clock::now() > deadline) {
                err = "rdma expert tier: verify window " + std::to_string(pending_seq_) + " timed out";
                return false;
            }
            STRATA_TIER_SPIN_PAUSE();
        }
        // the rows never crossed this host, so returned_bytes_ does not count them
        return true;
    }

    const size_t payload = (size_t) pending_entries_ * (size_t) kHidden * sizeof(float);
    const size_t full = 8 + payload;   // the slot's 8-byte sequence header travels with the rows
    resp_.resize(full);
    const auto deadline = w0 + std::chrono::milliseconds(wait_ms_);
    // Poll just the 8-byte header (a ~2 us read) until the worker's release-store lands, then pull the rows
    // in one read.  Reading the whole slot every poll would put the payload on the wire once per iteration.
    uint64_t hdr = 0;
    while (true) {
        if (!stage_->rdma_read_slot(pending_slot_, resp_.data(), 8, err)) {
            err = "rdma expert tier: " + err;
            return false;
        }
        std::memcpy(&hdr, resp_.data(), 8);
        if (hdr == pending_seq_) break;
        if (std::chrono::steady_clock::now() > deadline) {
            err = "rdma expert tier: result wait timeout (seq " + std::to_string(pending_seq_) + ")";
            return false;
        }
    }
    if (payload > 0 && !stage_->rdma_read_slot(pending_slot_, resp_.data(), full, err)) {
        err = "rdma expert tier: " + err;
        return false;
    }
    std::memcpy(&hdr, resp_.data(), 8);
    if (hdr != pending_seq_) {   // the payload read must observe the same slot generation as the header poll
        err = "rdma expert tier: ring slot changed under the read";
        return false;
    }
    const float* rows = reinterpret_cast<const float*>(resp_.data() + 8);
    for (size_t j = 0; j < rows_.size(); ++j)
        std::memcpy(out + (size_t) rows_[j] * (size_t) kHidden, rows + j * (size_t) kHidden,
                    (size_t) kHidden * sizeof(float));
    returned_bytes_ += payload;
    return true;
}

// The worker publishes the PLE region record in ring slot 0 before it serves its first window (its table load
// takes tens of seconds, so the engine polls).  Slot 0's payload starts after the ring's 8-byte seq header.
bool RDMAExpertTier::fetch_ple_info(PleInfo& info, int timeout_ms, std::string& err) {
    if (!open_ || !stage_ || !stage_->is_rdma()) {
        err = "rdma expert tier: the PLE info fetch needs an open RDMA stage";
        return false;
    }
    std::vector<uint8_t> buf(8 + sizeof(PleInfo));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (!stage_->rdma_read_slot(0, buf.data(), buf.size(), err)) {
            err = "rdma expert tier: PLE info ring read: " + err;
            return false;
        }
        PleInfo got;
        std::memcpy(&got, buf.data() + 8, sizeof got);
        // v1: the original IQ4_NL-only record (no format fields). v2: format/fp8_scale are part of the
        // contract; anything else (a truncated write, a mismatched build) keeps polling till the deadline.
        const bool v1 = got.version == 1 && got.row_bytes == 90;
        const bool v2 = got.version == 2 &&
                        ((got.format == kPleFmtIq4Nl && got.row_bytes == 90) ||
                         (got.format == kPleFmtFp8E4m3 && got.row_bytes == 160 && got.fp8_scale > 0.0f));
        if (got.magic == PLE_INFO_MAGIC && (v1 || v2) && got.rows > 0 &&
            got.addr != 0 && got.rkey != 0) {
            if (v1) { got.format = kPleFmtIq4Nl; got.fp8_scale = 0.0f; }
            info = got;
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            err = "rdma expert tier: the worker published no PLE table in time (start it with --ple-gguf)";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

} // namespace strata::core
