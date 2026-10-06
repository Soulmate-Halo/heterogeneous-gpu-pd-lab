#pragma once

// include/strata/core/rdma_expert_tier.hpp - cold experts computed on a REMOTE machine over the one-sided
// RDMA flag ring (`remote_stage.hpp`), for hosts whose CPU cannot run the AVX-512 expert kernel at all.
//
// THE SHAPE, and why it is not an RPC: the peer (`tools/cold_expert_worker.cpp`) owns a result ring and an
// activation inbox in ITS registered memory.  `begin` packs the layer's unclaimed rows into one window and
// posts it with a chained write (payload -> inbox, then sequence -> doorbell); `finish` polls the ring slot's
// 8-byte sequence header with one-sided reads and pulls the rows back once the sequence lands.  The peer's CPU
// never touches the wire on the return path - it release-stores the sequence after the memcpy and moves on,
// the same flag mechanism the single-machine pinned-memory design uses.
//
// ONE WINDOW IN FLIGHT, BY CONSTRUCTION: the dispatch loop calls begin -> (CPU pool) -> finish per layer, so
// at most one request is outstanding and the ring depth only has to cover the caller's slot rotation.
//
// WIRE FORMAT (little-endian, packed, no implicit padding):
//   request payload:  ReqHeader{ u32 magic='RXE1', u32 version=1, i32 layer, i32 n_tok, i32 k, i32 n_entries }
//                     then f32 x[n_tok * 2560]  (the layer's normed activations, one row per token)
//                     then i32 entries[n_entries * 2]  ({row, expert} pairs, routing order)
//   response payload: f32 rows[n_entries * 2560], row j is entries[j]'s expert output, IN ENTRY ORDER.
// `row` is the flattened routing index (token * k + j); the peer derives the token as row / k.  Rows the
// peer cannot serve MUST NOT be claimed by `begin` in the first place - every claimed row is computed.
//
// FAIL-CLOSED, like the transport under it: no RDMA device, a geometry mismatch or a peer timeout is an
// error, never a fallback.

#include "strata/core/remote_stage.hpp"
#include "strata/core/remote_tier.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct RdmaExpertTierConfig {
    std::string peer_host;              ///< the cold-expert worker's host (Spark)
    uint16_t port = 39580;
    std::string device;                 ///< HCA name; empty = first verbs device
    int gid_index = 0;
    uint32_t slots = 4;                 ///< result ring depth (must match the worker)
    size_t slot_bytes = 4u << 20;       ///< ring slot bytes (must match the worker)
    /// 0 = Q2_0, 1 = NVFP4: the expert weight encoding this engine serves, published in the push bind so a
    /// mismatched worker (its --format says otherwise) refuses the pairing instead of computing plausible
    /// garbage out of the wrong arena (P4c; both sides rebuilt together, bind version bumped 1 -> 2).
    uint32_t expert_format = 0;
    int timeout_ms = 5000;              ///< per-completion verbs timeout
    int wait_ms = 30000;                ///< per-layer result wait budget
    /// PUSH MODE (default on): the single-token decode path stops driving the wire from this host entirely.
    /// `arm_push` registers the session's doorbell activation, the y_miss staging and the doorbell flag on
    /// this host's PD and hands their addresses to the worker; begin() then only publishes the miss list to
    /// a host-memory request block (plain stores, no verbs), and the WORKER one-sided-reads the request and
    /// the activation, computes, and RDMA-writes the rows straight into y_miss followed by the flag the
    /// token graph's doorbell_wait spins on - the exact mechanism the local CPU pool uses, with the wire in
    /// the middle.  finish() becomes a no-op and the host never posts a WR per layer.  Multi-token windows
    /// (verify) use push channel 2 when it is armed (arm_push_verify): the worker writes the rows straight
    /// into the window's y_miss segment and raises the window's own flag, which finish() spins on; a tier
    /// without channel 2 keeps the pull path above for them.
    bool push = true;
};

class RDMAExpertTier : public RemoteTier {
public:
    RDMAExpertTier() = default;
    ~RDMAExpertTier() override = default;
    RDMAExpertTier(const RDMAExpertTier&) = delete;
    RDMAExpertTier& operator=(const RDMAExpertTier&) = delete;

    /// Connects to an already-listening worker and validates the ring geometry.  Refuses (false) unless the
    /// data plane is real RDMA.
    bool open(const RdmaExpertTierConfig& config, int64_t n_layers, int64_t n_expert, std::string& err);
    void close();
    bool is_open() const { return open_; }

    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
               const int32_t* kind, const int32_t* primary_res, std::string& err) override;
    bool owns(int64_t index) const override {
        return index >= 0 && (size_t) index < owned_.size() && owned_[(size_t) index] != 0;
    }
    bool finish(float* out, std::string& err) override;
    /// Push mode: the peer RDMA-writes its rows straight into `out` (y_miss), so the dispatch must not
    /// zero them first.
    bool direct_write() const override { return push_armed_; }

    /// Arms push mode AFTER the session exists: registers the doorbell activation (x_f), the y_miss
    /// staging, the contiguous result block (y_contig, kPushYCDataOff + k_cap rows of n_embd f32) and the
    /// doorbell flag on this stage's PD and ships their addresses to the worker as a bind record through the
    /// flag ring.  The caller must then keep the doorbell flag writes to the worker
    /// (Doorbell::external_flag).  Fail-closed like open().
    bool arm_push(const void* x_f, size_t x_bytes, void* y_miss, size_t y_bytes, void* h_flag,
                  void* y_contig, size_t yc_bytes, std::string& err);
    bool push_armed() const { return push_armed_; }
    uint64_t push_windows() const { return push_windows_; }

    /// Batch ladder phase 2: arms push channel 3 (the second decode session's buffer set) at session setup.
    /// Call AFTER arm_push; the bind goes out as the next doorbell in the setup sequence.  Requests are
    /// published with `channel = 3` by setting set_decode_channel(3) before the second session's begin().
    bool arm_push_ch3(const void* x_f, size_t x_bytes, void* y_contig, size_t yc_bytes, void* h_flag,
                      std::string& err);
    bool push_ch3_armed() const { return push_ch3_armed_; }
    /// The channel the SINGLE-TOKEN push publish writes into the request block: 0 for the first session
    /// (default), 3 for the second.  Set per dispatch by the dual runner's pool adapter.
    void set_decode_channel(int ch) { decode_channel_ = ch; }
    int decode_channel() const { return decode_channel_; }

    /// Batch ladder phase 2: while set, the SINGLE-TOKEN path of begin() takes the legacy pull request even
    /// though push is armed.  The dual-session decode interleave runs both slots in pull mode: the push channel
    /// has exactly one registered buffer set (the worker RDMA-writes fixed addresses), so a second session's
    /// rows cannot be pushed, and mixing push and pull requests on one connection is a protocol question the
    /// worker was never asked.  Pull is per-call addressed, so two serialized pull calls share the link safely.
    /// Prefill/verify windows (n_tok > 1) are unaffected.  Restore to false when the interleave ends.
    void set_decode_pull(bool v) { decode_pull_ = v; }
    bool decode_pull() const { return decode_pull_; }

    /// PUSH channel 1 (batched prefill): registers the prefill path's activation block (t_cap tokens) and
    /// result block (t_cap * k rows) and ships them to the worker as an RXE3 bind record.  Call AFTER
    /// arm_push, once the prefill buffers exist.  The dtypes (kPushX*/kPushY*) declare the blocks' layouts;
    /// the x2/y2 byte counts must be whole rows of the declared dtype.  Fail-closed like arm_push.
    bool arm_push_prefill(const void* x2, size_t x2_bytes, void* y2, size_t y2_bytes, uint32_t x_dtype,
                          uint32_t y_dtype, std::string& err);
    bool push_prefill_armed() const { return push_pf_armed_; }
    /// Publishes one prefill window (channel 1): plain stores to the request block, seq last, no verbs.  The
    /// caller has already filled the x2 block (the worker one-sided-reads it).  `entries` are {row, expert}
    /// pairs in routing order, row = token * k + j.  Returns the window's seq for push_wait.
    bool push_begin_prefill(int64_t layer, int32_t n_tok, int32_t k, const int32_t* entries, int32_t n_entries,
                            uint64_t& seq_out, std::string& err);
    /// Spins on the doorbell flag until the worker's chained write lands this window's seq (the flag value for
    /// channel 1), wait_ms budget, fail-closed on timeout.
    bool push_wait(uint64_t seq, std::string& err);

    /// PUSH channel 2 (verify windows): registers the verifier's mapped activation block (v_tcap tokens of
    /// n_embd f32), its result block (v_rows rows of n_embd f32) and a doorbell flag of its own, and ships
    /// them to the worker as an RXE4 bind record.  Call AFTER arm_push, once the Verifier's staging exists.
    /// With channel 2 armed, a multi-token begin() publishes the miss list with channel = 2 and the token
    /// offset of the window's segment (x_off; y_off = x_off * k), and finish() spins on the channel's flag
    /// for the request's seq - the worker's chained write lands the rows straight into the segment, so
    /// finish() copies nothing and neither side posts a WR from this host.  Fail-closed like arm_push.
    bool arm_push_verify(const void* x3, size_t x3_bytes, void* y3, size_t y3_bytes, void* v_flag,
                         std::string& err);
    bool push_verify_armed() const { return push_vf_armed_; }

    /// PLE table over RDMA (the worker's --ple-gguf): the worker registers the whole IQ4_NL table as a
    /// REMOTE_READ region and publishes this record in ring slot 0 BEFORE it starts serving, so the engine
    /// can pick it up any time after open() and before its first window.  fetch_ple_info polls slot 0 until
    /// the magic shows up (the worker may still be loading its 28.8 GB) or the budget runs out.  Fail-closed.
    struct PleInfo {
        uint32_t magic;        ///< PLE_INFO_MAGIC
        uint32_t version;      ///< 1 = IQ4_NL 90 B rows; 2 = `format`/`fp8_scale` below are valid
        uint64_t addr;         ///< row 0's address in the worker's address space
        uint32_t rkey;         ///< the region's rkey
        uint32_t row_bytes;    ///< 90 (IQ4_NL) or 160 (FP8 E4M3)
        uint64_t rows;         ///< PLE_TABLE_ROWS
        uint32_t format;       ///< v2: kPleFmt* (v1 readers must treat the table as IQ4_NL)
        float fp8_scale;       ///< v2 + kPleFmtFp8E4m3: the table's single global dequant scale
    };
    static constexpr uint32_t kPleFmtIq4Nl = 0;      ///< 90 B rows, IQ4_NL blocks
    static constexpr uint32_t kPleFmtFp8E4m3 = 1;    ///< 160 B rows, one e4m3 byte per element x fp8_scale
    static constexpr uint32_t PLE_INFO_MAGIC = 0x52454C50;   // 'PLER'
    static constexpr uint64_t PLE_INFO_SEQ = 0x504C45ull;    // ring slot 0 header while the record sits there
    bool fetch_ple_info(PleInfo& info, int timeout_ms, std::string& err);
    /// The transport, for the PLE row source (rdma_read_rows / expose_region).  Valid after open().
    RemoteStage* stage() const { return stage_.get(); }

    /// Counters, same spirit as RemoteExperts'.
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    uint64_t returned_bytes() const { return returned_bytes_; }
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }

    /// The wire constants, exposed so the worker and the probe compile against ONE definition.
    static constexpr uint32_t kMagic = 0x31455852u; // little-endian "RXE1"
    static constexpr uint32_t kVersion = 1;
    static constexpr int64_t kHidden = 2560;        // the artifact's n_embd
    static constexpr int64_t kMaxTokens = 8;        // cpu::MAXT (the pull path's window bound)
    static constexpr int64_t kMaxWidth = 32;        // routing width bound (kind[] in the dispatch)
    static constexpr int64_t kMaxEntries = 128;     // the dispatch's kind[128] bounds n_tok * k (pull path)
    // PUSH channel 1 (batched prefill): one window covers a whole chunk of a layer, so the caps are the chunk's
    // T and T * k.  The MMQ prefill path is launch-bound below ~1K-token chunks (measured: gemm down runs at
    // ~10 us/row at 512-token chunks vs ~0.6 at 8192 upstream), so the cap sits at 2048 - the biggest chunk
    // whose result block (T*k rows = 210 MB pinned) the 3080's 14 GB host RAM holds comfortably.
    static constexpr int64_t kMaxPushTok = 2048;
    static constexpr int64_t kMaxPushEntries = kMaxPushTok * 10;

#pragma pack(push, 1)
    struct ReqHeader {
        uint32_t magic;
        uint32_t version;
        int32_t layer;
        int32_t n_tok;
        int32_t k;
        int32_t n_entries;
    };
#pragma pack(pop)

    /// Push-mode bind record, sent once through the flag ring right after arm_push.  Magic 'RXE2': the
    /// worker's first doorbell either carries this (push channel follows) or RXE1 (pure legacy peer).
    static constexpr uint32_t kMagicBind = 0x32455852u; // little-endian "RXE2"
#pragma pack(push, 1)
    struct PushBind {
        uint32_t magic;
        uint32_t version;
        uint64_t req_addr;    // the engine's PushReq block, REMOTE_READ
        uint32_t req_rkey;
        uint32_t n_embd;      // kHidden
        uint64_t x_addr;      // the doorbell activation (n_embd f32, one token), REMOTE_READ
        uint32_t x_rkey;
        uint32_t k_cap;       // y_miss rows capacity
        uint64_t y_addr;      // y_miss: row-major n_embd f32 rows, REMOTE_WRITE
        uint32_t y_rkey;
        uint32_t n_layers;
        uint64_t flag_addr;   // the token graph's doorbell flag (4 bytes), REMOTE_WRITE
        uint32_t flag_rkey;
        uint32_t expert_format;  ///< v2: 0 = Q2_0, 1 = NVFP4 (v1's pad; the worker refuses a mismatch)
        uint64_t yc_addr;    ///< v3: the contiguous entry-order result block, REMOTE_WRITE (layout below)
        uint32_t yc_rkey;
        uint32_t yc_rows;    ///< v3: data row capacity of yc (== k_cap)
    };
    /// The v3 contiguous result block (W1): instead of scattering `ne` rows into y_miss by routing row (one
    /// SGE per row), the worker's channel-0 chained write is [u32 count][u32 pad][i32 rows[count]] at offset
    /// 0, then `count` entry-order n_embd f32 rows at offset kPushYCDataOff, then the flag - two SGEs for any
    /// ne, RC keeps the chain in order.  The engine's token graph scatters the rows to their routing positions
    /// ON THE DEVICE after doorbell_wait (scatter_contig_mapped), zero-filling the rows the worker did not
    /// send (what the CPU pool's memset into y_miss used to do).  Bind version 2 peers have no yc fields, so
    /// the worker refuses them (both sides are rebuilt together, like the v1 -> v2 bump).
    static constexpr uint32_t kPushYCDataOff = 256;
    /// Push channel 1 (batched prefill) bind record, magic 'RXE3', sent through the flag ring right after the
    /// RXE2 bind when the engine's prefill path exists.  The worker keeps BOTH binds: a request block whose
    /// channel is 0 uses the RXE2 regions (decode, n_tok == 1, flag = layer + 1), channel 1 uses these (one
    /// window per layer per chunk, flag = the request's seq).
    static constexpr uint32_t kMagicBind2 = 0x33455852u; // little-endian "RXE3"
    // Channel-1 wire dtypes (bind version 2).  The engine OWNS both blocks, so it declares the layout and the
    // worker adapts; an engine and a worker built on different sides of this change refuse each other at the
    // bind (fail-closed), never silently misread a row.  x: 0 = n_embd f32 rows per token; 1 = the output of
    // quantize_q8_0_scaled packed as [n_tok * n_embd/32 blocks of 36 B][n_tok * n_embd/32 fp32 scales] -
    // exactly what moe_grouped_s2 eats, so the worker's own re-quantize drops out.  y: 0 = n_embd f32 rows;
    // 1 = n_embd f16 rows (the engine's scatter converts on load; half the wire bytes).
    static constexpr uint32_t kPushXf32 = 0, kPushXq8 = 1, kPushYf32 = 0, kPushYf16 = 1;
    struct PushBind2 {
        uint32_t magic;
        uint32_t version;    // 2: the dtype/stride fields below exist
        uint64_t x2_addr;    // the prefill activation block (t_cap tokens), REMOTE_READ
        uint32_t x2_rkey;
        uint32_t t_cap;      // token capacity of x2 (the init chunk)
        uint64_t y2_addr;    // the prefill result block (y2_rows rows), REMOTE_WRITE
        uint32_t y2_rkey;
        uint32_t y2_rows;    // row capacity of y2 (t_cap * k)
        uint32_t x2_dtype;   // kPushX*
        uint32_t y2_dtype;   // kPushY*
        uint32_t x2_row_bytes;   // one token in x2 (f32: n_embd*4; q8: n_embd/32*(36+4))
        uint32_t y2_row_bytes;   // one row in y2 (f32: n_embd*4; f16: n_embd*2)
    };
    /// Push channel 2 (verify windows) bind record, magic 'RXE4', sent through the flag ring after the RXE2
    /// bind (and the RXE3 one when the prefill path exists).  The window's activation and result blocks are
    /// the Verifier's mapped staging; a request carries x_off/y_off (token/row offsets of the window's
    /// segment, 0 for the other channels) and the worker writes the rows contiguous in entry order starting
    /// at y3 + y_off, then raises flag3 with the request's seq.  All rows are f32 on channel 2.
    static constexpr uint32_t kMagicBind3 = 0x34455852u; // little-endian "RXE4"
    struct PushBind3 {
        uint32_t magic;
        uint32_t version;    // 1
        uint64_t x3_addr;    // the verify window's activation block (t3_cap tokens), REMOTE_READ
        uint32_t x3_rkey;
        uint32_t t3_cap;     // token capacity of x3 (kVerifyMaxT)
        uint64_t y3_addr;    // the verify window's result block (y3_rows rows), REMOTE_WRITE
        uint32_t y3_rkey;
        uint32_t y3_rows;    // row capacity of y3 (t3_cap * k)
        uint64_t flag3_addr; // the window's own doorbell flag (4 bytes), REMOTE_WRITE
        uint32_t flag3_rkey;
        uint32_t pad;
    };
    /// Push channel 3 (batch ladder phase 2: the SECOND decode session's single-token pushes) bind record,
    /// magic 'RXE5', sent through the flag ring after the earlier binds at session setup.  Channel 3 is channel
    /// 0 with a second buffer set: same request block (the host serializes publishes and waits for the previous
    /// request's flag before publishing again), same contiguous entry-order result layout, same flag = layer+1
    /// semantics - but the rows and the flag land in the SECOND session's regions, so two token graphs can be
    /// in flight without either ever observing the other's rows.
    static constexpr uint32_t kMagicBind4 = 0x35455852u; // little-endian "RXE5"
    struct PushBind4 {
        uint32_t magic;
        uint32_t version;    // 1
        uint64_t x_addr;     // session B's doorbell activation (n_embd f32, one token), REMOTE_READ
        uint32_t x_rkey;
        uint32_t k_cap;      // y_contig row capacity
        uint64_t yc_addr;    // session B's contiguous entry-order result block, REMOTE_WRITE
        uint32_t yc_rkey;
        uint32_t yc_rows;    // == k_cap
        uint64_t flag_addr;  // session B's doorbell flag (4 bytes), REMOTE_WRITE
        uint32_t flag_rkey;
        uint32_t n_layers;
        uint32_t expert_format;  ///< 1 = NVFP4 (the worker refuses a mismatch, same as RXE2)
    };
    /// The head of PushReq, polled alone: one small read per iteration instead of the whole block.  Must be
    /// a prefix of PushReq (the static_assert is in the .cpp); seq lands first and is stored LAST by the
    /// publisher (release), so a head read that observes the new seq observes every field of the head, and the
    /// entries read that follows observes the whole request.  The whole head still shares one 64-byte cache
    /// line, so the polled bytes can never straddle two lines mid-publish.
    struct PushReqHead {
        uint64_t seq;
        int32_t layer;
        int32_t n_tok;
        int32_t k;
        int32_t n_entries;
        int32_t channel;     // 0 = decode (RXE2 regions), 1 = prefill (RXE3 regions), 2 = verify (RXE4)
        int32_t x_off;       // channel 2: the window segment's first token in x3; 0 elsewhere
        int32_t y_off;       // channel 2: the segment's first result row in y3 (= x_off * k); 0 elsewhere
    };
    /// The engine-side request block the worker one-sided-reads.  seq is stored LAST, release: a read that
    /// observes the new seq observes every field of the request.  alignas(64): the head shares one cache line,
    /// so the polled bytes can never straddle two lines mid-publish.
    struct alignas(64) PushReq {
        uint64_t seq;
        int32_t layer;
        int32_t n_tok;        // channel 0: always 1; channel 1: the chunk's T; channel 2: the window segment's n
        int32_t k;
        int32_t n_entries;
        int32_t channel;      // 0 = decode, 1 = prefill, 2 = verify
        int32_t x_off;        // channel 2: token offset of the segment in x3; 0 elsewhere
        int32_t y_off;        // channel 2: row offset of the segment in y3; 0 elsewhere
        int32_t entries[2 * kMaxPushEntries];   // {row, expert}, routing order; row = token * k + j
    };
#pragma pack(pop)

private:
    std::unique_ptr<RemoteStage> stage_;  // built in open(): the config is not valid until then
    bool open_ = false;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int wait_ms_ = 30000;
    uint32_t slots_ = 4;
    size_t slot_bytes_ = 0;
    uint32_t expert_format_ = 0;   ///< RdmaExpertTierConfig::expert_format, published in the push bind
    uint64_t seq_ = 0;                 ///< last POSTED sequence; 0 = nothing posted yet
    bool pending_ = false;             ///< a begin() is waiting for its finish()
    uint64_t pending_seq_ = 0;
    uint32_t pending_slot_ = 0;
    int64_t pending_entries_ = 0;
    std::vector<uint8_t> owned_;       ///< this layer's claims, n_tok * k entries
    std::vector<int32_t> rows_;        ///< claimed rows (flattened routing index), entry order
    std::vector<uint8_t> req_;         ///< staging for the request payload
    std::vector<uint8_t> resp_;        ///< staging for the ring slot read
    int64_t computed_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    // push mode
    bool push_cfg_ = false;
    bool push_armed_ = false;
    bool push_ch3_armed_ = false;
    int decode_channel_ = 0;   ///< see set_decode_channel: the channel single-token pushes publish on
    bool decode_pull_ = false;   ///< see set_decode_pull: pull mode for the single-token path while armed
    PushReq* req_block_ = nullptr;   // plain host memory (only the CPU and the NIC touch it), registered at arm
    volatile uint32_t* h_flag_ = nullptr;  // the doorbell flag, kept from arm_push for push_wait
    uint64_t push_seq_ = 0;
    uint64_t push_windows_ = 0;
    // push channel 1 (prefill)
    bool push_pf_armed_ = false;
    // push channel 2 (verify windows)
    bool push_vf_armed_ = false;
    const float* xv_base_ = nullptr;            // x3's base: a multi-token begin() derives x_off from it
    int64_t xv_tcap_ = 0;                       // x3's token capacity
    volatile uint32_t* v_flag_ = nullptr;       // the window's doorbell flag, kept for finish()
    bool pending_v_ = false;                    ///< the pending begin() is a channel-2 (verify) window
};

} // namespace strata::core
