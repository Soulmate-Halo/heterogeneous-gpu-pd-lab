#pragma once

// Transport for the contiguous-layer boundary.  The caller supplies one
// complete verify-window payload after its local CUDA graph and consumes the
// payload before launching the next graph; the transport never becomes a
// per-layer RPC.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {

enum class RemoteStageRole : uint8_t { Sender = 1, Receiver = 2 };

struct RemoteStageConfig {
    RemoteStageRole role = RemoteStageRole::Sender;
    std::string peer_host;
    std::string bind_host = "0.0.0.0";
    uint16_t port = 39580;
    std::string device;
    int gid_index = 0;
    int split_layer = -1;
    int timeout_ms = 5000;
    size_t max_payload = 64u * 1024u * 1024u;
    uint32_t slots = 4;              // one-sided result ring depth
    size_t slot_bytes = 64u * 1024u; // 8-byte seq header + payload per slot
    bool require_rdma = true;
};

struct RemoteStageWindow {
    uint64_t sequence = 0;
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    int32_t split_layer = -1;
};

struct RemoteStageAck {
    uint64_t sequence = 0;
    uint32_t accepted = 0;
    uint32_t status = 0;
};

class RemoteStage {
public:
    struct Impl;
    explicit RemoteStage(RemoteStageConfig config);
    ~RemoteStage();
    RemoteStage(const RemoteStage&) = delete;
    RemoteStage& operator=(const RemoteStage&) = delete;

    bool open(std::string& error);
    void close();
    bool is_open() const;
    bool is_rdma() const;
    const RemoteStageConfig& config() const { return config_; }

    bool send_window(const RemoteStageWindow& window, const void* payload, size_t bytes,
                     RemoteStageAck& ack, std::string& error);
    bool recv_window(RemoteStageWindow& window, void* payload, size_t capacity,
                     std::string& error);
    bool send_ack(const RemoteStageAck& ack, std::string& error);
    bool recv_commit(uint64_t sequence, uint32_t& accepted, std::string& error);
    bool send_commit(uint64_t sequence, uint32_t accepted, std::string& error);

    // One-sided flag-ring data plane.  The receiver exposes a result ring
    // (slots * slot_bytes, REMOTE_READ), an activation inbox (slot_bytes,
    // REMOTE_WRITE) and an 8-byte doorbell sequence (REMOTE_WRITE).  The
    // sender posts an activation and polls result slots without any peer-side
    // verbs completion, mirroring the same-process pinned-flag mechanism.
    bool rdma_write_doorbell(uint32_t slot, const void* data, size_t len, uint64_t seq, std::string& error);
    bool rdma_read_slot(uint32_t slot, void* dst, size_t len, std::string& error);
    bool poll_remote_seq(uint64_t& seq, std::string& error);
    uint8_t* local_ring_slot(uint32_t slot);   // receiver only, else nullptr
    uint8_t* local_inbox();                    // receiver only, else nullptr
    uint64_t load_local_doorbell();            // acquire load, receiver only
    // Receiver-side publish: copy payload into ring slot then release-store
    // the 8-byte seq header so the sender's RDMA read observes both together.
    bool ring_publish(uint32_t slot, const void* data, size_t len, uint64_t seq, std::string& error);

    // Push-channel extensions (the RDMA expert tier's zero-host-verbs mode; docs/RDMA_EXPERTS.md).  Either
    // role may call these once open: `expose_region` pins an extra buffer on this stage's PD so the PEER can
    // read or write it one-sided; `rdma_read_remote` pulls `len` bytes from a peer-exported address into a
    // locally registered buffer; `rdma_write_scatter` lands up to N rows at arbitrary offsets of one peer
    // region and then writes a 4-byte flag - one ordered WR chain per 48 rows, so on an RC QP the flag is
    // observed only after every row is.  All fail-closed.
    bool expose_region(const void* p, size_t bytes, bool remote_read, bool remote_write, uint32_t& rkey,
                       std::string& error);
    bool rdma_read_remote(uint64_t remote_addr, uint32_t rkey, void* dst, size_t len, std::string& error);
    // Batched one-sided READ of `n` scattered rows of `row_bytes` from a peer-exported table region: one
    // ordered WR chain per post (48 rows, the QP's max_send_wr budget with headroom), signalled on the last,
    // so a single poll completes the whole chain.  `dst` must lie inside a locally registered region
    // (expose_region) and the rows land contiguously at dst + i*row_bytes.
    bool rdma_read_rows(uint64_t remote_base, uint32_t rkey, const uint32_t* rows, size_t n,
                        uint32_t row_bytes, void* dst, std::string& error);
    struct RdmaRowSeg { const void* src; uint64_t remote_addr; uint32_t len; };
    bool rdma_write_scatter(const RdmaRowSeg* segs, size_t n, uint32_t data_rkey, uint64_t flag_addr,
                            uint32_t flag_rkey, uint32_t flag_val, std::string& error);

    uint64_t windows() const;
    uint64_t bytes_sent() const;
    uint64_t bytes_received() const;
    static bool selftest(std::string& error);

private:
    RemoteStageConfig config_;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
