#include "strata/core/remote_stage.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#if defined(STRATA_REMOTE_STAGE_VERBS)
#  include <arpa/inet.h>
#  include <infiniband/verbs.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
#  include <deque>
#  include <vector>
#endif

namespace strata::core {
namespace {

constexpr uint32_t kMagic = 0x31534752u; // little-endian "RGS1"
constexpr uint16_t kVersion = 1;
constexpr uint16_t kWindow = 1;
constexpr uint16_t kAck = 2;
constexpr uint16_t kCommit = 3;
constexpr uint32_t kMaxHidden = 1u << 20;

#pragma pack(push, 1)
struct WireMessage {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint64_t sequence;
    int32_t split_layer;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t accepted;
    uint32_t status;
    uint32_t payload_bytes;
    uint32_t payload_crc;
};

struct QpInfo {
    uint32_t qp_num;
    uint32_t psn;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t rkey;
    uint64_t address;
    uint64_t bytes;
    // one-sided flag ring (receiver exports; sender validates geometry)
    uint32_t ring_slots;
    uint32_t slot_bytes;
    uint32_t ring_rkey;
    uint32_t inbox_rkey;
    uint32_t bell_rkey;
    uint64_t ring_address;
    uint64_t inbox_address;
    uint64_t bell_address;
};
#pragma pack(pop)

uint32_t crc32(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= p[i];
        for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

bool validate_message(const WireMessage& m, uint16_t type, size_t max_payload, std::string& error) {
    if (m.magic != kMagic || m.version != kVersion || m.type != type) {
        error = "remote stage: bad message header";
        return false;
    }
    if (m.payload_bytes > max_payload || m.hidden > kMaxHidden) {
        error = "remote stage: payload geometry exceeds configured bounds";
        return false;
    }
    if (type == kWindow && (m.tokens == 0 || m.hidden == 0)) {
        error = "remote stage: empty window geometry";
        return false;
    }
    if (m.hidden != 0 && m.tokens > std::numeric_limits<uint32_t>::max() / m.hidden) {
        error = "remote stage: token geometry overflow";
        return false;
    }
    return true;
}

} // namespace

struct RemoteStage::Impl {
    explicit Impl(RemoteStageConfig c) : config(std::move(c)) {}
    RemoteStageConfig config;
    bool opened = false;
    bool rdma = false;
    uint64_t windows = 0;
    uint64_t sent = 0;
    uint64_t received = 0;
#if defined(STRATA_REMOTE_STAGE_VERBS)
    int control = -1;
    ibv_context* context = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_qp* qp = nullptr;
    ibv_mr* tx_mr = nullptr;
    ibv_mr* rx_mr = nullptr;
    ibv_mr* ring_mr = nullptr;   // receiver: result ring (REMOTE_READ)
    ibv_mr* inbox_mr = nullptr;  // receiver: activation inbox (REMOTE_WRITE)
    ibv_mr* bell_mr = nullptr;   // receiver: 8-byte doorbell seq (REMOTE_WRITE)
    ibv_mr* snd_mr = nullptr;    // sender: local send/read buffer
    struct Extra { const uint8_t* base; size_t bytes; ibv_mr* mr; };
    std::vector<Extra> extra;    // expose_region registry: lkey lookup by address range
    uint32_t flag_slot = 0;      // scatter-write flag staging (registered lazily)
    ibv_mr* flag_slot_mr = nullptr;
    std::vector<uint8_t> tx;
    std::vector<uint8_t> rx;
    std::vector<uint8_t> ring;   // slots * slot_bytes
    std::vector<uint8_t> inbox;  // slot_bytes
    std::vector<uint8_t> bell;   // 8 bytes seq
    std::vector<uint8_t> snd;    // max(slot_bytes, 8) for WR and RDMA_READ
    QpInfo local{};
    QpInfo peer{};
    uint8_t port = 1;
    std::mutex mu;
    std::deque<ibv_wc> completions;
#endif
};

RemoteStage::RemoteStage(RemoteStageConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>(config_)) {}
RemoteStage::~RemoteStage() { close(); }
bool RemoteStage::is_open() const { return impl_->opened; }
bool RemoteStage::is_rdma() const { return impl_->opened && impl_->rdma; }
uint64_t RemoteStage::windows() const { return impl_->windows; }
uint64_t RemoteStage::bytes_sent() const { return impl_->sent; }
uint64_t RemoteStage::bytes_received() const { return impl_->received; }

#if defined(STRATA_REMOTE_STAGE_VERBS)
namespace {

void close_fd(int& fd) {
    if (fd >= 0) { ::close(fd); fd = -1; }
}

bool write_all(int fd, const void* data, size_t bytes, int timeout_ms, std::string& error) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t done = 0;
    while (done < bytes) {
        pollfd f{fd, POLLOUT, 0};
        if (::poll(&f, 1, timeout_ms) <= 0) { error = "remote stage: control write timeout"; return false; }
        const ssize_t n = ::send(fd, p + done, bytes - done, MSG_NOSIGNAL);
        if (n <= 0) { error = "remote stage: control write failed"; return false; }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool read_all(int fd, void* data, size_t bytes, int timeout_ms, std::string& error) {
    auto* p = static_cast<uint8_t*>(data);
    size_t done = 0;
    while (done < bytes) {
        pollfd f{fd, POLLIN, 0};
        if (::poll(&f, 1, timeout_ms) <= 0) { error = "remote stage: control read timeout"; return false; }
        const ssize_t n = ::recv(fd, p + done, bytes - done, MSG_WAITALL);
        if (n <= 0) { error = "remote stage: control read failed"; return false; }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool modify_init(ibv_qp* qp, uint8_t port, std::string& error) {
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT; a.port_num = port; a.pkey_index = 0;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        error = "remote stage: QP INIT failed"; return false;
    }
    return true;
}

bool modify_rtr(ibv_qp* qp, const QpInfo& peer, uint8_t port, int gid_index, std::string& error) {
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_1024; a.dest_qp_num = peer.qp_num;
    a.rq_psn = peer.psn; a.max_dest_rd_atomic = 16; a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1; a.ah_attr.dlid = peer.lid; a.ah_attr.sl = 0;
    a.ah_attr.src_path_bits = 0; a.ah_attr.port_num = port;
    std::memcpy(&a.ah_attr.grh.dgid, peer.gid, 16);
    a.ah_attr.grh.sgid_index = static_cast<uint8_t>(gid_index); a.ah_attr.grh.hop_limit = 1;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
        error = "remote stage: QP RTR failed"; return false;
    }
    return true;
}

bool modify_rts(ibv_qp* qp, uint32_t psn, std::string& error) {
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTS; a.sq_psn = psn; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.max_rd_atomic = 16;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                               IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
        error = "remote stage: QP RTS failed"; return false;
    }
    return true;
}

bool ensure_buffers(RemoteStage::Impl& x, size_t payload_bytes, std::string& error) {
    if (payload_bytes > x.config.max_payload || payload_bytes > std::numeric_limits<uint32_t>::max()) {
        error = "remote stage: payload exceeds configured bounds";
        return false;
    }
    const size_t bytes = sizeof(WireMessage) + payload_bytes;
    if (x.tx.size() >= bytes && x.rx.size() >= bytes && x.tx_mr && x.rx_mr) return true;
    if (x.tx_mr) { ibv_dereg_mr(x.tx_mr); x.tx_mr = nullptr; }
    if (x.rx_mr) { ibv_dereg_mr(x.rx_mr); x.rx_mr = nullptr; }
    x.tx.resize(bytes);
    x.rx.resize(bytes);
    x.tx_mr = ibv_reg_mr(x.pd, x.tx.data(), x.tx.size(), IBV_ACCESS_LOCAL_WRITE);
    x.rx_mr = ibv_reg_mr(x.pd, x.rx.data(), x.rx.size(), IBV_ACCESS_LOCAL_WRITE);
    if (!x.tx_mr || !x.rx_mr) {
        error = "remote stage: memory registration failed (check RLIMIT_MEMLOCK)";
        return false;
    }
    return true;
}

bool poll_wc(RemoteStage::Impl& x, uint64_t wr_id, int timeout_ms, std::string& error) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        ibv_wc wc{};
        if (!x.completions.empty()) {
            wc = x.completions.front();
            x.completions.pop_front();
        } else {
            const int n = ibv_poll_cq(x.cq, 1, &wc);
            if (n < 0) { error = "remote stage: CQ poll failed"; return false; }
            if (n == 0) { std::this_thread::yield(); continue; }
        }
        if (wc.status != IBV_WC_SUCCESS) { error = "remote stage: completion failed"; return false; }
        if (wc.wr_id != wr_id) { x.completions.push_back(wc); continue; }
        return true;
    }
    error = "remote stage: completion timeout"; return false;
}

bool post_recv(RemoteStage::Impl& x, std::string& error) {
    ibv_sge sge{}; sge.addr = reinterpret_cast<uint64_t>(x.rx.data()); sge.length = static_cast<uint32_t>(x.rx.size());
    sge.lkey = x.rx_mr->lkey; ibv_recv_wr wr{}; wr.wr_id = 2; wr.sg_list = &sge; wr.num_sge = 1;
    ibv_recv_wr* bad = nullptr;
    if (ibv_post_recv(x.qp, &wr, &bad) != 0) { error = "remote stage: post receive failed"; return false; }
    return true;
}

bool post_send(RemoteStage::Impl& x, size_t bytes, std::string& error) {
    ibv_sge sge{}; sge.addr = reinterpret_cast<uint64_t>(x.tx.data()); sge.length = static_cast<uint32_t>(bytes); sge.lkey = x.tx_mr->lkey;
    ibv_send_wr wr{}; wr.wr_id = 1; wr.opcode = IBV_WR_SEND; wr.send_flags = IBV_SEND_SIGNALED; wr.sg_list = &sge; wr.num_sge = 1;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(x.qp, &wr, &bad) != 0) { error = "remote stage: post send failed"; return false; }
    return true;
}

} // namespace
#endif

bool RemoteStage::open(std::string& error) {
    close();
    if (config_.port == 0 || config_.max_payload < sizeof(WireMessage) || config_.timeout_ms <= 0 ||
        config_.slots == 0 || config_.slots > 4096 || config_.slot_bytes < 16 ||
        (config_.slot_bytes % 8) != 0 || config_.slot_bytes > (64u << 20)) {
        error = "remote stage: invalid configuration"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    error = "remote stage: libibverbs support is not compiled in"; return false;
#else
    Impl& x = *impl_; x.tx.resize(sizeof(WireMessage)); x.rx.resize(sizeof(WireMessage));
    addrinfo hints{}; hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC; addrinfo* ai = nullptr;
    const std::string service = std::to_string(config_.port);
    if (config_.role == RemoteStageRole::Sender) {
        if (config_.peer_host.empty()) { error = "remote stage: sender needs --peer"; return false; }
        if (::getaddrinfo(config_.peer_host.c_str(), service.c_str(), &hints, &ai) != 0) { error = "remote stage: peer lookup failed"; return false; }
        for (addrinfo* p = ai; p; p = p->ai_next) {
            x.control = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
            if (x.control >= 0 && ::connect(x.control, p->ai_addr, p->ai_addrlen) == 0) break;
            close_fd(x.control);
        }
        ::freeaddrinfo(ai);
        if (x.control < 0) { error = "remote stage: peer control connection failed"; return false; }
    } else {
        int s = ::socket(AF_INET6, SOCK_STREAM, 0); if (s < 0) { error = "remote stage: listen socket failed"; return false; }
        int one = 1; ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in6 sa{}; sa.sin6_family = AF_INET6; sa.sin6_port = htons(config_.port); sa.sin6_addr = in6addr_any;
        if (::bind(s, reinterpret_cast<sockaddr*>(&sa), sizeof sa) != 0 || ::listen(s, 1) != 0) { ::close(s); error = "remote stage: listen failed"; return false; }
        pollfd f{s, POLLIN, 0}; if (::poll(&f, 1, config_.timeout_ms) <= 0) { ::close(s); error = "remote stage: listen timeout"; return false; }
        x.control = ::accept(s, nullptr, nullptr); ::close(s); if (x.control < 0) { error = "remote stage: accept failed"; return false; }
    }
    int ndev = 0; ibv_device** devices = ibv_get_device_list(&ndev);
    if (!devices || ndev == 0) { error = "remote stage: no RDMA device"; close(); return false; }
    ibv_device* selected = nullptr;
    for (int i = 0; i < ndev; ++i) if (config_.device.empty() || config_.device == ibv_get_device_name(devices[i])) { selected = devices[i]; break; }
    if (!selected) { ibv_free_device_list(devices); error = "remote stage: requested HCA not found"; close(); return false; }
    x.context = ibv_open_device(selected); ibv_free_device_list(devices);
    if (!x.context) { error = "remote stage: ibv_open_device failed"; close(); return false; }
    ibv_device_attr da{}; ibv_port_attr pa{}; ibv_gid gid{};
    if (ibv_query_device(x.context, &da) != 0 || ibv_query_port(x.context, x.port, &pa) != 0 || pa.state != IBV_PORT_ACTIVE ||
        ibv_query_gid(x.context, x.port, config_.gid_index, &gid) != 0) { error = "remote stage: HCA port/GID is not active"; close(); return false; }
    std::memcpy(x.local.gid, &gid, 16); x.local.lid = pa.lid;
    x.local.psn = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) & 0xffffffu;
    x.pd = ibv_alloc_pd(x.context); x.cq = ibv_create_cq(x.context, 128, nullptr, nullptr, 0);
    if (!x.pd || !x.cq) { error = "remote stage: PD/CQ creation failed"; close(); return false; }
    ibv_qp_init_attr qi{}; qi.send_cq=x.cq; qi.recv_cq=x.cq; qi.cap.max_send_wr=64; qi.cap.max_recv_wr=32; qi.cap.max_send_sge=1; qi.cap.max_recv_sge=1; qi.qp_type=IBV_QPT_RC;
    x.qp = ibv_create_qp(x.pd, &qi); if (!x.qp) { error = "remote stage: QP creation failed"; close(); return false; }
    x.local.qp_num = x.qp->qp_num;
    // one-sided flag ring: receiver exports ring/inbox/doorbell, sender pins a
    // local work buffer; geometry is exchanged with the QP attributes so both
    // sides fail closed on any mismatch.
    const uint32_t slots = x.config.slots ? x.config.slots : 1;
    const size_t slot_bytes = x.config.slot_bytes >= 8 ? x.config.slot_bytes : 8;
    x.local.ring_slots = slots;
    x.local.slot_bytes = static_cast<uint32_t>(slot_bytes);
    if (x.config.role == RemoteStageRole::Receiver) {
        x.ring.assign(static_cast<size_t>(slots) * slot_bytes, 0);
        x.inbox.assign(slot_bytes, 0);
        x.bell.assign(8, 0);
        x.ring_mr = ibv_reg_mr(x.pd, x.ring.data(), x.ring.size(), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        x.inbox_mr = ibv_reg_mr(x.pd, x.inbox.data(), x.inbox.size(), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        x.bell_mr = ibv_reg_mr(x.pd, x.bell.data(), x.bell.size(), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!x.ring_mr || !x.inbox_mr || !x.bell_mr) { error = "remote stage: flag ring registration failed (check RLIMIT_MEMLOCK)"; close(); return false; }
        x.local.ring_rkey = x.ring_mr->rkey; x.local.inbox_rkey = x.inbox_mr->rkey; x.local.bell_rkey = x.bell_mr->rkey;
        x.local.ring_address = reinterpret_cast<uint64_t>(x.ring.data());
        x.local.inbox_address = reinterpret_cast<uint64_t>(x.inbox.data());
        x.local.bell_address = reinterpret_cast<uint64_t>(x.bell.data());
        x.local.rkey = x.ring_mr->rkey; x.local.address = x.local.ring_address; x.local.bytes = x.ring.size();
    } else {
        x.snd.assign(slot_bytes + 8, 0);
        x.snd_mr = ibv_reg_mr(x.pd, x.snd.data(), x.snd.size(), IBV_ACCESS_LOCAL_WRITE);
        if (!x.snd_mr) { error = "remote stage: sender buffer registration failed (check RLIMIT_MEMLOCK)"; close(); return false; }
        x.local.rkey = x.snd_mr->rkey; x.local.address = reinterpret_cast<uint64_t>(x.snd.data()); x.local.bytes = x.snd.size();
    }
    if (!ensure_buffers(x, 0, error) || !modify_init(x.qp,x.port,error) ||
        !write_all(x.control,&x.local,sizeof x.local,config_.timeout_ms,error) ||
        !read_all(x.control,&x.peer,sizeof x.peer,config_.timeout_ms,error) ||
        !modify_rtr(x.qp,x.peer,x.port,config_.gid_index,error) ||
        !modify_rts(x.qp,x.local.psn,error)) { close(); return false; }
    if (x.config.role == RemoteStageRole::Sender &&
        (x.peer.ring_slots != x.local.ring_slots || x.peer.slot_bytes != x.local.slot_bytes ||
         !x.peer.ring_rkey || !x.peer.inbox_rkey || !x.peer.bell_rkey)) {
        error = "remote stage: flag ring geometry mismatch with peer"; close(); return false;
    }
    x.rdma=true; x.opened=true; return true;
#endif
}

void RemoteStage::close() {
#if defined(STRATA_REMOTE_STAGE_VERBS)
    Impl& x=*impl_; for (auto& e : x.extra) if (e.mr) ibv_dereg_mr(e.mr); x.extra.clear(); if (x.flag_slot_mr) ibv_dereg_mr(x.flag_slot_mr); x.flag_slot_mr=nullptr; if (x.qp) ibv_destroy_qp(x.qp); if (x.tx_mr) ibv_dereg_mr(x.tx_mr); if (x.rx_mr) ibv_dereg_mr(x.rx_mr); if (x.ring_mr) ibv_dereg_mr(x.ring_mr); if (x.inbox_mr) ibv_dereg_mr(x.inbox_mr); if (x.bell_mr) ibv_dereg_mr(x.bell_mr); if (x.snd_mr) ibv_dereg_mr(x.snd_mr); if (x.cq) ibv_destroy_cq(x.cq); if (x.pd) ibv_dealloc_pd(x.pd); if (x.context) ibv_close_device(x.context); close_fd(x.control);
    x.qp=nullptr; x.tx_mr=x.rx_mr=nullptr; x.ring_mr=x.inbox_mr=x.bell_mr=x.snd_mr=nullptr; x.cq=nullptr; x.pd=nullptr; x.context=nullptr;
#endif
    impl_->opened=false; impl_->rdma=false;
}

bool RemoteStage::send_window(const RemoteStageWindow& w, const void* payload, size_t bytes, RemoteStageAck& ack, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Sender || !payload || bytes > config_.max_payload) { error="remote stage: sender is not open or payload is invalid"; return false; }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)w; (void)payload; (void)bytes; (void)ack; error="remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x=*impl_; std::lock_guard<std::mutex> lk(x.mu); if (!ensure_buffers(x, bytes, error)) return false; WireMessage m{}; m.magic=kMagic; m.version=kVersion; m.type=kWindow; m.sequence=w.sequence; m.split_layer=w.split_layer; m.tokens=w.tokens; m.hidden=w.hidden; m.payload_bytes=static_cast<uint32_t>(bytes); m.payload_crc=crc32(payload,bytes);
    std::memcpy(x.tx.data(),&m,sizeof m); std::memcpy(x.tx.data()+sizeof m,payload,bytes);
    if (!post_recv(x,error)||!post_send(x,sizeof m+bytes,error)||!poll_wc(x,1,config_.timeout_ms,error)||!poll_wc(x,2,config_.timeout_ms,error)) return false;
    WireMessage am{}; std::memcpy(&am,x.rx.data(),sizeof am); if (!validate_message(am,kAck,config_.max_payload,error)||am.sequence!=w.sequence) return false;
    ack={am.sequence,am.accepted,am.status}; ++x.windows; x.sent+=bytes; return ack.status==0;
#endif
}

bool RemoteStage::recv_window(RemoteStageWindow& w, void* payload, size_t capacity, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Receiver || !payload) { error="remote stage: receiver is not open or payload is invalid"; return false; }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)w; (void)payload; (void)capacity; error="remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x=*impl_; std::lock_guard<std::mutex> lk(x.mu); if (!ensure_buffers(x, capacity, error) || !post_recv(x,error) || !poll_wc(x,2,config_.timeout_ms,error)) return false;
    WireMessage m{}; std::memcpy(&m,x.rx.data(),sizeof m); if (!validate_message(m,kWindow,config_.max_payload,error)||m.payload_bytes>capacity) return false;
    if (crc32(x.rx.data()+sizeof m,m.payload_bytes)!=m.payload_crc) { error="remote stage: payload CRC mismatch"; return false; }
    w={m.sequence,m.tokens,m.hidden,m.split_layer}; std::memcpy(payload,x.rx.data()+sizeof m,m.payload_bytes); x.received+=m.payload_bytes; return true;
#endif
}

bool RemoteStage::send_ack(const RemoteStageAck& a, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Receiver) { error="remote stage: receiver is not open"; return false; }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)a; error="remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x=*impl_; std::lock_guard<std::mutex> lk(x.mu); WireMessage m{}; m.magic=kMagic; m.version=kVersion; m.type=kAck; m.sequence=a.sequence; m.accepted=a.accepted; m.status=a.status; std::memcpy(x.tx.data(),&m,sizeof m); if (!post_send(x,sizeof m,error)||!poll_wc(x,1,config_.timeout_ms,error)) return false; return true;
#endif
}

bool RemoteStage::send_commit(uint64_t sequence, uint32_t accepted, std::string& error) {
    if (!is_open()) { error="remote stage: not open"; return false; }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)sequence; (void)accepted; error="remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x=*impl_; std::lock_guard<std::mutex> lk(x.mu); WireMessage m{}; m.magic=kMagic; m.version=kVersion; m.type=kCommit; m.sequence=sequence; m.accepted=accepted; std::memcpy(x.tx.data(),&m,sizeof m); return post_send(x,sizeof m,error)&&poll_wc(x,1,config_.timeout_ms,error);
#endif
}

bool RemoteStage::recv_commit(uint64_t sequence, uint32_t& accepted, std::string& error) {
    if (!is_open()) { error="remote stage: not open"; return false; }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)sequence; (void)accepted; error="remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x=*impl_; std::lock_guard<std::mutex> lk(x.mu); if (!ensure_buffers(x, 0, error) || !post_recv(x,error) || !poll_wc(x,2,config_.timeout_ms,error)) return false; WireMessage m{}; std::memcpy(&m,x.rx.data(),sizeof m); if (!validate_message(m,kCommit,config_.max_payload,error)||m.sequence!=sequence) return false; accepted=m.accepted; return true;
#endif
}

bool RemoteStage::rdma_write_doorbell(uint32_t slot, const void* data, size_t len, uint64_t seq, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Sender || !data || len == 0 ||
        slot >= config_.slots || len > config_.slot_bytes) {
        error = "remote stage: doorbell write arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)slot; (void)data; (void)len; (void)seq; error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    std::memcpy(x.snd.data(), data, len);
    std::memcpy(x.snd.data() + len, &seq, sizeof seq);
    ibv_sge sge1{}; sge1.addr = reinterpret_cast<uint64_t>(x.snd.data()); sge1.length = static_cast<uint32_t>(len); sge1.lkey = x.snd_mr->lkey;
    ibv_sge sge2{}; sge2.addr = reinterpret_cast<uint64_t>(x.snd.data() + len); sge2.length = 8; sge2.lkey = x.snd_mr->lkey;
    ibv_send_wr wr2{}; wr2.wr_id = 4; wr2.opcode = IBV_WR_RDMA_WRITE; wr2.send_flags = IBV_SEND_SIGNALED;
    wr2.sg_list = &sge2; wr2.num_sge = 1;
    wr2.wr.rdma.remote_addr = x.peer.bell_address; wr2.wr.rdma.rkey = x.peer.bell_rkey;
    ibv_send_wr wr1{}; wr1.wr_id = 3; wr1.opcode = IBV_WR_RDMA_WRITE;
    wr1.sg_list = &sge1; wr1.num_sge = 1; wr1.next = &wr2;
    wr1.wr.rdma.remote_addr = x.peer.inbox_address; wr1.wr.rdma.rkey = x.peer.inbox_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(x.qp, &wr1, &bad) != 0) { error = "remote stage: doorbell post failed"; return false; }
    if (!poll_wc(x, 4, config_.timeout_ms, error)) return false;
    x.sent += len; return true;
#endif
}

bool RemoteStage::rdma_read_slot(uint32_t slot, void* dst, size_t len, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Sender || !dst || len == 0 ||
        slot >= config_.slots || len > config_.slot_bytes) {
        error = "remote stage: ring read arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)slot; (void)dst; (void)len; error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    ibv_sge sge{}; sge.addr = reinterpret_cast<uint64_t>(x.snd.data()); sge.length = static_cast<uint32_t>(len); sge.lkey = x.snd_mr->lkey;
    ibv_send_wr wr{}; wr.wr_id = 5; wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.wr.rdma.remote_addr = x.peer.ring_address + static_cast<uint64_t>(slot) * x.peer.slot_bytes;
    wr.wr.rdma.rkey = x.peer.ring_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(x.qp, &wr, &bad) != 0) { error = "remote stage: ring read post failed"; return false; }
    if (!poll_wc(x, 5, config_.timeout_ms, error)) return false;
    std::memcpy(dst, x.snd.data(), len); x.received += len; return true;
#endif
}

bool RemoteStage::poll_remote_seq(uint64_t& seq, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Sender) {
        error = "remote stage: sender is not open"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)seq; error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    ibv_sge sge{}; sge.addr = reinterpret_cast<uint64_t>(x.snd.data()); sge.length = 8; sge.lkey = x.snd_mr->lkey;
    ibv_send_wr wr{}; wr.wr_id = 6; wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.wr.rdma.remote_addr = x.peer.bell_address; wr.wr.rdma.rkey = x.peer.bell_rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(x.qp, &wr, &bad) != 0) { error = "remote stage: doorbell read post failed"; return false; }
    if (!poll_wc(x, 6, config_.timeout_ms, error)) return false;
    std::memcpy(&seq, x.snd.data(), 8); return true;
#endif
}

uint8_t* RemoteStage::local_ring_slot(uint32_t slot) {
#if defined(STRATA_REMOTE_STAGE_VERBS)
    if (!impl_->opened || config_.role != RemoteStageRole::Receiver || slot >= config_.slots || impl_->ring.empty()) return nullptr;
    return impl_->ring.data() + static_cast<size_t>(slot) * config_.slot_bytes;
#else
    (void)slot; return nullptr;
#endif
}

uint8_t* RemoteStage::local_inbox() {
#if defined(STRATA_REMOTE_STAGE_VERBS)
    if (!impl_->opened || config_.role != RemoteStageRole::Receiver || impl_->inbox.empty()) return nullptr;
    return impl_->inbox.data();
#else
    return nullptr;
#endif
}

uint64_t RemoteStage::load_local_doorbell() {
#if defined(STRATA_REMOTE_STAGE_VERBS)
    if (!impl_->opened || config_.role != RemoteStageRole::Receiver || impl_->bell.size() < 8) return 0;
    return __atomic_load_n(reinterpret_cast<const uint64_t*>(impl_->bell.data()), __ATOMIC_ACQUIRE);
#else
    return 0;
#endif
}

bool RemoteStage::ring_publish(uint32_t slot, const void* data, size_t len, uint64_t seq, std::string& error) {
    if (!is_open() || config_.role != RemoteStageRole::Receiver || slot >= config_.slots ||
        !data || len == 0 || len + 8 > config_.slot_bytes) {
        error = "remote stage: ring publish arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)slot; (void)data; (void)len; (void)seq; error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_;
    uint8_t* base = x.ring.data() + static_cast<size_t>(slot) * config_.slot_bytes;
    std::memcpy(base + 8, data, len);
    __atomic_store_n(reinterpret_cast<uint64_t*>(base), seq, __ATOMIC_RELEASE);
    return true;
#endif
}

bool RemoteStage::expose_region(const void* p, size_t bytes, bool remote_read, bool remote_write,
                                uint32_t& rkey, std::string& error) {
    if (!is_open() || !p || bytes == 0) {
        error = "remote stage: expose arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)p; (void)bytes; (void)remote_read; (void)remote_write; (void)rkey;
    error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    int access = IBV_ACCESS_LOCAL_WRITE;
    if (remote_read) access |= IBV_ACCESS_REMOTE_READ;
    if (remote_write) access |= IBV_ACCESS_REMOTE_WRITE;
    ibv_mr* mr = ibv_reg_mr(x.pd, const_cast<void*>(p), bytes, access);
    if (!mr) { error = "remote stage: region registration failed (check RLIMIT_MEMLOCK)"; return false; }
    x.extra.push_back({static_cast<const uint8_t*>(p), bytes, mr});
    rkey = mr->rkey;
    return true;
#endif
}

#if defined(STRATA_REMOTE_STAGE_VERBS)
namespace {
// The MR covering [p, p+len): the sender work buffer or any expose_region block.  A WR's local side must be
// registered, and failing closed here is what keeps a stray pointer off the wire.
ibv_mr* find_local_mr(RemoteStage::Impl& x, const void* p, size_t len) {
    const uint8_t* q = static_cast<const uint8_t*>(p);
    auto in = [q, len](const void* base, size_t bytes, ibv_mr* mr) -> ibv_mr* {
        const uint8_t* b = static_cast<const uint8_t*>(base);
        return mr && bytes > 0 && q >= b && q + len <= b + bytes ? mr : nullptr;
    };
    if (!x.snd.empty()) if (ibv_mr* m = in(x.snd.data(), x.snd.size(), x.snd_mr)) return m;
    for (const auto& e : x.extra) if (ibv_mr* m = in(e.base, e.bytes, e.mr)) return m;
    return nullptr;
}
} // namespace
#endif

bool RemoteStage::rdma_read_remote(uint64_t remote_addr, uint32_t rkey, void* dst, size_t len,
                                   std::string& error) {
    if (!is_open() || !dst || len == 0 || remote_addr == 0 || rkey == 0) {
        error = "remote stage: remote read arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)remote_addr; (void)rkey; (void)dst; (void)len;
    error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    ibv_mr* m = find_local_mr(x, dst, len);
    if (!m) { error = "remote stage: read destination is not a registered region"; return false; }
    ibv_sge sge{}; sge.addr = reinterpret_cast<uint64_t>(dst); sge.length = static_cast<uint32_t>(len);
    sge.lkey = m->lkey;
    ibv_send_wr wr{}; wr.wr_id = 7; wr.opcode = IBV_WR_RDMA_READ; wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr; wr.wr.rdma.rkey = rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(x.qp, &wr, &bad) != 0) { error = "remote stage: remote read post failed"; return false; }
    if (!poll_wc(x, 7, config_.timeout_ms, error)) return false;
    x.received += len; return true;
#endif
}

bool RemoteStage::rdma_read_rows(uint64_t remote_base, uint32_t rkey, const uint32_t* rows, size_t n,
                                 uint32_t row_bytes, void* dst, std::string& error) {
    if (!is_open() || !rows || !dst || n == 0 || remote_base == 0 || rkey == 0 || row_bytes == 0) {
        error = "remote stage: remote row read arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)remote_base; (void)rkey; (void)rows; (void)n; (void)row_bytes; (void)dst;
    error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    constexpr size_t CHAIN = 48;   // QP cap is max_send_wr=64; leave headroom, same as rdma_write_scatter
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < n) {
        const size_t batch = n - done < CHAIN ? n - done : CHAIN;
        ibv_sge sge[CHAIN]{};
        ibv_send_wr wr[CHAIN]{};
        for (size_t i = 0; i < batch; ++i) {
            void* d = out + (done + i) * (size_t) row_bytes;
            ibv_mr* m = find_local_mr(x, d, row_bytes);
            if (!m) { error = "remote stage: row read destination is not a registered region"; return false; }
            sge[i].addr = reinterpret_cast<uint64_t>(d);
            sge[i].length = row_bytes;
            sge[i].lkey = m->lkey;
            wr[i].wr_id = 8;
            wr[i].opcode = IBV_WR_RDMA_READ;
            wr[i].num_sge = 1;
            wr[i].sg_list = &sge[i];
            wr[i].wr.rdma.remote_addr = remote_base + (uint64_t) rows[done + i] * (uint64_t) row_bytes;
            wr[i].wr.rdma.rkey = rkey;
            if (i + 1 == batch) wr[i].send_flags = IBV_SEND_SIGNALED;
            else wr[i].next = &wr[i + 1];
        }
        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(x.qp, &wr[0], &bad) != 0) {
            error = "remote stage: row read post failed"; return false;
        }
        if (!poll_wc(x, 8, config_.timeout_ms, error)) return false;
        x.received += batch * (size_t) row_bytes;
        done += batch;
    }
    return true;
#endif
}

bool RemoteStage::rdma_write_scatter(const RdmaRowSeg* segs, size_t n, uint32_t data_rkey,
                                     uint64_t flag_addr, uint32_t flag_rkey, uint32_t flag_val,
                                     std::string& error) {
    if (!is_open() || (n > 0 && !segs) || data_rkey == 0 || flag_addr == 0 || flag_rkey == 0) {
        error = "remote stage: scatter write arguments invalid"; return false;
    }
#if !defined(STRATA_REMOTE_STAGE_VERBS)
    (void)segs; (void)n; (void)data_rkey; (void)flag_addr; (void)flag_rkey; (void)flag_val;
    error = "remote stage: RDMA support is unavailable"; return false;
#else
    Impl& x = *impl_; std::lock_guard<std::mutex> lk(x.mu);
    if (!x.flag_slot_mr) {
        x.flag_slot_mr = ibv_reg_mr(x.pd, &x.flag_slot, sizeof x.flag_slot, IBV_ACCESS_LOCAL_WRITE);
        if (!x.flag_slot_mr) { error = "remote stage: flag staging registration failed"; return false; }
    }
    constexpr size_t kChunk = 48;   // QP send-queue depth is 64; a chain is one post, one completion
    size_t done = 0;
    uint64_t wid = 8;
    while (true) {
        ibv_send_wr wrs[kChunk + 1];
        ibv_sge sges[kChunk + 1];
        size_t cnt = 0;
        while (done < n && cnt < kChunk) {
            const RdmaRowSeg& s = segs[done];
            ibv_mr* m = find_local_mr(x, s.src, s.len);
            if (!m) { error = "remote stage: scatter source is not a registered region"; return false; }
            sges[cnt].addr = reinterpret_cast<uint64_t>(s.src); sges[cnt].length = s.len; sges[cnt].lkey = m->lkey;
            wrs[cnt] = {}; wrs[cnt].opcode = IBV_WR_RDMA_WRITE; wrs[cnt].sg_list = &sges[cnt]; wrs[cnt].num_sge = 1;
            wrs[cnt].wr.rdma.remote_addr = s.remote_addr; wrs[cnt].wr.rdma.rkey = data_rkey;
            if (cnt > 0) wrs[cnt - 1].next = &wrs[cnt];
            ++cnt; ++done;
        }
        const bool final_chunk = (done == n);
        if (final_chunk) {
            x.flag_slot = flag_val;
            sges[cnt].addr = reinterpret_cast<uint64_t>(&x.flag_slot); sges[cnt].length = sizeof x.flag_slot;
            sges[cnt].lkey = x.flag_slot_mr->lkey;
            wrs[cnt] = {}; wrs[cnt].wr_id = wid; wrs[cnt].opcode = IBV_WR_RDMA_WRITE;
            wrs[cnt].send_flags = IBV_SEND_SIGNALED; wrs[cnt].sg_list = &sges[cnt]; wrs[cnt].num_sge = 1;
            wrs[cnt].wr.rdma.remote_addr = flag_addr; wrs[cnt].wr.rdma.rkey = flag_rkey;
            if (cnt > 0) wrs[cnt - 1].next = &wrs[cnt];
            ++cnt;
        } else {
            wrs[cnt - 1].wr_id = wid; wrs[cnt - 1].send_flags |= IBV_SEND_SIGNALED;
        }
        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(x.qp, &wrs[0], &bad) != 0) { error = "remote stage: scatter post failed"; return false; }
        if (!poll_wc(x, wid, config_.timeout_ms, error)) return false;
        ++wid;
        if (final_chunk) break;
    }
    for (size_t i = 0; i < n; ++i) x.sent += segs[i].len;
    return true;
#endif
}

bool RemoteStage::selftest(std::string& error) {
    std::array<uint8_t,257> p{}; for(size_t i=0;i<p.size();++i)p[i]=static_cast<uint8_t>(i*37u); WireMessage m{}; m.magic=kMagic; m.version=kVersion; m.type=kWindow; m.sequence=7; m.tokens=3; m.hidden=17; m.payload_bytes=static_cast<uint32_t>(p.size()); m.payload_crc=crc32(p.data(),p.size());
    if (!validate_message(m, kWindow, 1024, error) || m.payload_crc != crc32(p.data(), p.size())) return false;
    m.payload_bytes = 2048;
    if (validate_message(m, kWindow, 1024, error)) {
        error = "remote stage: bounds selftest failed";
        return false;
    }
    error.clear();
    return true;
}

} // namespace strata::core
