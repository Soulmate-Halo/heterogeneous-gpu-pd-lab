// src/core/cold_expert_host.cpp - P1②: the in-process cold-expert doorbell.  Read the header first.
//
// The serve loop below is tools/cold_expert_worker.cpp's run_serve with exactly three differences:
//   1. it runs on a thread inside the engine (stop flag + join instead of process lifetime);
//   2. there is no arena of its own and no PLE service - compute reads the engine's resident marlin plane;
//   3. the grouped tile kernel (moe_grouped_nvfp4 / moe_grouped_s2) is replaced by the marlin batch entry
//      moe_prefill_grouped_nvfp4, whose h_slot/h_dst/h_tok contract the same expert grouping feeds.
// Everything byte-visible to the peer - bind validation, the W1 channel-0 header, flag values, the write
// chains, the legacy RXE1 ring - is the worker's code, verbatim.  NVFP4 only: the whole point is sharing
// the NVFP4 marlin plane, and a Q2_0 peer is refused at the bind (expert_format), same as the worker.

#include "strata/core/cold_expert_host.hpp"

#include <atomic>

#if defined(STRATA_ENABLE_NVFP4)

#include "strata/core/rdma_expert_tier.hpp"
#include "strata/core/remote_stage.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/nvfp4_cold.hpp"
#include "strata/kernels/nvfp4_marlin_expert.hpp"

#include <cuda_runtime.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

namespace strata::core {
namespace {

constexpr int64_t H = RDMAExpertTier::kHidden;   // 2560
constexpr int CAP_TOK = 2048;                    // push channel 1 serves a whole prefill chunk per window
constexpr int CAP_ENTRIES = CAP_TOK * 10;        // worst case: every routed entry of the chunk is a miss

bool cuda_ok(cudaError_t st, const char* what, std::string& err) {
    if (st == cudaSuccess) return true;
    err = std::string("cuda: ") + what + ": " + cudaGetErrorString(st);
    return false;
}

[[noreturn]] void die(const char* what, const std::string& err) {
    std::fprintf(stderr, "cold-host: %s: %s - refusing, fail-closed\n", what, err.c_str());
    std::exit(1);
}

// Wakes a Receiver blocked in its accept poll (stop() path): the bogus connection makes accept return,
// the handshake read then fails fast on EOF, open() returns false and the loop sees the stop flag.
void kick_accept(int port) {
    const int s = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (s < 0) return;
    sockaddr_in6 sa{};
    sa.sin6_family = AF_INET6;
    sa.sin6_port = htons((uint16_t) port);
    sa.sin6_addr = in6addr_loopback;
    ::connect(s, reinterpret_cast<sockaddr*>(&sa), sizeof sa);
    ::close(s);
}

// The device side of one compute stream: activation in, marlin scratch, rows out.  Everything is carved
// once at start() and reused per window (the marlin batch entry re-initialises what it needs per call).
struct Dev {
    cudaStream_t stream = nullptr;
    float* d_x = nullptr;          // CAP_TOK x H f32 (the q8 wire dequant lands here too)
    uint8_t* d_q8 = nullptr;       // CAP_TOK x (H/32) 36 B blocks
    float* d_scales = nullptr;     // CAP_TOK x (H/32) f32
    float* d_out = nullptr;        // CAP_ENTRIES x H f32, entry order
    uint16_t* d_out16 = nullptr;   // CAP_ENTRIES x H f16 (channel-1 f16 wire)
    void* scratch = nullptr;       // moe_prefill_nvfp4_scratch_bytes
    bool open(std::string& err) {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream", err) ||
            !cuda_ok(cudaMalloc((void**) &d_x, (size_t) CAP_TOK * H * sizeof(float)), "alloc x", err) ||
            !cuda_ok(cudaMalloc((void**) &d_q8, (size_t) CAP_TOK * (H / 32) * 36), "alloc q8", err) ||
            !cuda_ok(cudaMalloc((void**) &d_scales, (size_t) CAP_TOK * (H / 32) * sizeof(float)),
                     "alloc scales", err) ||
            !cuda_ok(cudaMalloc((void**) &d_out, (size_t) CAP_ENTRIES * H * sizeof(float)), "alloc out",
                     err) ||
            !cuda_ok(cudaMalloc((void**) &d_out16, (size_t) CAP_ENTRIES * H * sizeof(uint16_t)),
                     "alloc out16", err) ||
            !cuda_ok(cudaMalloc(&scratch, kernels::moe_prefill_nvfp4_scratch_bytes(sms)),
                     "alloc marlin scratch", err))
            return false;
        return true;
    }
};

// One window's cold entries on the shared plane.  Same grouping as the worker's compute_request (experts
// in first-appearance order, contiguous spans, h_dst = the wire entry index so rows come back in request
// order), minus the WGMAX sub-split - the marlin batch path pads each expert's span to its own tile
// internally.  Calls over NV4_PREFILL_ROWS entries are chunked; every chunk keeps the grouping because the
// full array is grouped.
bool compute_marlin(const kernels::Nvfp4HotArena& arena, const int32_t* slot_of, int64_t n_expert, Dev& d,
                    int32_t layer, int32_t n_tok, int32_t k, const float* x, const uint8_t* x_q8,
                    const int32_t* entries, int32_t n_entries, float* out_host, uint16_t* out16,
                    std::string& err) {
    if (n_tok < 1 || n_tok > CAP_TOK || k < 1 || k > 32 || n_entries < 1 || n_entries > CAP_ENTRIES ||
        n_entries > n_tok * k) {
        err = "cold-host: request geometry out of bounds";
        return false;
    }
    std::vector<int32_t> gmap((size_t) n_expert, -1), gexp, group_of((size_t) n_entries);
    for (int32_t j = 0; j < n_entries; ++j) {
        const int32_t row = entries[2 * j], e = entries[2 * j + 1];
        if (row < 0 || row >= n_tok * k || e < 0 || e >= n_expert) {
            err = "cold-host: request entry out of range";
            return false;
        }
        if (gmap[(size_t) e] < 0) {
            gmap[(size_t) e] = (int32_t) gexp.size();
            gexp.push_back(e);
        }
        group_of[(size_t) j] = gmap[(size_t) e];
    }
    std::vector<int32_t> ecnt(gexp.size(), 0), h_dst((size_t) n_entries), h_tok((size_t) n_entries),
        h_slot((size_t) n_entries);
    for (int32_t j = 0; j < n_entries; ++j) ++ecnt[(size_t) group_of[(size_t) j]];
    std::vector<int32_t> estart(gexp.size() + 1, 0);
    for (size_t q = 0; q < gexp.size(); ++q) estart[q + 1] = estart[q] + ecnt[q];
    std::vector<int32_t> gcur(estart.begin(), estart.end() - 1);
    for (int32_t j = 0; j < n_entries; ++j) {
        const int32_t p = gcur[(size_t) group_of[(size_t) j]]++;
        const int32_t slot = slot_of[(int64_t) layer * n_expert + entries[2 * j + 1]];
        if (slot < 0 || slot >= arena.n_slots) {
            err = "cold-host: entry maps outside the resident plane (fill bug)";
            return false;
        }
        h_slot[(size_t) p] = slot;
        h_dst[(size_t) p] = j;                    // entry order IS the wire order
        h_tok[(size_t) p] = entries[2 * j] / k;
    }
    if (x_q8 != nullptr) {
        // The slim wire's q8 codes dequantize through the fp32 scales - x[i] = qs[i] * scales[i/32] is
        // exactly the activation value the fp32 wire carries, so both wire formats feed the same numbers.
        const size_t qb = (size_t) n_tok * (size_t) (H / 32) * 36;
        if (!cuda_ok(cudaMemcpyAsync(d.d_q8, x_q8, qb, cudaMemcpyHostToDevice, d.stream), "copy q8", err) ||
            !cuda_ok(cudaMemcpyAsync(d.d_scales, x_q8 + qb, (size_t) n_tok * (size_t) (H / 32) * sizeof(float),
                                     cudaMemcpyHostToDevice, d.stream), "copy scales", err))
            return false;
        kernels::nvfp4_q8_act_dequant(d.d_q8, d.d_scales, d.d_x, (int64_t) n_tok * H, d.stream);
    } else if (!cuda_ok(cudaMemcpyAsync(d.d_x, x, (size_t) n_tok * H * sizeof(float),
                                        cudaMemcpyHostToDevice, d.stream), "copy x", err)) {
        return false;
    }
    for (int32_t base = 0; base < n_entries; base += kernels::NV4_PREFILL_ROWS) {
        const int32_t n = std::min<int32_t>(kernels::NV4_PREFILL_ROWS, n_entries - base);
        kernels::moe_prefill_grouped_nvfp4(arena, h_slot.data() + base, h_dst.data() + base,
                                           h_tok.data() + base, n, d.d_x, d.scratch, d.d_out, d.stream);
    }
    if (out16 != nullptr)
        kernels::f32_to_f16_bulk(d.d_out, d.d_out16, (int64_t) n_entries * H, d.stream);
    if (!cuda_ok(out16 != nullptr
                     ? cudaMemcpyAsync(out16, d.d_out16, (size_t) n_entries * H * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, d.stream)
                     : cudaMemcpyAsync(out_host, d.d_out, (size_t) n_entries * H * sizeof(float),
                                       cudaMemcpyDeviceToHost, d.stream), "copy out", err) ||
        !cuda_ok(cudaStreamSynchronize(d.stream), "sync", err))
        return false;
    return true;
}

// The channel-0 (decode, one token) compute fast path: the same grouping and the same marlin sequence as
// compute_marlin, but the routing rows live in pinned staging and the GPU sequence runs as a captured
// CUDA graph - the 5-7 kernel launches + 7 copies + TVM FFI of every push collapse into one
// cudaGraphLaunch (measured 120 us of CPU-side compute time per push, the GEMMs themselves are ~30 us).
// Numerics are untouched: identical kernels, arguments and order, only the H2D source addresses moved
// into pinned memory.  The graph is keyed by (ne, nblk) because the expansion's copy SIZES are baked at
// capture: two pushes with the same ne can still differ in block count, and replaying a graph against a
// different nblk would copy the wrong span lengths.  Any staging/capture/launch failure sticks the host
// on the direct compute_marlin path - a graph problem can never produce a wrong row.
bool compute_ch0(const kernels::Nvfp4HotArena& arena, const int32_t* slot_of, int64_t n_expert, Dev& d,
                 int32_t layer, int32_t k, const float* x, const int32_t* entries, int32_t ne,
                 float* out_host, int32_t* p_slot, int32_t* p_dstg, int32_t* p_tokg,
                 std::map<int32_t, cudaGraphExec_t>& graphs, bool& graphs_ok, std::string& err) {
    // Grouping identical to compute_marlin's (expert first-appearance spans, dst = the wire entry index),
    // specialised for n_tok == 1: tok[] is always 0 (the caller validated 1 <= ne <= k <= 32).
    static std::vector<int32_t> gmap;   // the doorbell thread is the only caller
    if ((int64_t) gmap.size() != n_expert)
        gmap.assign((size_t) n_expert, -1);
    else
        std::fill(gmap.begin(), gmap.end(), -1);
    int32_t ecnt[32], estart[33], gcur[32], group_of[32];
    int32_t ng = 0;
    for (int32_t j = 0; j < ne; ++j) {
        const int32_t e = entries[2 * j + 1];
        if (gmap[(size_t) e] < 0) {
            gmap[(size_t) e] = ng;
            ++ng;
        }
        group_of[j] = gmap[(size_t) e];
        ecnt[group_of[j]] = 0;
    }
    for (int32_t j = 0; j < ne; ++j) ++ecnt[group_of[j]];
    estart[0] = 0;
    for (int32_t q = 0; q < ng; ++q) estart[q + 1] = estart[q] + ecnt[q];
    for (int32_t q = 0; q < ng; ++q) gcur[q] = estart[q];
    for (int32_t j = 0; j < ne; ++j) {
        const int32_t p = gcur[group_of[j]]++;
        const int32_t slot = slot_of[(int64_t) layer * n_expert + entries[2 * j + 1]];
        if (slot < 0 || slot >= arena.n_slots) {
            err = "cold-host: entry maps outside the resident plane (fill bug)";
            return false;
        }
        p_slot[p] = slot;
        p_dstg[p] = j;   // entry order IS the wire order
        p_tokg[p] = 0;
    }

    const int32_t post = kernels::nvfp4_push_expand(p_slot, p_dstg, p_tokg, ne);
    if (post >= 0 && graphs_ok) {
        const int32_t key = ne * 64 + post / 16;   // NV4_BLOCKM = 16; nblk <= ne < 64
        const auto it = graphs.find(key);
        if (it != graphs.end()) {
            if (cudaGraphLaunch(it->second, d.stream) == cudaSuccess &&
                cuda_ok(cudaStreamSynchronize(d.stream), "graph sync", err))
                return true;
            std::fprintf(stderr, "cold-host: decode graph launch failed - direct path from here on\n");
            graphs_ok = false;
            (void) cudaGetLastError();
        } else {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            cudaError_t st = cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal);
            if (st == cudaSuccess) {
                (void) cudaMemcpyAsync(d.d_x, x, (size_t) H * sizeof(float), cudaMemcpyHostToDevice,
                                       d.stream);
                kernels::nvfp4_push_issue_staged(arena, ne, d.d_x, d.scratch, d.d_out, d.stream);
                (void) cudaMemcpyAsync(out_host, d.d_out, (size_t) ne * H * sizeof(float),
                                       cudaMemcpyDeviceToHost, d.stream);
                st = cudaStreamEndCapture(d.stream, &graph);
            }
            if (st == cudaSuccess) st = cudaGraphInstantiate(&exec, graph, 0);
            if (graph != nullptr) (void) cudaGraphDestroy(graph);
            if (st == cudaSuccess) st = cudaGraphLaunch(exec, d.stream);
            if (st == cudaSuccess && cuda_ok(cudaStreamSynchronize(d.stream), "graph sync", err)) {
                graphs.emplace(key, exec);
                return true;
            }
            std::fprintf(stderr,
                         "cold-host: decode graph capture/instantiate failed (%s) - direct path from "
                         "here on\n",
                         cudaGetErrorString(st));
            if (exec != nullptr) (void) cudaGraphExecDestroy(exec);
            graphs_ok = false;
            (void) cudaGetLastError();
        }
    } else if (graphs_ok) {
        std::fprintf(stderr,
                     "cold-host: decode push pinned staging unavailable - direct path from here on\n");
        graphs_ok = false;
    }
    // Direct fallback, the pre-graph code path byte for byte: a failed capture recorded nothing and a
    // failed launch executed nothing, so re-running the sequence cannot double-apply anything.
    return compute_marlin(arena, slot_of, n_expert, d, layer, 1, k, x, nullptr, entries, ne, out_host,
                          nullptr, err);
}

}  // namespace

namespace {
std::atomic<int>* g_cold_hold = nullptr;
std::atomic<int>* g_cold_inflight = nullptr;

// A doorbell window holds this across compute (including a captured ch0 graph replay). A planar
// slot swap sets the hold, waits until no window is inside, then publishes the new slot.
struct ColdInflight {
    std::atomic<int>* n = nullptr;
    ColdInflight(std::atomic<int>& hold, std::atomic<int>& inflight) : n(&inflight) {
        for (;;) {
            while (hold.load(std::memory_order_acquire) != 0)
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            inflight.fetch_add(1, std::memory_order_acq_rel);
            if (hold.load(std::memory_order_acquire) == 0) return;
            inflight.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    ~ColdInflight() { if (n != nullptr) n->fetch_sub(1, std::memory_order_acq_rel); }
    ColdInflight(const ColdInflight&) = delete;
    ColdInflight& operator=(const ColdInflight&) = delete;
};
}  // namespace

struct ColdExpertHost::Impl {
    const kernels::Nvfp4HotArena* arena = nullptr;   // not owned
    const int32_t* slot_of = nullptr;                // not owned
    int n_layers = 0, n_expert = 0;
    ColdExpertHostConfig cfg;
    std::unique_ptr<RemoteStage> stage;
    Dev dev;
    // push-channel I/O staging (pinned; registered once the stage is open)
    float* p_out = nullptr;
    float* p_x = nullptr;
    uint16_t* p_out16 = nullptr;
    RDMAExpertTier::PushReqHead* p_req = nullptr;
    int32_t* p_ent = nullptr;
    uint32_t* p_hdr = nullptr;
    // decode push (channel 0): pinned routing rows + the per-(ne, nblk) CUDA graph cache of compute_ch0
    int32_t* p_slot = nullptr;
    int32_t* p_dstg = nullptr;
    int32_t* p_tokg = nullptr;
    std::map<int32_t, cudaGraphExec_t> ch0_graphs;
    bool ch0_graphs_ok = true;   // sticky: one failure keeps the direct compute_marlin path for good
    std::thread th;
    std::atomic<bool> stop{false};
    std::atomic<bool> accepted{false};   // the accept returned (kick_accept no longer needed/useful)
    std::atomic<uint64_t> served{0};
    std::atomic<int> swap_hold{0};          // non-zero: a planar slot swap owns the resident plane
    std::atomic<int> compute_inflight{0};   // doorbell windows inside compute / ch0 graph replay

    bool serve_session();   // one peer session; true = transport loss, re-listen
    void loop();
};

// One peer session: the doorbell loop below runs until stop() or a transport failure.  Transport loss
// (the peer's engine exits, the QP breaks) returns true so the caller re-listens - unlike the standalone
// worker, which exits on peer loss, this host is a daemon thread of a resident engine and MUST stay up
// (P2: the same process serves the prefill leg).  Protocol violations and compute failures still die(),
// fail-closed: a peer that mis-speaks the wire format must not get a plausible wrong row.
bool ColdExpertHost::Impl::serve_session() {
    std::string err;
    uint32_t io_rkey = 0;
    if (!stage->expose_region(p_out, (size_t) CAP_ENTRIES * H * sizeof(float), false, false, io_rkey, err) ||
        !stage->expose_region(p_x, (size_t) CAP_TOK * H * sizeof(float), false, false, io_rkey, err) ||
        !stage->expose_region(p_out16, (size_t) CAP_ENTRIES * H * sizeof(uint16_t), false, false, io_rkey,
                              err) ||
        !stage->expose_region(p_req, sizeof(RDMAExpertTier::PushReqHead), false, false, io_rkey, err) ||
        !stage->expose_region(p_ent, (size_t) CAP_ENTRIES * 2 * sizeof(int32_t), false, false, io_rkey,
                              err) ||
        !stage->expose_region(p_hdr, (size_t) RDMAExpertTier::kPushYCDataOff, false, false, io_rkey, err))
        die("push buffer registration", err);
    std::fprintf(stderr,
                 "cold-host: READY rdma=true slots=%d slot_bytes=%lld - waiting for doorbells "
                 "(in-process, marlin on the resident plane)\n",
                 cfg.slots, (long long) cfg.slot_bytes);

    std::vector<float> h_out((size_t) CAP_ENTRIES * H);
    uint64_t last = 0, last_push = 0;
    uint64_t served_pull = 0;
    bool push_mode = false;
    RDMAExpertTier::PushBind bind{};
    RDMAExpertTier::PushBind2 bind2{};
    bool have_bind2 = false;
    RDMAExpertTier::PushBind3 bind3{};
    bool have_bind3 = false;
    RDMAExpertTier::PushBind4 bind4{};
    bool have_bind4 = false;
    // STRATA_PF_PROBE=1: per-push-window service breakdown (idle / entries+activation read / compute /
    // result write), printed every 512 windows and at exit - the worker's probe, minus the kernel split
    // (the marlin entry has no probe argument).
    const bool probe = std::getenv("STRATA_PF_PROBE") != nullptr;
    std::vector<double> pb_idle, pb_read, pb_compute, pb_write;
    auto pb_prev = std::chrono::steady_clock::now();
    auto pb_line = [](const char* name, std::vector<double> v, char* b, size_t bn) {
        std::sort(v.begin(), v.end());
        double s = 0.0;
        for (double d : v) s += d;
        std::snprintf(b, bn, "%s mean %.2f p50 %.2f p99 %.2f ms", name, s / (double) v.size(),
                      v[v.size() / 2], v[(size_t) ((double) (v.size() - 1) * 0.99)]);
        return b;
    };
    auto pb_dump = [&]() {
        if (!probe || pb_read.empty()) return;
        char b1[96], b2[96], b3[96], b4[96];
        std::fprintf(stderr, "cold-host probe: %zu push windows | %s | %s | %s | %s\n", pb_read.size(),
                     pb_line("idle", pb_idle, b1, sizeof b1), pb_line("read", pb_read, b2, sizeof b2),
                     pb_line("compute", pb_compute, b3, sizeof b3), pb_line("write", pb_write, b4, sizeof b4));
    };
    while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t seq = stage->load_local_doorbell();
        if (seq > last) {
            const uint8_t* inbox = stage->local_inbox();
            if (!inbox) die("no inbox", "");
            uint32_t magic = 0;
            std::memcpy(&magic, inbox, sizeof magic);
            if (magic == RDMAExpertTier::kMagicBind) {
                // Push mode: the engine peer registered its request block, activation, y_miss and doorbell
                // flag; from here this host drives BOTH directions and the peer's host leaves the wire.
                std::memcpy(&bind, inbox, sizeof bind);
                if (bind.version != 3 || bind.n_embd != (uint32_t) H || bind.k_cap == 0 ||
                    bind.n_layers != (uint32_t) n_layers || !bind.req_addr || !bind.req_rkey ||
                    !bind.x_addr || !bind.x_rkey || !bind.y_addr || !bind.y_rkey || !bind.flag_addr ||
                    !bind.flag_rkey || !bind.yc_addr || !bind.yc_rkey || bind.yc_rows < bind.k_cap ||
                    bind.expert_format != 1u) {
                    std::fprintf(stderr, "cold-host: bad push bind record (version, geometry or expert "
                                         "format mismatch: bind=%u, host=1) - refusing, fail-closed\n",
                                 bind.expert_format);
                    std::exit(1);
                }
                push_mode = true;
                last_push = 0;
                last = seq;
                std::fprintf(stderr, "cold-host: push mode bound (y_miss %u rows, %u layers) - the peer's "
                                     "host verbs are off the wire\n", bind.k_cap, bind.n_layers);
                continue;
            }
            if (magic == RDMAExpertTier::kMagicBind2) {
                std::memcpy(&bind2, inbox, sizeof bind2);
                const uint32_t xrb = bind2.x2_dtype == RDMAExpertTier::kPushXq8
                                         ? (uint32_t) ((H / 32) * (36 + sizeof(float)))
                                         : bind2.x2_dtype == RDMAExpertTier::kPushXf32
                                               ? (uint32_t) (H * sizeof(float)) : 0;
                const uint32_t yrb = bind2.y2_dtype == RDMAExpertTier::kPushYf16
                                         ? (uint32_t) (H * sizeof(uint16_t))
                                         : bind2.y2_dtype == RDMAExpertTier::kPushYf32
                                               ? (uint32_t) (H * sizeof(float)) : 0;
                if (bind2.version != 2 || bind2.t_cap == 0 || bind2.y2_rows == 0 || !bind2.x2_addr ||
                    !bind2.x2_rkey || !bind2.y2_addr || !bind2.y2_rkey || xrb == 0 || yrb == 0 ||
                    bind2.x2_row_bytes != xrb || bind2.y2_row_bytes != yrb)
                    die("bad push channel-1 bind record", "");
                have_bind2 = true;
                last = seq;
                std::fprintf(stderr, "cold-host: push channel 1 bound (x %u tokens %s, y %u rows %s)\n",
                             bind2.t_cap, bind2.x2_dtype == RDMAExpertTier::kPushXq8 ? "q8_0" : "f32",
                             bind2.y2_rows, bind2.y2_dtype == RDMAExpertTier::kPushYf16 ? "f16" : "f32");
                continue;
            }
            if (magic == RDMAExpertTier::kMagicBind3) {
                std::memcpy(&bind3, inbox, sizeof bind3);
                if (bind3.version != 1 || bind3.t3_cap == 0 || bind3.y3_rows == 0 || !bind3.x3_addr ||
                    !bind3.x3_rkey || !bind3.y3_addr || !bind3.y3_rkey || !bind3.flag3_addr ||
                    !bind3.flag3_rkey)
                    die("bad push channel-2 bind record", "");
                have_bind3 = true;
                last = seq;
                std::fprintf(stderr, "cold-host: push channel 2 bound (x %u tokens, y %u rows, own flag)\n",
                             bind3.t3_cap, bind3.y3_rows);
                continue;
            }
            if (magic == RDMAExpertTier::kMagicBind4) {
                // Batch ladder phase 2: the peer's SECOND decode session registered its activation, its
                // contiguous result block and its own doorbell flag.  Channel-3 requests get exactly the
                // channel-0 treatment, only the destination regions differ.
                std::memcpy(&bind4, inbox, sizeof bind4);
                if (bind4.version != 1 || bind4.k_cap == 0 || bind4.n_layers != (uint32_t) n_layers ||
                    !bind4.x_addr || !bind4.x_rkey || !bind4.yc_addr || !bind4.yc_rkey ||
                    bind4.yc_rows < bind4.k_cap || !bind4.flag_addr || !bind4.flag_rkey ||
                    bind4.expert_format != 1u)
                    die("bad push channel-3 bind record", "");
                have_bind4 = true;
                last = seq;
                std::fprintf(stderr, "cold-host: push channel 3 bound (second decode session, %u rows)\n",
                             bind4.k_cap);
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            RDMAExpertTier::ReqHeader h;
            std::memcpy(&h, inbox, sizeof h);
            if (h.magic != RDMAExpertTier::kMagic || h.version != RDMAExpertTier::kVersion || h.layer < 0 ||
                h.layer >= n_layers || h.n_tok < 1 || h.n_tok > CAP_TOK || h.k < 1 || h.k > 32 ||
                h.n_entries < 1 || h.n_entries > h.n_tok * h.k)
                die("bad request header", std::to_string(h.magic));
            const size_t x_bytes = (size_t) h.n_tok * H * sizeof(float);
            const size_t ent_bytes = (size_t) h.n_entries * 2 * sizeof(int32_t);
            if (sizeof h + x_bytes + ent_bytes > (size_t) cfg.slot_bytes)
                die("request exceeds the inbox", "");
            const float* x = reinterpret_cast<const float*>(inbox + sizeof h);
            const int32_t* entries = reinterpret_cast<const int32_t*>(inbox + sizeof h + x_bytes);
            {
                ColdInflight gate(swap_hold, compute_inflight);
                if (!compute_marlin(*arena, slot_of, n_expert, dev, h.layer, h.n_tok, h.k, x, nullptr, entries,
                                    h.n_entries, h_out.data(), nullptr, err))
                    die("compute failed", err);
            }
            const uint32_t slot = (uint32_t) ((seq - 1) % (uint64_t) cfg.slots);
            if (!stage->ring_publish(slot, h_out.data(), (size_t) h.n_entries * H * sizeof(float), seq,
                                     err)) {
                std::fprintf(stderr, "cold-host: publish failed: %s - session lost\n", err.c_str());
                return true;
            }
            last = seq;
            ++served_pull;
            served.fetch_add(1);
            if (cfg.verbose || served_pull <= 3 || (served_pull % 480) == 0) {
                const double us = std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count();
                std::fprintf(stderr, "cold-host: seq=%" PRIu64 " layer=%d tok=%d entries=%d %.0f us\n",
                             seq, h.layer, h.n_tok, h.n_entries, us);
            }
            continue;
        }
        if (push_mode) {
            // Poll only the 32-byte head of the peer's request block: one small one-sided read per
            // iteration, no peer involvement.  The entries follow in a second read once the seq moved.
            if (!stage->rdma_read_remote(bind.req_addr, bind.req_rkey, p_req,
                                         sizeof(RDMAExpertTier::PushReqHead), err)) {
                std::fprintf(stderr, "cold-host: push request read failed: %s - session lost\n",
                             err.c_str());
                return true;
            }
            if (p_req->seq == last_push) { std::this_thread::yield(); continue; }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t rseq = p_req->seq;
            const int32_t layer = p_req->layer;
            const int32_t n_tok = p_req->n_tok;
            const int32_t k = p_req->k;
            const int32_t ne = p_req->n_entries;
            const int32_t channel = p_req->channel;
            const int32_t x_off = p_req->x_off;
            const int32_t y_off = p_req->y_off;
            if (layer < 0 || layer >= n_layers || k < 1 || k > 32 || ne < 0 || ne > n_tok * k ||
                ne > CAP_ENTRIES)
                die("bad push request header", "");
            uint64_t x_addr = 0, y_addr = 0, flag_addr = bind.flag_addr;
            uint32_t x_rkey = 0, y_rkey = 0, flag_rkey = bind.flag_rkey, flag_val = 0;
            if (channel == 0) {
                // decode: one token, rows written CONTIGUOUS in entry order into the v3 y_contig block
                // (the peer's graph scatters them on the device), flag = layer + 1
                if (n_tok != 1 || (uint32_t) k > bind.k_cap)
                    die("bad decode push request", "");
                x_addr = bind.x_addr; x_rkey = bind.x_rkey;
                y_addr = bind.yc_addr; y_rkey = bind.yc_rkey;
                flag_val = (uint32_t) (layer + 1);
            } else if (channel == 1 && have_bind2) {
                // prefill: a whole chunk, rows contiguous in entry order, flag = the request's seq
                if ((uint32_t) n_tok > bind2.t_cap || (uint32_t) ne > bind2.y2_rows)
                    die("push channel-1 request beyond the bound buffers", "");
                x_addr = bind2.x2_addr; x_rkey = bind2.x2_rkey;
                y_addr = bind2.y2_addr; y_rkey = bind2.y2_rkey;
                flag_val = (uint32_t) rseq;
            } else if (channel == 2 && have_bind3) {
                // verify: the window's segment (x3 + x_off tokens), rows contiguous in entry order at
                // y3 + y_off, the window's own flag = the request's seq.  All f32.
                if (x_off < 0 || y_off < 0 || n_tok < 1 || (uint32_t) (x_off + n_tok) > bind3.t3_cap ||
                    (uint32_t) (y_off + ne) > bind3.y3_rows)
                    die("push channel-2 request beyond the bound buffers", "");
                x_addr = bind3.x3_addr + (uint64_t) x_off * (uint64_t) (H * sizeof(float));
                x_rkey = bind3.x3_rkey;
                y_addr = bind3.y3_addr + (uint64_t) y_off * (uint64_t) (H * sizeof(float));
                y_rkey = bind3.y3_rkey;
                flag_addr = bind3.flag3_addr; flag_rkey = bind3.flag3_rkey;
                flag_val = (uint32_t) rseq;
            } else if (channel == 3 && have_bind4) {
                // the second decode session: channel 0's shape into the RXE5 regions, flag = layer + 1
                if (n_tok != 1 || (uint32_t) k > bind4.k_cap)
                    die("bad decode push request (channel 3)", "");
                x_addr = bind4.x_addr; x_rkey = bind4.x_rkey;
                y_addr = bind4.yc_addr; y_rkey = bind4.yc_rkey;
                flag_addr = bind4.flag_addr; flag_rkey = bind4.flag_rkey;
                flag_val = (uint32_t) (layer + 1);
            } else {
                die("push request on an unbound channel", "");
            }
            const bool x_q8 = channel == 1 && bind2.x2_dtype == RDMAExpertTier::kPushXq8;
            const bool y_f16 = channel == 1 && bind2.y2_dtype == RDMAExpertTier::kPushYf16;
            RemoteStage::RdmaRowSeg segs[80];   // a verify window's worst case is kVerifyMaxT * k = 80 rows
            RemoteStage::RdmaRowSeg one{y_f16 ? (const void*) p_out16 : (const void*) p_out, y_addr,
                                        (uint32_t) ((size_t) ne * (channel == 1 ? bind2.y2_row_bytes
                                                                                : (uint32_t) (H * sizeof(float))))};
            std::chrono::steady_clock::time_point t1 = t0, t2 = t0;
            if (ne > 0) {
                if (!stage->rdma_read_remote(bind.req_addr + offsetof(RDMAExpertTier::PushReq, entries),
                                             bind.req_rkey, p_ent, (size_t) ne * 2 * sizeof(int32_t), err)) {
                    std::fprintf(stderr, "cold-host: push entries read failed: %s - session lost\n",
                                 err.c_str());
                    return true;
                }
                for (int32_t j = 0; j < ne; ++j) {
                    const int32_t row = p_ent[2 * j];
                    if (row < 0 || row >= n_tok * k || p_ent[2 * j + 1] < 0 ||
                        p_ent[2 * j + 1] >= n_expert)
                        die("push request entry out of range", "");
                }
                if (!stage->rdma_read_remote(x_addr, x_rkey, p_x,
                                             (size_t) n_tok * (channel == 1 ? bind2.x2_row_bytes
                                                                            : (uint32_t) (H * sizeof(float))),
                                             err)) {
                    std::fprintf(stderr, "cold-host: push activation read failed: %s - session lost\n",
                                 err.c_str());
                    return true;
                }
                if (probe) t1 = std::chrono::steady_clock::now();
                bool computed = false;
                {
                    ColdInflight gate(swap_hold, compute_inflight);
                    computed = channel == 0 || channel == 3
                                   ? compute_ch0(*arena, slot_of, n_expert, dev, layer, k, (const float*) p_x,
                                                 p_ent, ne, p_out, p_slot, p_dstg, p_tokg, ch0_graphs,
                                                 ch0_graphs_ok, err)
                                   : compute_marlin(*arena, slot_of, n_expert, dev, layer, n_tok, k,
                                                    x_q8 ? nullptr : (const float*) p_x,
                                                    x_q8 ? (const uint8_t*) p_x : nullptr, p_ent, ne,
                                                    y_f16 ? nullptr : p_out, y_f16 ? p_out16 : nullptr, err);
                }
                if (!computed)
                    die("push compute failed", err);
                if (probe) t2 = std::chrono::steady_clock::now();
                if (channel == 0 || channel == 3) {
                    // W1: [count][pad][rows...] header first, the entry-order rows second, the flag last -
                    // two SGEs for any ne, and the RC chain keeps the order so the consumer that observes
                    // the flag observes the whole block.
                    p_hdr[0] = (uint32_t) ne;
                    p_hdr[1] = 0;
                    for (int32_t j = 0; j < ne; ++j) p_hdr[2 + j] = (uint32_t) p_ent[2 * j];
                    segs[0] = RemoteStage::RdmaRowSeg{p_hdr, y_addr,
                                                      (uint32_t) (8 + (size_t) ne * sizeof(uint32_t))};
                    segs[1] = RemoteStage::RdmaRowSeg{p_out, y_addr + RDMAExpertTier::kPushYCDataOff,
                                                      (uint32_t) ((size_t) ne * H * sizeof(float))};
                } else if (channel == 2)
                    // verify scatters inside the window's segment (y3 + y_off + row) - the dispatch expects
                    // each row AT its routing position, only channel 1's consumer reads them contiguous
                    for (int32_t j = 0; j < ne; ++j)
                        segs[j] = RemoteStage::RdmaRowSeg{
                            p_out + (size_t) j * H,
                            y_addr + (uint64_t) p_ent[2 * j] * (uint64_t) H * sizeof(float),
                            (uint32_t) ((size_t) H * sizeof(float))};
            }
            if (ne == 0 && (channel == 0 || channel == 3)) {
                // W1: an empty decode window still rewrites the header (count = 0), or the peer's scatter
                // would replay the previous layer's rows.
                p_hdr[0] = 0;
                p_hdr[1] = 0;
                segs[0] = RemoteStage::RdmaRowSeg{p_hdr, y_addr, 8};
            }
            // Rows first, flag last, one ordered chain: the consumer passes only once every row it will
            // read has landed.  Channels 0/3's flag is layer + 1; channels 1/2's is the request's seq.
            const RemoteStage::RdmaRowSeg* sp = channel == 1 ? &one : segs;
            const size_t ns = channel == 1 ? (ne > 0 ? 1 : 0)
                                           : ((channel == 0 || channel == 3) ? (ne > 0 ? 2 : 1) : (size_t) ne);
            if (!stage->rdma_write_scatter(sp, ns, y_rkey, flag_addr, flag_rkey, flag_val, err)) {
                std::fprintf(stderr, "cold-host: push result write failed: %s - session lost\n",
                             err.c_str());
                return true;
            }
            last_push = rseq;
            served.fetch_add(1);
            if (probe) {
                const auto t3 = std::chrono::steady_clock::now();
                pb_idle.push_back(std::chrono::duration<double, std::milli>(t0 - pb_prev).count());
                pb_read.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                pb_compute.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
                pb_write.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
                pb_prev = t3;
                if (pb_read.size() % 512 == 0) pb_dump();
            }
            if (cfg.verbose || served.load() <= 3 || (served.load() % 480) == 0) {
                const double us = std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count();
                std::fprintf(stderr,
                             "cold-host: push seq=%" PRIu64 " ch=%d layer=%d tok=%d entries=%d %.0f us\n",
                             rseq, channel, layer, n_tok, ne, us);
            }
            continue;
        }
        std::this_thread::yield();
    }
    pb_dump();
    return false;   // stop() requested
}

void ColdExpertHost::Impl::loop() {
    std::string err;
    int open_fails = 0;
    while (!stop.load(std::memory_order_relaxed)) {
        RemoteStageConfig sc;
        sc.role = RemoteStageRole::Receiver;
        sc.bind_host = cfg.bind;
        sc.port = (uint16_t) cfg.port;
        sc.device = cfg.device;
        sc.gid_index = cfg.gid;
        sc.timeout_ms = cfg.timeout_ms;
        sc.slots = (uint32_t) cfg.slots;
        sc.slot_bytes = (size_t) cfg.slot_bytes;
        sc.max_payload = (size_t) cfg.slot_bytes;
        stage.reset(new RemoteStage(sc));
        if (!stage->open(err)) {
            // stop() kicked the accept during engine shutdown, or the accept window ended with no peer:
            // either way just cycle - a late peer finds a listener, a persistent config error dies on the
            // repetition bound instead of spinning the log forever.
            stage.reset();
            if (stop.load()) return;
            if (++open_fails >= 10) die("repeated accept failures", err);
            std::fprintf(stderr, "cold-host: accept ended (%s) - re-listening\n", err.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        open_fails = 0;
        accepted.store(true);
        if (!stage->is_rdma()) die("the data plane is not RDMA", "");
        const bool lost = serve_session();
        stage.reset();
        if (stop.load()) return;
        if (lost)
            std::fprintf(stderr, "cold-host: session ended on transport loss - re-listening "
                                 "(the engine and its prefill leg stay up)\n");
    }
}

ColdExpertHost::ColdExpertHost(const kernels::Nvfp4HotArena* shared_arena, const int32_t* slot_of,
                               int n_layers, int n_expert)
    : impl_(new Impl) {
    impl_->arena = shared_arena;
    impl_->slot_of = slot_of;
    impl_->n_layers = n_layers;
    impl_->n_expert = n_expert;
}

ColdExpertHost::~ColdExpertHost() { stop(); }

bool ColdExpertHost::start(const ColdExpertHostConfig& cfg, std::string& err) {
    if (impl_->th.joinable()) { err = "cold-host: already started"; return false; }
    if (impl_->arena == nullptr || !impl_->arena->on()) {
        err = "cold-host: the shared marlin arena is not live (needs --expert-resident-all on NVFP4)";
        return false;
    }
    if (impl_->slot_of == nullptr || impl_->n_layers <= 0 || impl_->n_expert <= 0) {
        err = "cold-host: no slot map";
        return false;
    }
    // Synchronous port probe: the accept itself happens on the thread (see the header), but an old
    // strata-cold-expert-worker still holding the port is the deployment's #1 hazard and must fail the
    // engine's boot HERE, not after the 60-80 s plane fill.
    {
        const int s = ::socket(AF_INET6, SOCK_STREAM, 0);
        if (s >= 0) {
            sockaddr_in6 sa{};
            sa.sin6_family = AF_INET6;
            sa.sin6_port = htons((uint16_t) cfg.port);
            sa.sin6_addr = in6addr_any;
            if (::bind(s, reinterpret_cast<sockaddr*>(&sa), sizeof sa) != 0) {
                ::close(s);
                err = "cold-host: port " + std::to_string(cfg.port) +
                      " is already taken (an old strata-cold-expert-worker still running?)";
                return false;
            }
            ::close(s);
        }
    }
    Impl* x = impl_.get();
    if (!cuda_ok(cudaHostAlloc((void**) &x->p_out, (size_t) CAP_ENTRIES * H * sizeof(float),
                               cudaHostAllocDefault), "pin out", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_x, (size_t) CAP_TOK * H * sizeof(float),
                               cudaHostAllocDefault), "pin x", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_out16, (size_t) CAP_ENTRIES * H * sizeof(uint16_t),
                               cudaHostAllocDefault), "pin out16", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_req, sizeof(RDMAExpertTier::PushReqHead),
                               cudaHostAllocDefault), "pin req", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_ent, (size_t) CAP_ENTRIES * 2 * sizeof(int32_t),
                               cudaHostAllocDefault), "pin ent", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_hdr, (size_t) RDMAExpertTier::kPushYCDataOff,
                               cudaHostAllocDefault), "pin hdr", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_slot, 128 * sizeof(int32_t), cudaHostAllocDefault),
                 "pin slot", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_dstg, 128 * sizeof(int32_t), cudaHostAllocDefault),
                 "pin dstg", err) ||
        !cuda_ok(cudaHostAlloc((void**) &x->p_tokg, 128 * sizeof(int32_t), cudaHostAllocDefault),
                 "pin tokg", err))
        return false;
    if (!x->dev.open(err)) return false;
    x->cfg = cfg;
    g_cold_hold = &x->swap_hold;
    g_cold_inflight = &x->compute_inflight;
    x->th = std::thread([x] { x->loop(); });
    std::fprintf(stderr, "cold-host: doorbell thread up on port %d (accept runs in the background)\n",
                 cfg.port);
    return true;
}

void ColdExpertHost::stop() {
    if (!impl_ || !impl_->th.joinable()) return;
    impl_->stop.store(true);
    impl_->swap_hold.store(0, std::memory_order_release);
    // Wake a pending accept (no-op once the listener is gone or a session is live); the second kick covers
    // a listener that comes up between the first kick and the loop's stop check.
    kick_accept(impl_->cfg.port);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    kick_accept(impl_->cfg.port);
    impl_->th.join();
    g_cold_hold = nullptr;
    g_cold_inflight = nullptr;
}

uint64_t ColdExpertHost::served_windows() const { return impl_ ? impl_->served.load() : 0; }

void nvfp4_cold_swap_begin() {
    std::atomic<int>* hold = g_cold_hold;
    std::atomic<int>* n = g_cold_inflight;
    if (hold == nullptr || n == nullptr) return;
    hold->store(1, std::memory_order_release);
    while (n->load(std::memory_order_acquire) != 0)
        std::this_thread::sleep_for(std::chrono::microseconds(50));
}

void nvfp4_cold_swap_end() {
    if (std::atomic<int>* hold = g_cold_hold) hold->store(0, std::memory_order_release);
}

}  // namespace strata::core

#else  // !STRATA_ENABLE_NVFP4: the host is a marlin-plane consumer, so without the tier it is a stub that
       // refuses at start() - a Q2_0 or CPU build keeps compiling, the doorbell simply cannot be armed.

#include <cstdio>

namespace strata::core {

struct ColdExpertHost::Impl {};

ColdExpertHost::ColdExpertHost(const kernels::Nvfp4HotArena*, const int32_t*, int, int) : impl_(new Impl) {}
ColdExpertHost::~ColdExpertHost() = default;

bool ColdExpertHost::start(const ColdExpertHostConfig&, std::string& err) {
    err = "cold-host: not built with STRATA_ENABLE_NVFP4 (the in-process doorbell serves the marlin plane)";
    return false;
}

void ColdExpertHost::stop() {}
uint64_t ColdExpertHost::served_windows() const { return 0; }

}  // namespace strata::core

#endif
