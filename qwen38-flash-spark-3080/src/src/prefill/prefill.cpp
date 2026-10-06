// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/rdma_expert_tier.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/quantize_act.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_fp8.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef STRATA_PREFILL_MMQ
// A build without the llama.cpp sources (no STRATA_NATIVE_EXPERTS): no MMQ, the FP16 expert path everywhere.
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {}
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*) {}
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void iota(int32_t*, int64_t, void*) {}
}  // namespace strata::prefill::mmq
#endif

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below STREAM_ALL_MIN)
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// RING_MAX-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 512;           // the arrays; the ring itself is ring_slots()
constexpr int64_t STREAM_ALL_MIN = 2048;
double g_pinned_share = 1.0;
// The streamed ring: 384 slots when (nearly) every streamed expert is DMA'd from pinned RAM - measured on Q2_0,
// 8192-token chunks: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  STRATA_PREFILL_RING overrides.
inline int ring_slots(size_t T) {
    const char* v = std::getenv("STRATA_PREFILL_RING");
    const int r = v ? std::atoi(v) : (g_pinned_share >= 0.9 ? 384 : 96);
    const int big = r < 16 ? 16 : r > RING_MAX ? RING_MAX : r;
    return (int64_t) T >= STREAM_ALL_MIN ? big : STAGE;
}
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)

// F-1: STRATA_GR_UNFUSED=1 keeps the FP32 copy of the normalized rows (gr_norm + gr_mix), the A/B arm
inline bool gr_unfused() {
    static const bool v = [] { const char* e = std::getenv("STRATA_GR_UNFUSED"); return e && e[0] == '1'; }();
    return v;
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

}  // namespace

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    struct Job { const uint8_t* src; size_t bytes; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        cudaSetDevice(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing) {
                    while (issued.load(std::memory_order_acquire) <= j - kRing) std::this_thread::yield();
                    cudaEventSynchronize(dma_done[b]);
                }
                std::memcpy(buf[b], jobs[(size_t) j].src, jobs[(size_t) j].bytes);
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // The grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then the MMQ bounds: kernels
    // read and write them in place.  A cudaMemcpyAsync of them queues behind the expert blobs the copy stream already
    // holds (up to `ring` of them, ~70 us each), and the GPU idles meanwhile - measured 4.2 s of a 128K prompt's
    // 70 s at the default ring, 0.7 s with a 16-slot one.  STRATA_GROUP_COPY=1: the copies (the A/B arm).
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    // RDMA cold-expert tier (push channel 1): the non-resident experts of a layer go to the peer as ONE window
    // per chunk instead of being streamed from the host arena.  x_pf: the chunk's MoE inputs in the wire's
    // declared dtype (f32 rows, or q8_0 blocks + fp32 scales - the grouped kernel's input, quantized here
    // anyway), pinned, REMOTE_READ by the peer; y_pf: the peer's result rows (T_max*K rows of f32 or f16,
    // mapped pinned, written by the peer's chained RDMA write and read by the scatter through y_pf_dev);
    // dst: each entry's row in Dm (the expert-grouped layout), host staging + device.
    core::RDMAExpertTier* remote = nullptr;
    uint8_t* x_pf = nullptr;
    uint8_t* y_pf = nullptr;
    uint8_t* y_pf_dev = nullptr;             // device alias of y_pf (mapped)
    int32_t* pf_dst_host = nullptr;
    int32_t* pf_dst_dev = nullptr;
    cudaEvent_t pf_x_done = nullptr;
    int64_t pf_windows = 0, pf_entries = 0;  // stats
    double pf_ms_wait = 0;
    bool pf_x_f32 = false, pf_y_f32 = false; // STRATA_PF_X_F32 / STRATA_PF_Y_F32: the legacy wide wire formats
    bool pf_probe = false;                   // STRATA_PF_PROBE: per-window wire stats at run() end
    int64_t pf_x_bytes = 0, pf_y_bytes = 0;
    std::vector<double> pf_wait_ms;
    FILE* trace_f = nullptr;                 // STRATA_PREFILL_TRACE: --dump-routing records from prefill
    // The resident rows of a layer are computed by the decode path's grouped kernel (moe_grouped_s2, one
    // launch for ALL hit experts) instead of the per-expert MMQ/f16 loop: the per-expert launches cost
    // ~3.4 us/row on this path (measured, 3080), the grouped kernel is the one the worker serves remote rows
    // with.  x quantized per token (q8_0, the decode doorbell's quantization), groups in expert-id order,
    // output rows written STRAIGHT into Dm at their grouped positions (ent_dst = p, ent_tok = src[p]).
    uint8_t* pf_q8 = nullptr;
    float* pf_scales = nullptr;
    void* pf_gs_scratch = nullptr;
    const kernels::Nvfp4HotArena* nv4_hot = nullptr;
    void* pf_nv4_scratch = nullptr;
    std::vector<int32_t> pf_nv4_slot;
    unsigned long long* pf_gptr_dev = nullptr;
    int32_t *pf_gstart_dev = nullptr, *pf_edst_dev = nullptr, *pf_etok_dev = nullptr, *pf_gcount_dev = nullptr;
    std::vector<unsigned long long> pf_gptr;
    std::vector<int32_t> pf_gstart, pf_edst, pf_etok;
    int64_t pf_hit_rows = 0;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = std::getenv("STRATA_KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[0];
    if (q0.kv_mode != 1) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_fp8) {
        st.k_fp8 = o.take<uint8_t>(rows * s.head_dim, ok);
        st.v_fp8 = o.take<uint8_t>(rows * s.head_dim, ok);
        st.k_fp8_scale = o.take<uint16_t>(rows * (s.head_dim / strata::kernels::KV_FP8_GROUP), ok);
        st.v_fp8_scale = o.take<uint16_t>(rows * (s.head_dim / strata::kernels::KV_FP8_GROUP), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.k_fp8 = h.k_fp8; p.v_fp8 = h.v_fp8; p.k_fp8_scale = h.k_fp8_scale; p.v_fp8_scale = h.v_fp8_scale;
    p.page_table = table;
    return p;
}
}  // namespace

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() {
    if (!impl_) return;
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    if (impl_->copy) cudaStreamSynchronize(impl_->copy);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    for (void* p : impl_->owned) cudaFree(p);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (both weight types covered; the Strata Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            if (!mmq::supported(gt) || !mmq::supported(dt)) { p.fallback = true; continue; }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert) {
    const MmqPlan& mp = mmq_plan();
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(T * K * 1280, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
        a.take<float>(T * K * 640, ok);
        a.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), 640), ok);
    }
    return a.used;
}
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes,
                   core::RDMAExpertTier* remote) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    m.remote = remote != nullptr && remote->push_armed() ? remote : nullptr;
#ifdef STRATA_ENABLE_NVFP4
    const char* nv4pf = std::getenv("STRATA_PF_NVFP4_RESIDENT");
    // Resident marlin batches no longer require the remote tier: standalone (resident-all, no
    // --remote-rdma) runs the same grouped path with every row computed locally.
    if (nv4pf && nv4pf[0] == '1' && cache && cache->planar() &&
        nv4_hot_ && nv4_hot_->on()) m.nv4_hot = nv4_hot_;
#endif
    if (m.remote != nullptr && (cache == nullptr || host_res == nullptr)) {
        err = "prefill: the remote expert tier needs the residency table (the expert cache)";
        return false;
    }
    if (m.remote != nullptr && chunk > core::RDMAExpertTier::kMaxPushTok) {
        err = "prefill: a chunk of " + std::to_string(chunk) + " exceeds the remote window cap of " +
              std::to_string(core::RDMAExpertTier::kMaxPushTok) + " tokens";
        return false;
    }
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    cudaGetDevice(&m.device);
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (m.tok_dev == nullptr) {
        if (cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)) != cudaSuccess) {
            err = "prefill: the token id buffer";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = std::getenv("STRATA_STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        if (!m.stager->init((size_t) MAXBLOB(), stv ? std::clamp(std::atoi(stv), 1, 32) : std::max(2, std::min(4, hw / 4))))
            ok = false;
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    {
        const size_t need = 3 * T * K + (size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2));
        const char* gc = std::getenv("STRATA_GROUP_COPY");
        if (m.grp_n < need && !(gc && gc[0] == '1')) {
            if (m.grp_host) cudaFreeHost(m.grp_host);
            m.grp_host = m.grp_dev = nullptr;
            m.grp_n = m.grp_tk = 0;
            void *h = nullptr, *d = nullptr;
            if (cudaHostAlloc(&h, need * 4, cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess) {
                m.grp_host = (int32_t*) h;
                m.grp_dev = (int32_t*) d;
                m.grp_n = need;
                m.grp_tk = T * K;
            } else {                              // the copies, as before
                if (h) cudaFreeHost(h);
                cudaGetLastError();
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            cudaHostAlloc((void**) &m.ple_emb_host[b], (size_t) T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[0].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[0].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
            cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    // RDMA push channel 1 and/or the NVFP4 grouped path: the buffers, once, for the largest chunk
    // (relayout only shrinks T).  The peer one-sided-reads x_pf and RDMA-writes y_pf; both blocks live on
    // the host, so they cost no VRAM.  Standalone NVFP4 (resident-all, no remote tier) skips the window
    // blocks and takes only the event, the grouped-hit buffers and the marlin scratch.
    if ((m.remote != nullptr || m.nv4_hot) && m.pf_x_done == nullptr) {
        // STRATA_PF_X_F32 / STRATA_PF_Y_F32: fall back to the wide wire formats (A/B and numerics triage).
        // The defaults ship the q8_0 the grouped kernel eats anyway plus f16 result rows: a full window is
        // ~106 MB on the wire instead of 220 (and the mapped result block drops from 210 to 105 MB of
        // pinned host RAM on the 14 GB machine).  They are wire formats, so they only exist with a peer.
        m.pf_x_f32 = m.remote != nullptr && std::getenv("STRATA_PF_X_F32") != nullptr;
        m.pf_y_f32 = m.remote != nullptr && std::getenv("STRATA_PF_Y_F32") != nullptr;
        m.pf_probe = std::getenv("STRATA_PF_PROBE") != nullptr;
        if (cudaEventCreateWithFlags(&m.pf_x_done, cudaEventDisableTiming) != cudaSuccess) {
            err = "prefill: the grouped path's event for a chunk of " + std::to_string(chunk) + " tokens";
            return false;
        }
        size_t xb = 0, yb = 0;
        if (m.remote != nullptr) {
            xb = m.pf_x_f32 ? T * (size_t) N * sizeof(float)
                            : T * (size_t) (N / 32) * (36 + sizeof(float));
            yb = T * (size_t) K * (size_t) N * (m.pf_y_f32 ? sizeof(float) : sizeof(uint16_t));
            if (cudaHostAlloc((void**) &m.x_pf, xb, cudaHostAllocDefault) != cudaSuccess ||
                cudaHostAlloc((void**) &m.y_pf, yb, cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &m.y_pf_dev, m.y_pf, 0) != cudaSuccess ||
                cudaHostAlloc((void**) &m.pf_dst_host, T * (size_t) K * sizeof(int32_t),
                              cudaHostAllocDefault) != cudaSuccess ||
                cudaMalloc((void**) &m.pf_dst_dev, T * (size_t) K * sizeof(int32_t)) != cudaSuccess) {
                err = "prefill: the remote window buffers for a chunk of " + std::to_string(chunk) +
                      " tokens";
                return false;
            }
            m.owned.push_back(m.pf_dst_dev);
        }
        // the grouped-hit kernel's buffers (same shapes the worker sizes for a full-chunk window)
        const size_t q8b = T * (size_t) (N / 32) * 36, scb = T * (size_t) (N / 32) * sizeof(float);
        if (cudaMalloc((void**) &m.pf_q8, q8b) != cudaSuccess ||
            cudaMalloc((void**) &m.pf_scales, scb) != cudaSuccess ||
            cudaMalloc(&m.pf_gs_scratch,
                       (size_t) strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), N, 640)) !=
                cudaSuccess ||
            cudaMalloc((void**) &m.pf_gptr_dev, T * (size_t) K * sizeof(unsigned long long)) != cudaSuccess ||
            cudaMalloc((void**) &m.pf_gstart_dev, (T * (size_t) K + 1) * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &m.pf_edst_dev, T * (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &m.pf_etok_dev, T * (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &m.pf_gcount_dev, 4 * sizeof(int32_t)) != cudaSuccess) {
            err = "prefill: the grouped-hit buffers for a chunk of " + std::to_string(chunk) + " tokens";
            return false;
        }
        m.owned.push_back(m.pf_q8);
        m.owned.push_back(m.pf_scales);
        m.owned.push_back(m.pf_gs_scratch);
        m.owned.push_back(m.pf_gptr_dev);
        m.owned.push_back(m.pf_gstart_dev);
        m.owned.push_back(m.pf_edst_dev);
        m.owned.push_back(m.pf_etok_dev);
        m.owned.push_back(m.pf_gcount_dev);
#ifdef STRATA_ENABLE_NVFP4
        if (m.nv4_hot) {
            int sms = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, m.device);
            const auto bytes = kernels::moe_prefill_nvfp4_scratch_bytes(sms);
            if (cudaMalloc(&m.pf_nv4_scratch, bytes) != cudaSuccess) {
                err = "prefill: NVFP4 hot batch scratch does not fit";
                return false;
            }
            m.owned.push_back(m.pf_nv4_scratch);
            std::fprintf(stderr, "strata prefill: NVFP4 resident batches enabled, max %d rows, scratch %llu bytes\n",
                         kernels::NV4_PREFILL_ROWS, (unsigned long long) bytes);
        }
#endif
        if (m.remote != nullptr) {
            if (!m.remote->arm_push_prefill(m.x_pf, xb, m.y_pf, yb,
                                            m.pf_x_f32 ? core::RDMAExpertTier::kPushXf32
                                                       : core::RDMAExpertTier::kPushXq8,
                                            m.pf_y_f32 ? core::RDMAExpertTier::kPushYf32
                                                       : core::RDMAExpertTier::kPushYf16, err)) {
                err = "prefill: " + err;
                return false;
            }
            std::fprintf(stderr, "strata prefill: remote experts over RDMA push channel 1 (one window per "
                                 "layer per chunk, up to %lld tokens, x %s y %s)\n", (long long) T,
                         m.pf_x_f32 ? "f32" : "q8_0", m.pf_y_f32 ? "f32" : "f16");
        }
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);
    m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
    m.grs = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * D, ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[0].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        m.Xs = mp.fallback ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(T * K * 1280, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
            m.H = c.take<float>(T * K * 640, ok);
            m.Hq = c.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), 640), ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    for (int i = 0; i < m.ring; ++i) {
        m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
        m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    void* ws = o.take<uint8_t>(GEMM_WS, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }
void Prefill::set_session(core::SessionState& ss) { impl_->ss = &ss; }
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
double Prefill::pinned_share() { return g_pinned_share; }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D); f(T * D); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[0].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert)}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}  // namespace

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut,
               kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj"};
struct PfTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    std::vector<cudaEvent_t> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    void mark(int phase, cudaStream_t s) {
        if (!on) return;
        if (used == ev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        cudaEventRecord(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, ev[i], ev[i + 1]) == cudaSuccess) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PfTimer() {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
};
}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    // STRATA_PREFILL_TRACE=path: dump every layer's routing in --dump-routing's record format, so
    // tools/make_profile.py can rebuild the expert profile from what PREFILL really routes (the decode
    // trace the shipped profiles come from does not cover this path).
    if (const char* tp = std::getenv("STRATA_PREFILL_TRACE"))
        m.trace_f = std::fopen(tp, "wb");
    const int64_t LB = stage_lb_, LE = stage_le_;
    // the next stage reads chunk c on a thread while this one reads chunk c + 1 (declared first: an early return
    // waits for it before anything it reads goes away)
    std::string next_err;
    std::future<bool> next_run;
    int hand_buf = 0;
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    PfTimer pt;
    const cudaStream_t cs = (cudaStream_t) m.cs;
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = std::getenv("STRATA_PLE_BATCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env && strata::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / ((uint64_t) (3 * strata::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (std::getenv("STRATA_TRACE")) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        ++stats_.chunks;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr) {
            if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice, m.cs) !=
                cudaSuccess) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // C-4: the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr;
        const core::NativeEmbed* nemb = core::native_embed();
        const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
        if (batched && nemb == nullptr &&
            (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
             (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
            batched = false;
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            if (cudaMemcpyAsync(m.tok_dev, m.tok_host.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                                m.cs) != cudaSuccess) {
                err = "prefill: the token id upload failed";
                return false;
            }
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
            } else {
                const auto* codes = (const uint8_t*) wemb->data;
                const auto* scales = (const float*) (codes + wemb->codes_bytes);
                const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                       : nullptr;
                strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                      wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                      (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                      (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
            }
        }
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                if (cudaMemcpyAsync(m.emb + t * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request (read ahead on a thread, see ple_gather)
        if (ple_on) {
            const auto tp = Clock::now();
            if (!ple_next.valid()) {
                if (!ple_gather(c0, ple_buf, err)) return false;
            } else if (!ple_next.get()) {
                err = ple_next_err;
                return false;
            }
            cudaMemcpyAsync(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs);
            cudaEventRecord(m.ple_copied[ple_buf], m.cs);
            if (c0 + m.T < n) {
                cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]);   // the other buffer's upload (a chunk ago) is done
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + m.T, b = ple_buf ^ 1] {
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
        }
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount * 4,
                        cudaMemcpyHostToDevice, m.cs);

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        // With the remote tier the non-resident experts come from the peer, not the host arena: nothing to stream.
        const bool stream_all = m.remote == nullptr && m.ring > STAGE && T >= STREAM_ALL_MIN && m.src != nullptr;
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    const uint8_t* b = m.src->blob(l, e);
                    if (!b) { err = "prefill: expert source has no blob"; return false; }
                    int job = -1;
                    if (!m.src->pinned(l, e)) {
                        job = (int) js.size();
                        js.push_back({b, (size_t) lay0.blob_bytes(l)});
                    }
                    seq.push_back({(int32_t) l, e, b, job});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) seq_start[(size_t) l] = seq.size();
            m.stager->start(std::move(js));
        }
        struct StagerDone {
            Stager* st;
            ~StagerDone() { if (st) st->finish(); }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
        auto issue_until = [&](size_t limit) {
            limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                if (en.job < 0) {
                    cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                    m.stager->issued_one(en.job, m.copy);
                }
                cudaEventRecord(m.copied[sl], m.copy);
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
            }
        };
        // D-5: the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
        // behind a host copy of an unpinned blob (that wait left the GPU idle: the 'wait copy' / 'dequant' time of the
        // i-quant prompts).  The same copies in the same order into the same slots, and a slot is refilled only once
        // the compute stream has recorded that it is done with it: the same results.  STRATA_PREFILL_ISSUER=0: inline.
        static const bool issuer_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_ISSUER");
            return v == nullptr || std::atoi(v) != 0;
        }();
        std::atomic<size_t> a_issued{0}, a_consumed{0};
        std::atomic<bool> a_stop{false};
        double iss_ms = 0;
        int64_t iss_streamed = 0, iss_dma = 0;
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>* stop;
            std::thread* t;
            ~IssuerJoin() { if (t->joinable()) { stop->store(true); t->join(); } }
        } issuer_join{&a_stop, &issuer};
        const bool threaded_issue = stream_all && issuer_on;
        if (threaded_issue) {
            issuer = std::thread([&] {
                const core::OnDevice od(m.device);
                for (size_t idx = 0; idx < seq.size(); ++idx) {
                    while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                    const StreamEntry& en = seq[idx];
                    const int sl = (int) (idx % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                    if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                    if (en.job < 0) {
                        cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    cudaEventRecord(m.copied[sl], m.copy);
                    m.stage_live[sl] = true;
                    iss_ms += ms_since(th);
                    ++iss_streamed;
                    a_issued.store(idx + 1, std::memory_order_release);
                }
            });
        } else if (stream_all) {
            issue_until((size_t) m.ring);   // layer 0's first experts, behind the embedding and the PLE
        }
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            if (!threaded_issue) return;
            while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
        };
        auto give_back = [&](size_t upto) {
            if (threaded_issue) a_consumed.store(upto, std::memory_order_release);
            else issue_until(upto + (size_t) m.ring);
        };
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (grs, xn16)
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            const core::LayerView v(*m.wt, l);
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on && ple_batch) {
                // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                const strata::kernels::PleWeights& pw = ss.ple.w;
                constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * 2 + 4096;
                const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = m.region;
                    auto carve_f = [&](size_t n) { float* p = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return p; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    const float* emb = m.ple_emb + s0 * N;
                    if (pw.key_bf16 != nullptr) {
                        to_bf16(emb, e16, nb * N, m.cs);
                        m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                    } else {
                        to_f16(emb, e16, nb * N, m.cs);
                        m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                        to_bf16(emb, e16, nb * N, m.cs);
                    }
                    m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                    try {
                        strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                  gate, (int) nb, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                stats_.ms_ple += ms_since(tp);
            } else if (l == 1 && ple_on) {
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs);
                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs);
                normed = false;
                if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs);
                if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err)) return false;
                if (gr_unfused()) gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);
                else gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    pt.mark(kPfGdnRec, cs);
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                    pt.mark(kPfGdnOut, cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                    }
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, nullptr,   // mode 0: no host mirror
                                  staged ? &m.stage : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      nullptr, staged ? &m.stage : nullptr);
                    } else if (st.kv_q4) {   // Q4_0 KV (kv_q4.hpp): rotated K and V, the queries below too, the output back
                        strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                      &st.host, staged ? &m.stage : nullptr);
                    } else if (st.kv_fp8) {   // KV_MODE 2: e4m3 codes + fp16 scale; prompt attention reads the pool directly
                        strata::kernels::kv_append_fp8(st.k_fp8, st.v_fp8, st.k_fp8_scale, st.v_fp8_scale, st.page_table,
                                                       p0, T, m.Kc, m.Vc, s, m.cs, &st.host, staged ? &m.stage : nullptr);
                    } else {
                        kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                  st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                  &st.host, staged ? &m.stage : nullptr);
                    }
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                    if (st.kv_q4) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended). STRATA_INDEXER_PER_TOKEN=1: the old
                    try {
                        static const bool per_token = std::getenv("STRATA_INDEXER_PER_TOKEN") != nullptr;
                        if (!per_token) {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             (float) strata::kernels::qsa_freq_base(), m.cs);
                        }
                        for (int64_t t = 0; per_token && t < T; ++t) {
                            const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       (float) strata::kernels::qsa_freq_base(), m.cs);
                        }
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise); STRATA_SELECT_OLD=1: the warp kernel
                        static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = std::getenv("STRATA_SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        cudaMemcpyAsync(ids.data(), m.sel_ids, ids.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "strata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (cudaMalloc((void**) &pooled16, (size_t) rows * s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &dead16, (size_t) s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &ids16, (size_t) (m.T * m.cap) * 4) != cudaSuccess)) {
                            err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        cudaMemcpyAsync(a.data(), m.sel_ids, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(b.data(), ids16, b.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            cudaMemcpyAsync(h.data(), m.sel_ids, h.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    // perf-review D-1: the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV, or STRATA_PROMPT_ATTN_OLD=1: the
                    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
                    // V dequantized from its q4_0 blocks to fp16 at gather)
                    static const bool old_attn = std::getenv("STRATA_PROMPT_ATTN_OLD") != nullptr;
                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_q4 || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    pt.mark(kPfQsa, cs);
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    // group the (token, k) pairs by expert on the host
                    pt.mark(kPfHostGroup, cs);
                    // (the sync below also orders this layer's writes of slot/src/bounds after the previous
                    // layer's kernels that read them)
                    const bool grp_mapped = m.grp_host != nullptr;
                    int32_t* ids_h = grp_mapped ? m.grp_host : m.ids_host.data();
                    int32_t* slot_h = grp_mapped ? m.grp_host + m.grp_tk : m.slot_host.data();
                    int32_t* src_h = grp_mapped ? m.grp_host + 2 * m.grp_tk : m.src_host.data();
                    if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
                    else cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
                    cudaStreamSynchronize(m.cs);
                    pt.fold();
                    std::fill(m.cnt.begin(), m.cnt.end(), 0);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = ids_h[(size_t) i];
                        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                        ++m.cnt[(size_t) e];
                    }
                    if (m.trace_f != nullptr) {
                        // the record --dump-routing writes: i32 layer, i32 k, k expert ids, k f32 weights
                        // (make_profile.py only counts the ids; the weights slot stays zero).  ids_h is in
                        // routing order, so token t's ids sit at ids_h[t * K .. t * K + K).
                        const int32_t hdr[2] = {(int32_t) l, (int32_t) K};
                        const float zw[K] = {};
                        for (int64_t t = 0; t < T; ++t) {
                            std::fwrite(hdr, sizeof(int32_t), 2, m.trace_f);
                            std::fwrite(ids_h + (size_t) t * K, sizeof(int32_t), (size_t) K, m.trace_f);
                            std::fwrite(zw, sizeof(float), (size_t) K, m.trace_f);
                        }
                    }
                    m.off[0] = 0;
                    for (int64_t e = 0; e < m.g->n_expert; ++e) m.off[(size_t) e + 1] = m.off[(size_t) e] + m.cnt[(size_t) e];
                    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = ids_h[(size_t) i];
                        const int32_t p = fill[(size_t) e]++;
                        slot_h[(size_t) i] = p;
                        src_h[(size_t) p] = (int32_t) (i / K);
                    }
                    if (grp_mapped) {
                        copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
                        copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
                    } else {
                        cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                        cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                    }
                    // ---- RDMA push channel 1 + grouped hits: this layer's experts are computed in TWO batched
                    // launches and nothing per-expert.  Misses: ONE window to the peer (entries in routing order,
                    // row = token * K + j; pf_dst_host maps each entry to its grouped row of Dm).  Residents: ONE
                    // moe_grouped_s2 launch (the decode hit kernel's explicit form) writing its rows STRAIGHT into
                    // Dm (ent_dst = grouped position p, ent_tok = src[p]).  The per-expert MMQ/f16 loop below is
                    // skipped entirely - measured 3.4 us/row on the 3080, launch-bound.
                    // Standalone NVFP4 (resident-all) runs the same grouped gather with no peer: every
                    // row is a planar hit the marlin kernel computes locally, pf_ne stays 0, no window.
                    const bool remote_on = m.remote != nullptr || m.nv4_hot;
                    int32_t pf_ne = 0;
                    uint64_t pf_seq = 0;
                    if (remote_on) {
                        const int32_t* res_row = m.host_res + (size_t) l * (size_t) m.g->n_expert;
                        std::vector<int32_t> ent;
                        ent.reserve((size_t) T * K / 2);
                        const bool pf_all_remote = m.cache->planar() && !m.nv4_hot;
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            if (!pf_all_remote && res_row[e] >= 0) continue;
                            if (m.remote != nullptr)
                                m.pf_dst_host[(size_t) pf_ne] = slot_h[(size_t) i];
                            ent.push_back((int32_t) i);
                            ent.push_back(e);
                            ++pf_ne;
                        }
                        if (pf_ne > 0 && m.remote == nullptr) {
                            err = "prefill: NVFP4 resident-all: an expert missed the planar tier at layer " +
                                  std::to_string(l) + " and there is no remote tier (startup fill bug)";
                            return false;
                        }
                        m.pf_gptr.clear();
                        m.pf_gstart.clear();
                        m.pf_edst.clear();
                        m.pf_etok.clear();
                        m.pf_nv4_slot.clear();
                        // GMAX (s2_expert_grouped.cu): the small-block tile kernel serves at most 8 entries of
                        // a group; a chunk routes HUNDREDS of rows to one expert, so each expert's rows are
                        // split into sub-groups of 8 (same blob pointer, consecutive entries).  Many small
                        // independent blocks keep the latency-bound scheduler fed - measured faster than any
                        // big-block variant tried (tile-loop 4859ms / lane-per-entry 21121ms / this 3652ms
                        // of gather at 4K chunk 2048, including the peer wait).
                        constexpr int64_t kGMAX = 8;
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            const int64_t c = m.cnt[(size_t) e];
                            if (pf_all_remote || c <= 0 || res_row[e] < 0) continue;
                            const uint8_t* slot = m.nv4_hot ? nullptr : m.cache->device_slot(res_row[e]);
                            const int64_t o0 = m.off[(size_t) e];
                            for (int64_t p = o0; p < o0 + c; p += kGMAX) {
                                m.pf_gptr.push_back((unsigned long long) (uintptr_t) slot);
                                m.pf_gstart.push_back((int32_t) m.pf_edst.size());
                                const int64_t q1 = std::min(p + kGMAX, o0 + c);
                                for (int64_t q = p; q < q1; ++q) {
                                    m.pf_edst.push_back((int32_t) q);
                                    m.pf_etok.push_back(src_h[(size_t) q]);
                                    if (m.nv4_hot) m.pf_nv4_slot.push_back(res_row[e]);
                                }
                            }
                        }
                        m.pf_gstart.push_back((int32_t) m.pf_edst.size());
                        const int64_t nh = (int64_t) m.pf_edst.size();
                        m.pf_hit_rows += nh;
                        // the chunk's MoE inputs for the peer (m.mixed is final: the routing sync above drained
                        // the stream).  The slim wire ships the q8_0 the grouped kernel uses anyway, so the
                        // quantize runs ahead of the download whenever either side needs it; the event still
                        // lets the host publish as soon as the download lands, WITHOUT waiting for the grouped
                        // launch queued behind it
                        const bool need_q8 = (!m.nv4_hot && nh > 0) || (!m.pf_x_f32 && pf_ne > 0);
                        if (need_q8)
                            strata::kernels::quantize_q8_0_scaled(m.mixed, m.pf_q8, m.pf_scales, T * N, m.cs);
                        if (m.pf_x_f32) {
                            cudaMemcpyAsync(m.x_pf, m.mixed, (size_t) T * (size_t) N * sizeof(float),
                                            cudaMemcpyDeviceToHost, m.cs);
                        } else if (pf_ne > 0) {
                            const size_t qb = (size_t) T * (size_t) (N / 32) * 36;
                            cudaMemcpyAsync(m.x_pf, m.pf_q8, qb, cudaMemcpyDeviceToHost, m.cs);
                            cudaMemcpyAsync(m.x_pf + qb, m.pf_scales,
                                            (size_t) T * (size_t) (N / 32) * sizeof(float),
                                            cudaMemcpyDeviceToHost, m.cs);
                        }
                        cudaEventRecord(m.pf_x_done, m.cs);
                        if (pf_ne > 0)
                            cudaMemcpyAsync(m.pf_dst_dev, m.pf_dst_host, (size_t) pf_ne * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, m.cs);
                        pt.mark(kPfGather, cs);
                        if (nh > 0) {
                            if (static const bool dbg = std::getenv("STRATA_PF_DEBUG") != nullptr; dbg)
                                std::fprintf(stderr, "pf layer %lld: T %lld hits %lld groups %zu misses %d\n",
                                             (long long) l, (long long) T, (long long) nh, m.pf_gptr.size(),
                                             (int) pf_ne);
#ifdef STRATA_ENABLE_NVFP4
                            if (m.nv4_hot) {
                                for (int64_t off = 0; off < nh; off += kernels::NV4_PREFILL_ROWS) {
                                    const int nr = (int) std::min<int64_t>(kernels::NV4_PREFILL_ROWS, nh - off);
                                    kernels::moe_prefill_grouped_nvfp4(*m.nv4_hot, m.pf_nv4_slot.data() + off,
                                        m.pf_edst.data() + off, m.pf_etok.data() + off, nr, m.mixed,
                                        m.pf_nv4_scratch, m.Dm, m.cs);
                                }
                            } else
#endif
                            {
                            const int32_t counts[4] = {(int32_t) m.pf_gptr.size(), (int32_t) nh, 0, 0};
                            cudaMemcpyAsync(m.pf_gptr_dev, m.pf_gptr.data(),
                                            m.pf_gptr.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                                            m.cs);
                            cudaMemcpyAsync(m.pf_gstart_dev, m.pf_gstart.data(),
                                            m.pf_gstart.size() * sizeof(int32_t), cudaMemcpyHostToDevice, m.cs);
                            cudaMemcpyAsync(m.pf_edst_dev, m.pf_edst.data(), (size_t) nh * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, m.cs);
                            cudaMemcpyAsync(m.pf_etok_dev, m.pf_etok.data(), (size_t) nh * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, m.cs);
                            cudaMemcpyAsync(m.pf_gcount_dev, counts, sizeof counts, cudaMemcpyHostToDevice, m.cs);
                            strata::kernels::moe_grouped_s2(
                                m.pf_gptr_dev, m.pf_gstart_dev, m.pf_gcount_dev, m.pf_edst_dev, m.pf_etok_dev,
                                (int64_t) m.pf_gptr.size(), nh, m.pf_q8, m.pf_scales, m.pf_gs_scratch, m.Dm, m.cs);
                            }
                        }
                        if (pf_ne > 0) {
                            if (cudaEventSynchronize(m.pf_x_done) != cudaSuccess) {
                                err = "prefill: the remote window's activation download failed";
                                return false;
                            }
                            if (!m.remote->push_begin_prefill(l, (int32_t) T, K, ent.data(), pf_ne, pf_seq, err))
                                return false;
                            ++m.pf_windows;
                            m.pf_entries += pf_ne;
                            if (m.pf_probe) {
                                m.pf_x_bytes += (int64_t) T * (int64_t) (m.pf_x_f32 ? N * 4 : (N / 32) * 40);
                                m.pf_y_bytes += (int64_t) pf_ne * (int64_t) N * (m.pf_y_f32 ? 4 : 2);
                            }
                        }
                    }
                    if (!remote_on) {
                    // the experts, in id order: resident ones from VRAM, the others through the staging ring
                    std::vector<int32_t> order;
                    for (int32_t e = 0; e < m.g->n_expert; ++e) if (m.cnt[(size_t) e] > 0) order.push_back(e);
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                    const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                    pt.mark(kPfGather, cs);
                    if (use_mmq) {
                        // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
                        mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs);
                        // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                        // reads the group's own quantized H)
                        const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                        m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                        for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                        m.bounds_host[n] = (int32_t) (T * K);
                        for (size_t g = 0; g < ng; ++g)
                            for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                    m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
                        if (grp_mapped) {
                            int32_t* bh = m.grp_host + 3 * m.grp_tk;
                            std::memcpy(bh, m.bounds_host.data(), m.bounds_host.size() * 4);
                            copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
                        } else {
                            cudaMemcpyAsync(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4,
                                            cudaMemcpyHostToDevice, m.cs);
                        }
                    } else {
                        gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                    }
                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                    int stage_next = 0;
                    std::vector<int> stage_of(order.size(), -1);
                    // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                    std::vector<int> job_of(order.size(), -1);
                    if (!stream_all) {
                        std::vector<Stager::Job> js;
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                            if (m.src->pinned(l, e)) continue;
                            const uint8_t* b = m.src->blob(l, e);
                            if (!b) { err = "prefill: expert source has no blob"; return false; }
                            job_of[j] = (int) js.size();
                            js.push_back({b, (size_t) lay.blob_bytes(l)});
                        }
                        m.stager->start(std::move(js));
                    }
                    StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                    auto stage_one = [&](size_t j) -> bool {
                        const int32_t e = order[j];
                        const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        if (resident) return true;
                        const int sl = stage_next;
                        stage_next = (stage_next + 1) % STAGE;
                        const auto th = Clock::now();
                        const uint8_t* b = m.src->blob(l, e);
                        if (!b) { err = "prefill: expert source has no blob"; return false; }
                        if (m.src->pinned(l, e)) {
                            // DMA straight from the page-locked arena: the copy stream only waits for the slot
                            if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                            cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                            ++stats_.experts_dma;
                        } else {
                            // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                            const uint8_t* hb = m.stager->wait(job_of[j]);
                            if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                            cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                            m.stager->issued_one(job_of[j], m.copy);
                        }
                        cudaEventRecord(m.copied[sl], m.copy);
                        m.stage_live[sl] = true;
                        stage_of[j] = sl;
                        stats_.ms_experts_host += ms_since(th);
                        ++stats_.experts_streamed;
                        return true;
                    };
                    // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                    // expert) is released once the blob is read
                    auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                        const int32_t e = order[j];
                        pt.mark(kPfDequant, cs);
                        if (use_mmq) {
                            // gather the expert into its group slot (GGUF blocks, unchanged or converted)
                            const size_t q = j % MMQ_GROUP;
                            if (lay.native) {
                                const auto& f = lay.fmt[(size_t) l];
                                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                   mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                            } else {
                                mmq::gather_strata_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                            }
                            if (slot >= 0) cudaEventRecord(m.used[slot], m.cs);
                            if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                            // the group's products: gate/up, swiglu, the group's H to q8_1, down
                            const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                            const int ngx = (int) (q + 1);
                            const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                            int64_t maxr = 0;
                            for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                            pt.mark(kPfGemmGU, cs);
                            // the zeroed tail after the group's last expert (see MMQ_TAIL)
                            cudaMemsetAsync(m.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, m.cs);
                            cudaMemsetAsync(m.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, m.cs);
                            mmq::Product gu;
                            gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                            gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                            gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                            m.mmq_ctx->run(gu, m.cs);
                            mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs);
                            pt.mark(kPfGemmD, cs);
                            mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
                            mmq::Product dn;
                            dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                            dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                            dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                            dn.ld_dst = N;
                            m.mmq_ctx->run(dn, m.cs);
                            return true;
                        }
                        const int q = (int) (j % DQ);
                        if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                            const auto& f = lay.fmt[(size_t) l];
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                               m.dq_gu[q], m.cs);
                            strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs);
                        } else {
                            blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                        }
                        if (slot >= 0) cudaEventRecord(m.used[slot], m.cs);
                        const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                        pt.mark(kPfGemmGU, cs);
                        m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                        swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                        pt.mark(kPfGemmD, cs);
                        m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                        return true;
                    };
                    if (!stream_all) {
                        size_t staged = 0;
                        const size_t lookahead = STAGE - 1;
                        for (size_t j = 0; j < order.size(); ++j) {
                            while (staged < order.size() && staged <= j + lookahead) {
                                if (!stage_one(staged)) return false;
                                ++staged;
                            }
                            const int32_t e = order[j];
                            if (stage_of[j] < 0) {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            } else {
                                pt.mark(kPfWaitCopy, cs);
                                cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                                if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
                            }
                        }
                    } else {
                        // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                        // pick only gives its slot back
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        auto release_to = [&](int32_t e_stop) {
                            while (k < kend && seq[k].e < e_stop) {
                                cudaEventRecord(m.used[k % (size_t) m.ring], m.cs);
                                consumed = ++k;
                                give_back(consumed);
                            }
                        };
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            release_to(e);
                            if (k < kend && seq[k].e == e) {
                                const int sl = (int) (k % (size_t) m.ring);
                                pt.mark(kPfWaitCopy, cs);
                                wait_issued(k);
                                cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                                if (!compute(j, m.stage_dev[sl], sl)) return false;
                                consumed = ++k;
                                give_back(consumed);
                            } else {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            }
                        }
                        release_to(m.g->n_expert);
                    }
                    }   // !remote_on: the per-expert loop is replaced by the grouped hit kernel + the window
                    // ---- the peer's window: wait for its flag (the rows are already in y_pf, written by the
                    // peer's chained RDMA write), then scatter them into Dm at their grouped rows.  The next
                    // layer's routing sync drains the stream before the block/buffers are republished, so the
                    // worker never writes y_pf while this read is in flight.
                    if (remote_on && pf_ne > 0) {
                        const auto tw = Clock::now();
                        if (!m.remote->push_wait(pf_seq, err)) {
                            err = "prefill layer " + std::to_string(l) + ": " + err;
                            return false;
                        }
                        const double wms = ms_since(tw);
                        m.pf_ms_wait += wms;
                        if (m.pf_probe) m.pf_wait_ms.push_back(wms);
                        if (m.pf_y_f32)
                            scatter_rows_f32(m.Dm, (const float*) m.y_pf_dev, m.pf_dst_dev, pf_ne, N, m.cs);
                        else
                            scatter_rows_f16(m.Dm, (const uint16_t*) m.y_pf_dev, m.pf_dst_dev, pf_ne, N, m.cs);
                    }
                    pt.mark(kPfCombine, cs);
                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (!remote_on) {
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        cudaStreamSynchronize(m.cs);
                        auto bad = [&](const float* d, int64_t n) {
                            std::vector<float> h((size_t) n);
                            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        };
                        const int64_t bgu = bad(m.GU, T * K * 1280), bdm = bad(m.Dm, T * K * N), bbo = bad(m.bo, T * N);
                        const int64_t bh = m.H ? bad(m.H, T * K * 640) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) mmq_plan().any,
                                         (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
                    }
                    }   // !remote_on (debug block)
                }
                // ---- the hyper-connection write of this half; F-2: fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                const bool fuse = !gr_unfused() && nl < LE && !(half == 1 && nl == 1 && ple_on) &&
                                  !(half == 1 && strata::kernels::cvec().covers(l));
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                }
                if (half == 1 && strata::kernels::cvec().covers(l))   // --control-vector-scaled
                    strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
            }
        }
        if (issuer.joinable()) {
            issuer.join();
            stats_.ms_experts_host += iss_ms;
            stats_.experts_streamed += iss_streamed;
            stats_.experts_dma += iss_dma;
        }
        stats_.tokens += T;
        pt.mark(kPfStart, cs);
        if (next_ != nullptr) {
            // the rows to the host buffer the next stage read two chunks ago (it has finished: waited below)
            float* h = m.hand[hand_buf];
            if (cudaMemcpyAsync(h, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
                cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            // this stage's state is at the chunk's end now (synced) and moves on with the next chunk below
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (next_run.valid() && !next_run.get()) { err = next_err; return false; }
            next_->hand_in_ = h;
            next_run = std::async(std::launch::async, [this, tokens, c0, T, p0, &next_err] {
                return next_->run(tokens + c0, T, p0, next_err);
            });
            hand_buf ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            cudaStreamSynchronize(m.cs);                                  // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    cudaMemcpy(row.data(), m.R + t * D, (size_t) D * 4, cudaMemcpyDeviceToHost);
                    const int64_t pos = p0 + t;
                    std::fwrite(&pos, sizeof pos, 1, f);
                    std::fwrite(row.data(), 4, row.size(), f);
                }
                std::fclose(f);
            }
        }
        if (on_chunk || on_stage_chunk) {
            const auto toc = Clock::now();
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            const auto toc2 = Clock::now();
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
            host_sync_ms += std::chrono::duration<double, std::milli>(toc2 - toc).count();
            host_chunk_ms += ms_since(toc2);
        }
    }
    if (next_run.valid() && !next_run.get()) { err = next_err; return false; }
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (std::getenv("STRATA_DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        cudaStreamSynchronize(m.cs);
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % m.T;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // (PR #121) an expert copy that failed on the copy stream surfaces here, not in the next request
    if (const cudaError_t cst = cudaStreamSynchronize(m.copy); cst != cudaSuccess) {
        err = std::string("prefill: expert copy stream: ") + cudaGetErrorString(cst);
        return false;
    }
    stats_.ms_total += ms_since(t_start);
    if (m.trace_f != nullptr) {
        std::fclose(m.trace_f);
        m.trace_f = nullptr;
        std::fprintf(stderr, "strata prefill: routing trace for tools/make_profile.py written\n");
    }
    if (m.pf_probe && m.pf_windows > 0) {
        std::vector<double> w = m.pf_wait_ms;
        std::sort(w.begin(), w.end());
        const double mean = m.pf_ms_wait / (double) w.size();
        std::fprintf(stderr, "strata prefill probe: %lld windows, wire x %.1f MB y %.1f MB, %lld rows to the "
                             "peer, peer wait mean %.1f p50 %.1f p99 %.1f max %.1f ms\n",
                     (long long) m.pf_windows, m.pf_x_bytes / 1048576.0, m.pf_y_bytes / 1048576.0,
                     (long long) m.pf_entries, mean, w[w.size() / 2],
                     w[(size_t) ((double) (w.size() - 1) * 0.99)], w.back());
    }
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
        std::fprintf(stderr, "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, "
                             "waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, "
                             "PLE %.0f ms\n", host_setup_ms, host_sync_ms, host_chunk_ms, stats_.ms_ple);
        if (m.remote != nullptr)
            std::fprintf(stderr, "strata prefill timing: remote: %lld windows, %lld expert rows to the peer, "
                                 "%lld resident rows by the grouped kernel, waiting on the peer %.0f ms\n",
                         (long long) m.pf_windows, (long long) m.pf_entries, (long long) m.pf_hit_rows,
                         m.pf_ms_wait);
    }
    if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        cudaStreamSynchronize(m.cs);
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < g.n_gdn_layers(); ++i) {
            cudaMemcpy(b.data(), ss.gdn_state + (size_t) i * gdn_floats, b.size(), cudaMemcpyDeviceToHost);
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    return true;
}

}  // namespace strata::prefill
