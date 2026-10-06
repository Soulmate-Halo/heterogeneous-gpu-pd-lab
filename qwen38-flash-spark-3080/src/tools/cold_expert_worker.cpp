// tools/cold_expert_worker.cpp - the Spark half of the RDMA cold-expert tier (docs/RDMA_EXPERTS.md).
//
// Three modes:
//   serve (default)  Receiver on the flag ring: poll the doorbell, compute the requested Q2_0 experts on the
//                    GPU straight out of the pack's experts.bin arena, publish the rows into the result ring.
//   --selftest       numeric check of the CUDA grouped path against the portable scalar reference
//                    (include/strata/kernels/q2_0_scalar_ref.hpp), no peer needed.
//   --probe          SENDER side: drives a real worker over RDMA through RDMAExpertTier and verifies the
//                    returned rows against the scalar reference.  This is the end-to-end gate.
//
// The kernel path is the same `moe_grouped_s2` the in-process CUDA tiers use (src/core/remote_experts.cpp),
// so a row the worker computes is the row a local helper GPU would compute, bit for bit.

#include "strata/core/rdma_expert_tier.hpp"
#include "strata/core/remote_stage.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/q2_0_scalar_ref.hpp"
#include "strata/kernels/nvfp4_cold.hpp"
#include "strata/kernels/nvfp4_cpu.h"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using strata::core::RDMAExpertTier;
using strata::core::RemoteStage;
using strata::core::RemoteStageConfig;
using strata::core::RemoteStageRole;
namespace cpuk = strata::kernels::cpu;

constexpr int64_t H = cpuk::H;
constexpr int64_t FF = cpuk::FF;
constexpr int CAP_TOK = 2048;                // push channel 1 serves a whole prefill chunk per window
constexpr int CAP_ENTRIES = CAP_TOK * 10;    // worst case: every routed entry of the chunk is a miss
constexpr double kRelTol = 5e-3;           // vs the quantized-input reference: fp32 order is ~1e-4; layout bugs are O(1)

struct Args {
    std::string pack;
    std::string peer;
    std::string ple_gguf;      // --ple-gguf: serve the PLE table from RAM over RDMA (docs/RDMA_EXPERTS.md)
    std::string ple_fp8;       // --ple-fp8: the NVFP4 stack's raw FP8 E4M3 table (160 B rows), same service
    float ple_fp8_scale = 0.0f;  // --ple-fp8-scale: the table's single global dequant scale (required with --ple-fp8)
    std::string bind = "0.0.0.0";
    std::string device;
    std::string format = "auto";   // q2 | nvfp4 | auto (auto: manifest-cold.json present -> nvfp4)
    int port = 39580;
    int gid = 0;
    int slots = 4;
    long long slot_bytes = 4 << 20;
    int layers = 48;
    int experts = 512;
    int timeout_ms = 600000;   // the accept window: the engine may take a while to start
    int wait_ms = 30000;
    int max_windows = 0;       // 0 = forever
    int windows = 8;           // probe window count
    bool selftest = false;
    bool probe = false;
    bool verbose = false;
    int cpu_threads = 0;         // --cpu-threads: GB10 CPU split tier size (0 = off, decode windows only)
    float cpu_share = 0.5f;      // --cpu-share: fraction of a decode window's entries computed on the CPU
};

void usage() {
    std::fprintf(stderr,
        "usage: strata-cold-expert-worker --pack DIR [--ple-gguf PATH | --ple-fp8 PATH --ple-fp8-scale F]\n"
        "                                 [--bind H] [--port N] [--device HCA] [--gid N]\n"
        "                                 [--slots N] [--slot-bytes B] [--layers N] [--experts N]\n"
        "                                 [--format q2|nvfp4|auto] [--max-windows N] [--timeout-ms N] [--verbose]\n"
        "                                 [--cpu-threads N] [--cpu-share F]   (nvfp4 decode windows only)\n"
        "       strata-cold-expert-worker --selftest --pack DIR\n"
        "       strata-cold-expert-worker --probe --pack DIR --peer HOST [--windows N] [--wait-ms N]\n");
}

bool cuda_ok(cudaError_t st, const char* what, std::string& err) {
    if (st == cudaSuccess) return true;
    err = std::string("cuda: ") + what + ": " + cudaGetErrorString(st);
    return false;
}

// ---- NVFP4 cold pack helpers (P3) --------------------------------------------------------------
// The NVFP4 pack is per-layer files of 512 sequential canonical v2 blobs (tools/nvfp4_pack.py);
// the manifest only has to give up blob_bytes, and even that is cross-checked against the fixed
// canonical size - a malformed pack fails closed here, not inside a kernel.
bool nvfp4_manifest_blob_bytes(const std::string& pack, int64_t& out, std::string& err) {
    const std::string path = pack + "/manifest-cold.json";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "worker: cannot open " + path; return false; }
    std::string text;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    const size_t at = text.find("\"blob_bytes\"");
    const size_t colon = at == std::string::npos ? at : text.find(':', at);
    if (colon == std::string::npos) { err = "worker: manifest-cold.json has no blob_bytes"; return false; }
    const char* begin = text.c_str() + colon + 1;
    char* end = nullptr;
    const long long v = std::strtoll(begin, &end, 10);
    if (end == begin || v <= 0) { err = "worker: manifest-cold.json blob_bytes is not a number"; return false; }
    out = v;
    return true;
}

std::string nvfp4_layer_path(const std::string& pack, int64_t layer) {
    char name[64];
    std::snprintf(name, sizeof name, "layer-%05d-cold-nvfp4.bin", (int) layer);
    return pack + "/" + name;
}

// Geometry + the first blob's header (magic/version/flavor, and the header's own layer/expert
// when `want_expert0` is set) - fail closed on anything but the exact cold format.
bool nvfp4_check_layer_file(FILE* f, const std::string& path, int64_t experts, int64_t blob_bytes,
                            int64_t want_layer, std::string& err) {
    std::fseek(f, 0, SEEK_END);
    const long long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const long long want = (long long) experts * (long long) blob_bytes;
    if (size != want) {
        err = "worker: " + path + " is " + std::to_string(size) + " B, the geometry needs " +
              std::to_string(want) + " B";
        return false;
    }
    uint8_t head[24];
    if (std::fread(head, 1, sizeof head, f) != sizeof head ||
        std::memcmp(head, "STRNVFP4", 8) != 0 || head[8] != 2 || head[9] != 0 || head[10] != 2 ||
        head[11] != 0) {
        err = "worker: " + path + " is not a v2 cold NVFP4 pack (magic/version/flavor)";
        return false;
    }
    const uint32_t hlayer = (uint32_t) head[12] | ((uint32_t) head[13] << 8) |
                            ((uint32_t) head[14] << 16) | ((uint32_t) head[15] << 24);
    if ((int64_t) hlayer != want_layer) {
        err = "worker: " + path + " first blob is layer " + std::to_string(hlayer) + ", wanted " +
              std::to_string(want_layer);
        return false;
    }
    std::fseek(f, 0, SEEK_SET);
    return true;
}

// The whole pack resident on the GPU: blob (layer, expert) = base + (layer * experts + expert) * BLOB.
// P3: two formats behind one addressing contract - Q2_0's single experts.bin (cpuk::BLOB stride)
// and NVFP4's per-layer files (manifest blob_bytes stride); a mini mode for the selftest loads
// only the blobs it references instead of the whole 68 GB.
struct ExpertArena {
    uint8_t* d_base = nullptr;
    int64_t layers = 0, experts = 0;
    bool nvfp4 = false;
    int64_t blob_bytes = (int64_t) cpuk::BLOB;
    std::unordered_map<int64_t, int32_t> mini;                    // key -> slot; empty = full pack
    std::unordered_map<int64_t, std::vector<uint8_t>> mini_host;  // key -> blob copy (selftest ref)
    // P5 CPU split tier blob source: read-only mmaps of the per-layer pack files (page cache).
    // NOT cudaMallocManaged - CPU reads of a 63 GiB managed arena wedge the GB10 memory
    // subsystem (sshd starved, load 35 - measured twice 2026-10-03); file pages drop cleanly.
    std::vector<const uint8_t*> host_maps;   // per layer, PROT_READ MAP_PRIVATE
    std::vector<uint64_t> host_maps_len;
    static int64_t key(int64_t layer, int64_t expert, int64_t experts_) {
        return layer * experts_ + expert;
    }
    bool open_q2(const std::string& pack, int64_t n_layers, int64_t n_expert, std::string& err) {
        nvfp4 = false;
        blob_bytes = (int64_t) cpuk::BLOB;
        layers = n_layers;
        experts = n_expert;
        const std::string path = pack + "/experts.bin";
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "worker: cannot open " + path; return false; }
        std::fseek(f, 0, SEEK_END);
        const long long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        const long long want = (long long) n_layers * n_expert * (long long) cpuk::BLOB;
        if (size != want) {
            std::fclose(f);
            err = "worker: experts.bin is " + std::to_string(size) + " B, the geometry needs " +
                  std::to_string(want) + " B";
            return false;
        }
        size_t free_b = 0, total_b = 0;
        if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "meminfo", err)) { std::fclose(f); return false; }
        if ((uint64_t) size + (512ull << 20) > free_b) {
            std::fclose(f);
            err = "worker: experts.bin leaves less than 512 MiB of device memory free";
            return false;
        }
        if (!cuda_ok(cudaMalloc(&d_base, (size_t) size), "expert arena", err)) { std::fclose(f); return false; }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> chunk(1u << 28);   // 256 MiB staging
        long long off = 0;
        while (off < size) {
            const size_t n = (size_t) std::min<long long>((long long) chunk.size(), size - off);
            if (std::fread(chunk.data(), 1, n, f) != n) {
                std::fclose(f);
                err = "worker: short read on experts.bin";
                return false;
            }
            if (!cuda_ok(cudaMemcpy(d_base + off, chunk.data(), n, cudaMemcpyHostToDevice), "arena load", err)) {
                std::fclose(f);
                return false;
            }
            off += (long long) n;
        }
        // GB10 unified memory: the 34 GiB just streamed sits in the page cache, and the PLE table's 28.8 GiB
        // cudaHostAlloc right after would land in direct reclaim - which wedges the allocation on this
        // platform.  The arena lives on the device now; drop the file's pages.
        posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
        std::fclose(f);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "worker: expert arena %.2f GiB loaded in %.1f s (%.2f GiB/s)\n",
                     (double) size / 1073741824.0, s, s > 0 ? (double) size / 1073741824.0 / s : 0.0);
        return true;
    }
    bool open_nvfp4(const std::string& pack, int64_t n_layers, int64_t n_expert, std::string& err,
                    bool want_host = false) {
        nvfp4 = true;
        layers = n_layers;
        experts = n_expert;
        if (!nvfp4_manifest_blob_bytes(pack, blob_bytes, err)) return false;
        if (blob_bytes != strata::kernels::NVFP4_COLD_BLOB_BYTES) {
            err = "worker: manifest blob_bytes " + std::to_string(blob_bytes) +
                  " != canonical cold size " + std::to_string(strata::kernels::NVFP4_COLD_BLOB_BYTES);
            return false;
        }
        const uint64_t total = (uint64_t) layers * experts * (uint64_t) blob_bytes;
        size_t free_b = 0, total_b = 0;
        if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "meminfo", err)) return false;
        if (total + (512ull << 20) > free_b) {
            err = "worker: NVFP4 cold arena (" + std::to_string(total >> 20) +
                  " MiB) leaves less than 512 MiB of device memory free";
            return false;
        }
        if (!cuda_ok(cudaMalloc(&d_base, (size_t) total), "nvfp4 expert arena", err))
            return false;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> chunk(1u << 28);   // 256 MiB staging
        for (int64_t layer = 0; layer < layers; ++layer) {
            const std::string path = nvfp4_layer_path(pack, layer);
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) { err = "worker: cannot open " + path; return false; }
            if (!nvfp4_check_layer_file(f, path, experts, blob_bytes, layer, err)) { std::fclose(f); return false; }
            uint64_t base = (uint64_t) layer * experts * (uint64_t) blob_bytes;
            uint64_t off = 0;
            const uint64_t span = (uint64_t) experts * (uint64_t) blob_bytes;
            while (off < span) {
                const size_t n = (size_t) std::min<uint64_t>(chunk.size(), span - off);
                if (std::fread(chunk.data(), 1, n, f) != n) {
                    std::fclose(f);
                    err = "worker: short read on " + path;
                    return false;
                }
                if (!cuda_ok(cudaMemcpy(d_base + base + off, chunk.data(), n, cudaMemcpyHostToDevice),
                             "nvfp4 arena load", err)) {
                    std::fclose(f);
                    return false;
                }
                off += n;
            }
            // same GB10 page-cache rule as the Q2 arena (see open_q2)
            posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
            std::fclose(f);
        }
        if (want_host) {
            // CPU split tier blob source: lazy read-only mmaps.  Pages fault in from NVMe on
            // first touch (~1-2 ms per blob) and stay in the page cache afterwards; the kernel
            // can drop them cleanly under pressure (unlike a managed arena - see host_maps).
            for (int64_t layer = 0; layer < layers; ++layer) {
                const std::string path = nvfp4_layer_path(pack, layer);
                const int fd = ::open(path.c_str(), O_RDONLY);
                if (fd < 0) { err = "worker: cannot mmap-open " + path; return false; }
                const uint64_t span = (uint64_t) experts * (uint64_t) blob_bytes;
                void* m = mmap(nullptr, span, PROT_READ, MAP_PRIVATE, fd, 0);
                ::close(fd);
                if (m == MAP_FAILED) { err = "worker: mmap failed on " + path; return false; }
                host_maps.push_back((const uint8_t*) m);
                host_maps_len.push_back(span);
            }
            std::fprintf(stderr, "worker: CPU split tier blob maps ready (%d layers, page cache)\n",
                         (int) host_maps.size());
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "worker: NVFP4 cold arena %.2f GiB loaded in %.1f s (%.2f GiB/s)\n",
                     (double) total / 1073741824.0, s,
                     s > 0 ? (double) total / 1073741824.0 / s : 0.0);
        return true;
    }
    // Selftest mode: only the referenced blobs, host copies kept for the scalar reference.
    bool open_nvfp4_mini(const std::string& pack, int64_t n_expert,
                         const std::vector<std::pair<int64_t, int32_t>>& need, std::string& err,
                         bool want_host = false) {
        nvfp4 = true;
        (void) want_host;   // the mini arena's host side is mini_host already
        experts = n_expert;
        layers = 0;
        if (!nvfp4_manifest_blob_bytes(pack, blob_bytes, err)) return false;
        if (blob_bytes != strata::kernels::NVFP4_COLD_BLOB_BYTES) {
            err = "worker: manifest blob_bytes " + std::to_string(blob_bytes) +
                  " != canonical cold size " + std::to_string(strata::kernels::NVFP4_COLD_BLOB_BYTES);
            return false;
        }
        if (!cuda_ok(cudaMalloc(&d_base, (size_t) need.size() * (size_t) blob_bytes),
                     "nvfp4 mini arena", err))
            return false;
        std::vector<uint8_t> host((size_t) blob_bytes);
        for (size_t i = 0; i < need.size(); ++i) {
            const int64_t layer = need[i].first, expert = need[i].second;
            const std::string path = nvfp4_layer_path(pack, layer);
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) { err = "worker: cannot open " + path; return false; }
            const long long off = (long long) expert * (long long) blob_bytes;
            if (std::fseek(f, off, SEEK_SET) != 0 ||
                std::fread(host.data(), 1, host.size(), f) != host.size()) {
                std::fclose(f);
                err = "worker: blob read failed on " + path;
                return false;
            }
            std::fclose(f);
            if (std::memcmp(host.data(), "STRNVFP4", 8) != 0 || host[8] != 2 || host[9] != 0 ||
                host[10] != 2 || host[11] != 0) {
                err = "worker: " + path + " blob is not v2 cold NVFP4";
                return false;
            }
            const uint32_t hlayer = (uint32_t) host[12] | ((uint32_t) host[13] << 8) |
                                    ((uint32_t) host[14] << 16) | ((uint32_t) host[15] << 24);
            const uint32_t hexpert = (uint32_t) host[16] | ((uint32_t) host[17] << 8) |
                                     ((uint32_t) host[18] << 16) | ((uint32_t) host[19] << 24);
            if ((int64_t) hlayer != layer || (int64_t) hexpert != expert) {
                err = "worker: blob header says layer " + std::to_string(hlayer) + " expert " +
                      std::to_string(hexpert) + ", wanted " + std::to_string(layer) + "/" +
                      std::to_string(expert);
                return false;
            }
            if (!cuda_ok(cudaMemcpy(d_base + (size_t) i * (size_t) blob_bytes, host.data(),
                                    host.size(), cudaMemcpyHostToDevice), "nvfp4 mini load", err))
                return false;
            mini[key(layer, expert, experts)] = (int32_t) i;
            mini_host[key(layer, expert, experts)] = host;
        }
        return true;
    }
    bool open(const std::string& pack, int64_t n_layers, int64_t n_expert, bool want_nvfp4,
              std::string& err, bool want_host = false) {
        return want_nvfp4 ? open_nvfp4(pack, n_layers, n_expert, err, want_host)
                          : open_q2(pack, n_layers, n_expert, err);
    }
    const uint8_t* blob(int64_t layer, int64_t expert) const {
        const int64_t k = key(layer, expert, experts);
        if (!mini.empty()) {
            const auto it = mini.find(k);
            return it == mini.end() ? nullptr : d_base + (int64_t) it->second * blob_bytes;
        }
        return d_base + (uint64_t) k * (uint64_t) blob_bytes;
    }
    // Host-readable blob for the CPU split tier: pack-file mmap (full arena) or the mini arena's
    // host copy.  nullptr when the tier was not opened (want_host off) or the blob is absent.
    const uint8_t* blob_host(int64_t layer, int64_t expert) const {
        const int64_t k = key(layer, expert, experts);
        if (!mini_host.empty()) {
            const auto it = mini_host.find(k);
            return it == mini_host.end() ? nullptr : it->second.data();
        }
        if (layer < 0 || layer >= (int64_t) host_maps.size()) return nullptr;
        return host_maps[(size_t) layer] + (uint64_t) expert * (uint64_t) blob_bytes;
    }
    bool has_host_maps() const { return !mini_host.empty() || !host_maps.empty(); }
    void close() {
        if (d_base) cudaFree(d_base);
        d_base = nullptr;
        mini.clear();
        mini_host.clear();
        for (size_t i = 0; i < host_maps.size(); ++i)
            if (host_maps[i]) munmap((void*) host_maps[i], (size_t) host_maps_len[i]);
        host_maps.clear();
        host_maps_len.clear();
    }
    ~ExpertArena() { close(); }
};

// Device-side work buffers, allocated once (no allocation on the request path).
struct GpuCtx {
    cudaStream_t stream = nullptr;
    float* d_x = nullptr;
    uint8_t* d_q8 = nullptr;
    float* d_scales = nullptr;
    void* d_scratch = nullptr;
    float* d_out = nullptr;
    uint16_t* d_out16 = nullptr;
    unsigned long long* d_ptr = nullptr;
    int32_t* d_start = nullptr;
    int32_t* d_dst = nullptr;
    int32_t* d_tok = nullptr;
    int32_t* d_count = nullptr;
    float* h_stage = nullptr;   // pinned landing for the GPU rows when the CPU tier shares a window
    bool open(std::string& err, bool cpu_stage = false) {
        uint64_t scratch = strata::kernels::moe_hit_grouped_scratch_bytes(CAP_ENTRIES, H, FF);
#ifdef STRATA_ENABLE_NVFP4_COLD
        const uint64_t s4 = strata::kernels::moe_grouped_nvfp4_scratch_bytes(CAP_ENTRIES);
        if (s4 > scratch) scratch = s4;
#endif
        const bool ok =
            cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream", err) &&
            cuda_ok(cudaMalloc(&d_x, (size_t) CAP_TOK * H * sizeof(float)), "x", err) &&
            cuda_ok(cudaMalloc(&d_q8, (size_t) CAP_TOK * (H / 32) * 36), "q8", err) &&
            cuda_ok(cudaMalloc(&d_scales, (size_t) CAP_TOK * (H / 32) * sizeof(float)), "scales", err) &&
            cuda_ok(cudaMalloc(&d_scratch, (size_t) scratch), "scratch", err) &&
            cuda_ok(cudaMalloc(&d_out, (size_t) CAP_ENTRIES * H * sizeof(float)), "out", err) &&
            cuda_ok(cudaMalloc(&d_out16, (size_t) CAP_ENTRIES * H * sizeof(uint16_t)), "out16", err) &&
            cuda_ok(cudaMalloc(&d_ptr, sizeof(unsigned long long) * CAP_ENTRIES), "ptr", err) &&
            cuda_ok(cudaMalloc(&d_start, sizeof(int32_t) * (CAP_ENTRIES + 1)), "start", err) &&
            cuda_ok(cudaMalloc(&d_dst, sizeof(int32_t) * CAP_ENTRIES), "dst", err) &&
            cuda_ok(cudaMalloc(&d_tok, sizeof(int32_t) * CAP_ENTRIES), "tok", err) &&
            cuda_ok(cudaMalloc(&d_count, sizeof(int32_t) * 4), "count", err);
        if (ok && cpu_stage &&
            // the split tier only serves decode windows (n_tok==1, k<=32): 64 rows is 2x headroom
            !cuda_ok(cudaHostAlloc((void**) &h_stage, (size_t) 64 * H * sizeof(float),
                                   cudaHostAllocDefault), "cpu stage", err))
            return false;
        return ok;
    }
    void close() {
        if (stream) cudaStreamSynchronize(stream);
        for (void* p : {(void*) d_x, (void*) d_q8, (void*) d_scales, d_scratch, (void*) d_out,
                        (void*) d_out16, (void*) d_ptr, (void*) d_start, (void*) d_dst, (void*) d_tok,
                        (void*) d_count})
            if (p) cudaFree(p);
        if (h_stage) cudaFreeHost(h_stage);
        h_stage = nullptr;
        if (stream) cudaStreamDestroy(stream);
        stream = nullptr; d_x = nullptr; d_q8 = nullptr; d_scales = nullptr; d_scratch = nullptr;
        d_out = nullptr; d_out16 = nullptr; d_ptr = nullptr; d_start = nullptr; d_dst = nullptr;
        d_tok = nullptr; d_count = nullptr;
    }
    ~GpuCtx() { close(); }
};

// ---- CPU split tier (P5) ------------------------------------------------------
// Decode windows only (n_tok==1, fp32 activations, NVFP4 pack with host maps): a share of the window's
// entries is computed by the GB10 CPU cluster (NEON, nvfp4_cpu.h) WHILE the GPU grouped kernel
// serves the rest - the RDMA-read latency that dominates a decode window leaves the DRAM and the
// 20 cores idle, and the probe measured ~497 us/expert single-core, so one entry striped across
// the pool lands in tens of microseconds.  CPU rows are written straight into out_host (the RDMA
// scatter reads it), bypassing d_out; the wire contract and the engine are untouched.
struct CpuPool {
    int nt = 0;                  // pool threads (the request thread joins as one more worker)
    float share = 0.5f;
    bool any_ntok = false;       // selftest only: the production gate is n_tok==1 (decode windows)
    float lut[256];              // e4m3 decode table
    // job state, written by the request thread before bumping gen
    const uint8_t* blob = nullptr;
    const float* x = nullptr;
    float* out = nullptr;
    float h[FF];                 // shared swiglu intermediate (gate/up phase writes, down phase reads)
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> stop{false};
    // two spin barriers per entry: [gate/up rows -> h] then [down rows]
    std::atomic<int> bar_count{0};
    std::atomic<uint32_t> bar_phase{0};
    std::vector<std::thread> ths;
    // stats
    uint64_t entries = 0;
    double us = 0.0;

    int parts() const { return nt + 1; }   // pool threads + the request thread

    // spin hint: aarch64 YIELD is a pipeline hint, NOT sched_yield (which cost ~100 us per
    // barrier through the scheduler - measured); decode windows arrive every ~350 us so the pool
    // stays hot, and only a long idle falls back to sched_yield.
    static void spin_relax() {
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }

    void barrier() {
        const uint32_t p = bar_phase.load(std::memory_order_acquire);
        if (bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == parts()) {
            bar_count.store(0, std::memory_order_relaxed);
            bar_phase.store(p + 1, std::memory_order_release);
        } else {
            // bounded spin then yield: a pure spin here wedged the whole box when the scheduler
            // preempted a participant (load 34, sshd starved - measured 2026-10-03)
            uint32_t spins = 0;
            while (bar_phase.load(std::memory_order_acquire) == p) {
                if (++spins < 30000)
                    spin_relax();
                else
                    std::this_thread::yield();
            }
        }
    }

    // gate/up stripe -> h (silu(g)*u), barrier, down stripe.  Row stripes split exactly.
    void stripes(int part) {
        const int np = parts();
        const uint8_t* gate_w = blob + strata::kernels::NVFP4_OFF_GATE_W;
        const uint8_t* up_w = blob + strata::kernels::NVFP4_OFF_UP_W;
        const uint8_t* down_w = blob + strata::kernels::NVFP4_OFF_DOWN_W;
        const uint8_t* gate_s = blob + strata::kernels::NVFP4_OFF_GATE_S;
        const uint8_t* up_s = blob + strata::kernels::NVFP4_OFF_UP_S;
        const uint8_t* down_s = blob + strata::kernels::NVFP4_OFF_DOWN_S;
        float ws2[6];
        std::memcpy(ws2, blob + strata::kernels::NVFP4_OFF_GLOBAL, sizeof ws2);
        const int64_t g0 = FF * part / np, g1 = FF * (part + 1) / np;
        for (int64_t i = g0; i < g1; ++i) {
            const float gv = strata::kernels::NVFP4_CPU_ROW_DOT(
                gate_w + i * (H / 2), gate_s + i * (H / 16), 0.5f * ws2[0], x, H, lut);
            const float uv = strata::kernels::NVFP4_CPU_ROW_DOT(
                up_w + i * (H / 2), up_s + i * (H / 16), 0.5f * ws2[2], x, H, lut);
            h[i] = gv / (1.0f + std::exp(-gv)) * uv;
        }
        barrier();
        const int64_t d0 = H * part / np, d1 = H * (part + 1) / np;
        for (int64_t r = d0; r < d1; ++r)
            out[r] = strata::kernels::NVFP4_CPU_ROW_DOT(down_w + r * (FF / 2),
                                                        down_s + r * (FF / 16), 0.5f * ws2[4], h,
                                                        FF, lut);
        barrier();
    }

    void thread_main(int tid) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t spins = 0;
            while (gen.load(std::memory_order_acquire) == seen) {
                if (stop.load(std::memory_order_relaxed)) return;
                // ~100 us hot spin between windows, then sleep-wait (see barrier() note)
                if (++spins < 100000)
                    spin_relax();
                else
                    std::this_thread::yield();
            }
            if (stop.load(std::memory_order_relaxed)) return;
            seen = gen.load(std::memory_order_acquire);
            stripes(tid);
        }
    }

    bool open(int nthreads, float shr, std::string& err) {
        nt = nthreads;
        // oversubscription poison: the request thread joins as part nt, and a spin gang must
        // leave cores for the serve/RDMA threads and the system (measured wedge at 15+1).
        if (nt > 12) {
            std::fprintf(stderr, "worker: --cpu-threads %d clamped to 12 (spin-gang must fit cores)\n",
                         nt);
            nt = 12;
        }
        share = shr;
        strata::kernels::nvfp4_cpu_e4m3_table(lut);
        for (int t = 0; t < nt; ++t) ths.emplace_back([this, t] { thread_main(t); });
        std::fprintf(stderr, "worker: CPU split tier on: %d threads, share %.2f\n", nt, share);
        (void) err;
        return true;
    }

    // One entry on the whole pool (request thread works as part nt).  blob must be host-readable
    // (pack-file mmap or mini host copy); x is the token's fp32 row; out is the wire-order row in
    // out_host.
    void run(const uint8_t* bp, const float* xp, float* op) {
        blob = bp;
        x = xp;
        out = op;
        gen.fetch_add(1, std::memory_order_release);
        stripes(nt);
    }

    void close() {
        stop.store(true);
        gen.fetch_add(1, std::memory_order_release);
        for (auto& t : ths) t.join();
        ths.clear();
        nt = 0;
    }
    ~CpuPool() { if (nt > 0) close(); }
};

// One request, host side: grouped experts into CUDA, rows back in ENTRY order (the wire contract).
// x_q8: the slim wire (bind v2, kPushXq8) - the engine's own quantize_q8_0_scaled output, [n_tok *
// H/32 blocks of 36 B][n_tok * H/32 fp32 scales] back to back; it REPLACES the f32 x and the re-quantize.
// out16: the slim result (kPushYf16) - the rows are converted on device and land as f16, halving the write.
bool compute_request(const ExpertArena& arena, GpuCtx& g, int32_t layer, int32_t n_tok, int32_t k,
                     const float* x, const uint8_t* x_q8, const int32_t* entries, int32_t n_entries,
                     float* out_host, uint16_t* out16, std::string& err, CpuPool* cpu = nullptr,
                     float* probe_ms = nullptr, int32_t* probe_shape = nullptr) {
    if (n_tok < 1 || n_tok > CAP_TOK || k < 1 || k > 32 || n_entries < 1 || n_entries > CAP_ENTRIES ||
        n_entries > n_tok * k) {
        err = "worker: request geometry out of bounds";
        return false;
    }
    // P5 CPU split, decode windows only: the LAST cpu_n wire entries go to the CPU pool (the wire
    // order has no locality meaning), the GPU keeps [0, gpu_n) whose dst values then are already
    // compact, so the kernel/D2H path below needs no offset map - GPU rows land in g.h_stage and
    // are scattered host-side into out_host after the sync.
    int32_t cpu_n = 0;
    if (cpu != nullptr && cpu->nt > 0 && arena.nvfp4 && arena.has_host_maps() && x_q8 == nullptr &&
        out_host != nullptr && (n_tok == 1 || cpu->any_ntok) && n_entries >= 2) {
        cpu_n = (int32_t) ((float) n_entries * cpu->share + 0.5f);
        if (cpu_n > n_entries) cpu_n = n_entries;
        for (int32_t j = n_entries - cpu_n; j < n_entries; ++j) {
            const int32_t row = entries[2 * j], e = entries[2 * j + 1];
            if (row < 0 || row >= n_tok * k || e < 0 || e >= arena.experts ||
                arena.blob_host(layer, e) == nullptr) {
                err = "worker: cpu-split entry out of range";
                return false;
            }
        }
    }
    const int32_t gpu_n = n_entries - cpu_n;
    // group by expert (first-appearance order); the kernel's group ranges are CONTIGUOUS runs of the entry
    // arrays, so entries must be emitted group by group, not in request order - dst keeps the wire order
    // (entry index), so the result rows still come back in request order.  O(n): a chunk window carries up to
    // 5120 entries and a quadratic pass over them was worth milliseconds of the window's service time.
    std::vector<int32_t> gmap((size_t) arena.experts, -1), gexp, group_of((size_t) gpu_n);
    for (int32_t j = 0; j < gpu_n; ++j) {
        const int32_t row = entries[2 * j], e = entries[2 * j + 1];
        if (row < 0 || row >= n_tok * k || e < 0 || e >= arena.experts) {
            err = "worker: request entry out of range";
            return false;
        }
        if (gmap[(size_t) e] < 0) {
            gmap[(size_t) e] = (int32_t) gexp.size();
            gexp.push_back(e);
        }
        group_of[(size_t) j] = gmap[(size_t) e];
    }
    // scatter entries into expert-grouped order (contiguous span per expert)
    std::vector<int32_t> ecnt(gexp.size(), 0), h_dst((size_t) gpu_n), h_tok((size_t) gpu_n);
    for (int32_t j = 0; j < gpu_n; ++j) ++ecnt[(size_t) group_of[(size_t) j]];
    std::vector<int32_t> estart(gexp.size() + 1, 0);
    for (size_t q = 0; q < gexp.size(); ++q) estart[q + 1] = estart[q] + ecnt[q];
    std::vector<int32_t> gcur(estart.begin(), estart.end() - 1);
    for (int32_t j = 0; j < gpu_n; ++j) {
        const int32_t p = gcur[(size_t) group_of[(size_t) j]]++;
        h_dst[(size_t) p] = j;                    // entry order IS the wire order
        h_tok[(size_t) p] = entries[2 * j] / k;
    }
    // ... then split each expert's span into sub-groups of <= WGMAX entries: the grouped tile kernel serves
    // at most GMAX=8 entries per group (s2_expert_grouped.cu), and many small independent blocks keep the
    // GB10's latency-bound scheduler fed - a whole-window big group was measured 3-10x SLOWER here.
    constexpr int32_t WGMAX = 8;                 // must match GMAX in s2_expert_grouped.cu
    std::vector<unsigned long long> h_ptr;
    std::vector<int32_t> h_start;
    for (size_t q = 0; q < gexp.size(); ++q) {
        const uint8_t* bp = arena.blob(layer, gexp[q]);
        if (bp == nullptr) {
            err = "worker: expert " + std::to_string(gexp[q]) + " is not in the (mini) arena";
            return false;
        }
        for (int32_t p = estart[q]; p < estart[q + 1]; p += WGMAX) {
            h_ptr.push_back((unsigned long long) bp);
            h_start.push_back(p);
        }
    }
    h_start.push_back(gpu_n);
    const int32_t counts[4] = {(int32_t) h_ptr.size(), gpu_n, 0, 0};
    if (probe_shape) {   // STRATA_PF_PROBE: the window's geometry (gpu entries / experts / groups)
        probe_shape[0] = gpu_n;
        probe_shape[1] = (int32_t) gexp.size();
        probe_shape[2] = (int32_t) h_ptr.size();
    }
    const size_t qb = (size_t) n_tok * (size_t) (H / 32) * 36;
    const bool x_ok = x_q8 != nullptr
        ? (cuda_ok(cudaMemcpyAsync(g.d_q8, x_q8, qb, cudaMemcpyHostToDevice, g.stream), "copy q8", err) &&
           cuda_ok(cudaMemcpyAsync(g.d_scales, x_q8 + qb, (size_t) n_tok * (size_t) (H / 32) * sizeof(float),
                                   cudaMemcpyHostToDevice, g.stream), "copy scales", err))
        : cuda_ok(cudaMemcpyAsync(g.d_x, x, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice,
                                  g.stream), "copy x", err);
    if (!x_ok ||
        !cuda_ok(cudaMemcpyAsync(g.d_ptr, h_ptr.data(), h_ptr.size() * sizeof(unsigned long long),
                                 cudaMemcpyHostToDevice, g.stream), "copy ptr", err) ||
        !cuda_ok(cudaMemcpyAsync(g.d_start, h_start.data(), h_start.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, g.stream), "copy start", err) ||
        !cuda_ok(cudaMemcpyAsync(g.d_dst, h_dst.data(), h_dst.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                                 g.stream), "copy dst", err) ||
        !cuda_ok(cudaMemcpyAsync(g.d_tok, h_tok.data(), h_tok.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                                 g.stream), "copy tok", err) ||
        !cuda_ok(cudaMemcpyAsync(g.d_count, counts, sizeof counts, cudaMemcpyHostToDevice, g.stream),
                 "copy count", err))
        return false;
    if (arena.nvfp4) {
#ifdef STRATA_ENABLE_NVFP4_COLD
        // NVFP4 path: fp32 activations.  The slim wire's q8 codes dequantize through the fp32
        // scales - x[i] = qs[i] * scales[i/32] is exactly the activation value the Q2 kernels
        // multiply against, so both wire formats feed the same numbers.
        if (x_q8 != nullptr)
            strata::kernels::nvfp4_q8_act_dequant(g.d_q8, g.d_scales, g.d_x, (int64_t) n_tok * H,
                                                  g.stream);
        // the caps size the launch grids: the REAL group/entry counts, not the buffer caps (see below)
        if (!h_ptr.empty())
            strata::kernels::moe_grouped_nvfp4(g.d_ptr, g.d_start, g.d_count, g.d_dst, g.d_tok,
                                               (int64_t) h_ptr.size(), (int64_t) gpu_n, g.d_x,
                                               g.d_scratch, g.d_out, g.stream, probe_ms);
#else
        err = "worker: NVFP4 pack support not built (STRATA_ENABLE_NVFP4_COLD off)";
        return false;
#endif
    } else {
        if (x_q8 == nullptr)
            strata::kernels::quantize_q8_0_scaled(g.d_x, g.d_q8, g.d_scales, (int64_t) n_tok * H, g.stream);
        // the caps size the launch grids: the REAL group/entry counts, not the buffer caps - a decode window has
        // 1-3 rows, and launching CAP_ENTRIES (5120) blocks per window was worth ~600 us of nothing.
        strata::kernels::moe_grouped_s2(g.d_ptr, g.d_start, g.d_count, g.d_dst, g.d_tok, (int64_t) h_ptr.size(),
                                        (int64_t) gpu_n, g.d_q8, g.d_scales, g.d_scratch, g.d_out, g.stream);
    }
    if (out16 != nullptr)
        strata::kernels::f32_to_f16_bulk(g.d_out, g.d_out16, (int64_t) gpu_n * H, g.stream);
    // CPU split: out16 is never combined with it (gated above); the GPU rows land in the pinned
    // stage so the D2H cannot clobber the rows the CPU pool is writing into out_host.
    if (gpu_n > 0 && !cuda_ok(out16 != nullptr
                     ? cudaMemcpyAsync(out16, g.d_out16, (size_t) gpu_n * H * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, g.stream)
                     : cudaMemcpyAsync(cpu_n > 0 ? g.h_stage : out_host, g.d_out,
                                       (size_t) gpu_n * H * sizeof(float), cudaMemcpyDeviceToHost,
                                       g.stream), "copy out", err))
        return false;
    if (cpu_n > 0) {
        // overlap: the GPU work above is still in flight while the pool computes its entries
        const auto ct0 = std::chrono::steady_clock::now();
        for (int32_t j = gpu_n; j < n_entries; ++j) {
            const uint8_t* bp = arena.blob_host(layer, entries[2 * j + 1]);
            cpu->run(bp, x + (size_t) (entries[2 * j] / k) * (size_t) H,
                     out_host + (size_t) j * (size_t) H);
        }
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ct0).count();
        cpu->entries += (uint64_t) cpu_n;
        cpu->us += us;
        if (us > 500.0)
            std::fprintf(stderr, "worker: cpu split slow window: %d entries %.0f us\n", cpu_n, us);
    }
    if (!cuda_ok(cudaStreamSynchronize(g.stream), "sync", err))
        return false;
    if (cpu_n > 0)
        for (int32_t j = 0; j < gpu_n; ++j)
            std::memcpy(out_host + (size_t) j * (size_t) H, g.h_stage + (size_t) j * (size_t) H,
                        (size_t) H * sizeof(float));
    return true;
}

void fill_synthetic_x(float* x, int n_tok, uint64_t salt) {
    for (int t = 0; t < n_tok; ++t)
        for (int64_t i = 0; i < H; ++i) {
            const uint64_t v = (uint64_t) (i * 131 + t * 65537) ^ (salt * 2654435761u);
            x[(size_t) t * H + i] = ((float) (v % 20001) - 10000.0f) / 10000.0f;
        }
}

// Read one blob straight off the disk pack (probe/selftest reference side; the arena is device memory).
bool read_blob(const std::string& pack, int layers, int64_t layer, int64_t expert, std::vector<uint8_t>& dst,
               std::string& err) {
    (void) layers;
    const std::string path = pack + "/experts.bin";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "probe: cannot open " + path; return false; }
    dst.resize(cpuk::BLOB);
    const long long off = (long long) (layer * 512 + expert) * (long long) cpuk::BLOB;
    if (std::fseek(f, off, SEEK_SET) != 0 || std::fread(dst.data(), 1, dst.size(), f) != dst.size()) {
        std::fclose(f);
        err = "probe: blob read failed";
        return false;
    }
    std::fclose(f);
    return true;
}

int run_selftest(const Args& a, bool want_nvfp4) {
    std::string err;
    // two tokens, three entries across them, one expert shared: exercises grouping, dst and tok
    const int32_t n_tok = 2, k = 10;
    const int32_t entries[6] = {0, 137, 5, 42, 15, 137};
    ExpertArena arena;
    if (want_nvfp4) {
        // P3: the mini arena - the selftest references two blobs, there is no reason to stream
        // 68 GB to check them.
        const std::vector<std::pair<int64_t, int32_t>> need = {{3, 137}, {3, 42}};
        if (!arena.open_nvfp4_mini(a.pack, a.experts, need, err, a.cpu_threads > 0)) {
            std::fprintf(stderr, "selftest FAIL: %s\n", err.c_str());
            return 1;
        }
    } else if (!arena.open(a.pack, a.layers, a.experts, false, err)) {
        std::fprintf(stderr, "selftest FAIL: %s\n", err.c_str());
        return 1;
    }
    GpuCtx g;
    if (!g.open(err, a.cpu_threads > 0)) { std::fprintf(stderr, "selftest FAIL: %s\n", err.c_str()); return 1; }
    std::vector<float> xv((size_t) CAP_TOK * H);   // CAP_TOK is the chunk cap now: far past a stack frame
    fill_synthetic_x(xv.data(), n_tok, 7);
    std::vector<float> out(3 * H, 0.0f);
    if (!compute_request(arena, g, 3, n_tok, k, xv.data(), nullptr, entries, 3, out.data(), nullptr, err)) {
        std::fprintf(stderr, "selftest FAIL: %s\n", err.c_str());
        return 1;
    }
    bool ok = true;
    for (int j = 0; j < 3 && ok; ++j) {
        const int32_t row = entries[2 * j], e = entries[2 * j + 1];
        std::vector<float> ref(H);
        if (want_nvfp4) {
            const auto it = arena.mini_host.find(ExpertArena::key(3, e, a.experts));
            if (it == arena.mini_host.end()) {
                std::fprintf(stderr, "selftest FAIL: no host blob for expert %d\n", e);
                return 1;
            }
            strata::kernels::nvfp4_cold_expert_host_ref(it->second.data(),
                                                        xv.data() + (size_t) (row / k) * H, ref.data());
        } else {
            std::vector<uint8_t> blob;
            if (!read_blob(a.pack, a.layers, 3, e, blob, err)) { std::fprintf(stderr, "selftest FAIL: %s\n", err.c_str()); return 1; }
            strata::kernels::q2_0_expert_gpu_ref(blob.data(), xv.data() + (size_t) (row / k) * H, ref.data());
        }
        const double rel = strata::kernels::q2_ref_rel_l2(out.data() + (size_t) j * H, ref.data(), H);
        std::fprintf(stderr, "selftest: entry %d (row %d, expert %d) rel_l2 %.3e\n", j, row, e, rel);
        if (rel > kRelTol) ok = false;
    }
    if (ok && a.cpu_threads > 0 && want_nvfp4) {
        // P5 CPU split check: share 1.0 puts every entry on the NEON pool (gpu_n==0 exercises the
        // empty-launch path); rows must match both the fp64 reference and the GPU run.
        CpuPool pool;
        pool.open(a.cpu_threads, 1.0f, err);
        pool.any_ntok = true;   // the selftest window carries two tokens by design
        std::vector<float> out_cpu(3 * H, 0.0f);
        if (!compute_request(arena, g, 3, n_tok, k, xv.data(), nullptr, entries, 3, out_cpu.data(),
                             nullptr, err, &pool)) {
            std::fprintf(stderr, "selftest FAIL: cpu split: %s\n", err.c_str());
            return 1;
        }
        pool.close();
        for (int j = 0; j < 3 && ok; ++j) {
            const int32_t row = entries[2 * j], e = entries[2 * j + 1];
            const auto it = arena.mini_host.find(ExpertArena::key(3, e, a.experts));
            std::vector<float> ref(H);
            strata::kernels::nvfp4_cold_expert_host_ref(it->second.data(),
                                                        xv.data() + (size_t) (row / k) * H, ref.data());
            const double rel_ref = strata::kernels::q2_ref_rel_l2(out_cpu.data() + (size_t) j * H,
                                                                  ref.data(), H);
            const double rel_gpu = strata::kernels::q2_ref_rel_l2(out_cpu.data() + (size_t) j * H,
                                                                  out.data() + (size_t) j * H, H);
            std::fprintf(stderr, "selftest: cpu entry %d rel_l2 vs ref %.3e vs gpu %.3e\n", j,
                         rel_ref, rel_gpu);
            if (rel_ref > kRelTol || rel_gpu > 1e-4) ok = false;
        }
        std::fprintf(stderr, "worker_selftest_cpu %s\n", ok ? "PASS" : "FAIL");
    }
    std::fprintf(stderr, "worker_selftest %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int run_probe(const Args& a) {
    std::string err;
    strata::core::RdmaExpertTierConfig tc;
    tc.peer_host = a.peer;
    tc.port = (uint16_t) a.port;
    tc.device = a.device;
    tc.gid_index = a.gid;
    tc.slots = (uint32_t) a.slots;
    tc.slot_bytes = (size_t) a.slot_bytes;
    tc.wait_ms = a.wait_ms;
    tc.timeout_ms = 5000;
    RDMAExpertTier tier;
    if (!tier.open(tc, a.layers, a.experts, err)) { std::fprintf(stderr, "probe FAIL: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "probe: RDMA tier open, %d windows\n", a.windows);
    std::vector<double> us;
    bool ok = true;
    for (int w = 0; w < a.windows && ok; ++w) {
        const int32_t layer = w % a.layers;
        const int32_t n_tok = 1 + (w % 3);          // 1..3 tokens
        const int32_t k = 10;
        const int64_t n = (int64_t) n_tok * k;
        std::vector<float> x((size_t) n_tok * H);
        fill_synthetic_x(x.data(), n_tok, (uint64_t) w + 11);
        std::vector<int32_t> ids((size_t) n), kind((size_t) n, -1);
        for (int64_t i = 0; i < n; ++i) ids[(size_t) i] = (int32_t) ((i * 37 + w * 113) % a.experts);
        for (int64_t i = 0; i < n; i += 3) kind[(size_t) i] = 0;   // pretend CUDA0 owns every third row
        const auto t0 = std::chrono::steady_clock::now();
        if (!tier.begin(layer, x.data(), ids.data(), n_tok, k, kind.data(), nullptr, err)) {
            std::fprintf(stderr, "probe FAIL begin w=%d: %s\n", w, err.c_str());
            return 1;
        }
        std::vector<float> out((size_t) n * H, 0.0f);
        if (!tier.finish(out.data(), err)) {
            std::fprintf(stderr, "probe FAIL finish w=%d: %s\n", w, err.c_str());
            return 1;
        }
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        for (int64_t i = 0; i < n && ok; ++i) {
            if (kind[(size_t) i] != -1) continue;   // rows the tier did not own stay zero
            std::vector<uint8_t> blob;
            if (!read_blob(a.pack, a.layers, layer, ids[(size_t) i], blob, err)) {
                std::fprintf(stderr, "probe FAIL: %s\n", err.c_str());
                return 1;
            }
            std::vector<float> ref(H);
            strata::kernels::q2_0_expert_gpu_ref(blob.data(), x.data() + (size_t) (i / k) * H, ref.data());
            const double rel = strata::kernels::q2_ref_rel_l2(out.data() + (size_t) i * H, ref.data(), H);
            if (rel > kRelTol) {
                std::fprintf(stderr, "probe: w=%d row %lld expert %d rel_l2 %.3e\n", w, (long long) i,
                             ids[(size_t) i], rel);
                ok = false;
            }
        }
    }
    if (ok) {
        double s = 0;
        for (double v : us) s += v;
        std::fprintf(stderr, "probe: computed %lld entries, mean window %.0f us\n",
                     (long long) tier.computed(), us.empty() ? 0.0 : s / (double) us.size());
    }
    std::fprintf(stderr, "expert_probe %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int run_serve(const Args& a, bool want_nvfp4) {
    std::string err;
    ExpertArena arena;
    const bool cpu_on = a.cpu_threads > 0 && want_nvfp4;
    if (a.cpu_threads > 0 && !want_nvfp4)
        std::fprintf(stderr, "worker: --cpu-threads ignored (q2 format; the split tier is nvfp4-only)\n");
    if (!arena.open(a.pack, a.layers, a.experts, want_nvfp4, err, cpu_on)) { std::fprintf(stderr, "worker: %s\n", err.c_str()); return 1; }
    GpuCtx g;
    if (!g.open(err, cpu_on)) { std::fprintf(stderr, "worker: %s\n", err.c_str()); return 1; }
    CpuPool pool;
    CpuPool* pcpu = nullptr;
    if (cpu_on) {
        pool.open(a.cpu_threads, a.cpu_share, err);
        pcpu = &pool;
    }
    RemoteStageConfig sc;
    sc.role = RemoteStageRole::Receiver;
    sc.bind_host = a.bind;
    sc.port = (uint16_t) a.port;
    sc.device = a.device;
    sc.gid_index = a.gid;
    sc.timeout_ms = a.timeout_ms;
    sc.slots = (uint32_t) a.slots;
    sc.slot_bytes = (size_t) a.slot_bytes;
    sc.max_payload = (size_t) a.slot_bytes;
    RemoteStage stage(sc);
    if (!stage.open(err)) { std::fprintf(stderr, "worker: %s\n", err.c_str()); return 1; }
    if (!stage.is_rdma()) { std::fprintf(stderr, "worker: the data plane is not RDMA\n"); return 1; }

    // ---- PLE over RDMA (--ple-gguf): the whole IQ4_NL table resident in RAM, registered REMOTE_READ, and its
    // (addr, rkey) published in ring slot 0 BEFORE the serve loop - the engine picks it up with
    // fetch_ple_info() and then one-sided-reads rows instead of its own SSD (docs/RDMA_EXPERTS.md).
    // --ple-fp8 is the NVFP4 stack's form of the same service: the checkpoint's raw FP8 E4M3 table
    // (160 B rows, one global scale) published as a v2 record. The two are mutually exclusive.
    if (!a.ple_gguf.empty() && !a.ple_fp8.empty()) {
        std::fprintf(stderr, "worker: --ple-gguf and --ple-fp8 are mutually exclusive\n");
        return 1;
    }
    if (!a.ple_fp8.empty() && a.ple_fp8_scale <= 0.0f) {
        std::fprintf(stderr, "worker: --ple-fp8 needs --ple-fp8-scale (the table's global dequant scale)\n");
        return 1;
    }
    if (!a.ple_fp8.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        FILE* f = std::fopen(a.ple_fp8.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "worker: --ple-fp8: cannot open %s\n", a.ple_fp8.c_str()); return 1; }
        if (fseeko(f, 0, SEEK_END) != 0) { std::fprintf(stderr, "worker: --ple-fp8: size failed\n"); std::fclose(f); return 1; }
        const uint64_t bytes = (uint64_t) ftello(f);
        if (bytes == 0 || bytes % 160ull != 0) {
            std::fprintf(stderr, "worker: --ple-fp8: %llu bytes is not a whole number of 160 B rows\n",
                         (unsigned long long) bytes);
            std::fclose(f);
            return 1;
        }
        const uint64_t rows = bytes / 160ull;
        if (fseeko(f, 0, SEEK_SET) != 0) { std::fprintf(stderr, "worker: --ple-fp8: seek failed\n"); std::fclose(f); return 1; }
        uint8_t* ple = nullptr;
        if (!cuda_ok(cudaHostAlloc((void**) &ple, (size_t) bytes, cudaHostAllocDefault), "pin ple fp8", err)) {
            std::fprintf(stderr, "worker: --ple-fp8: %s\n", err.c_str());
            std::fclose(f);
            return 1;
        }
        size_t got = 0;
        while (got < bytes) {
            const size_t want = (size_t) std::min<uint64_t>(bytes - got, 256ull << 20);
            const size_t r = std::fread(ple + got, 1, want, f);
            if (r == 0) {
                std::fprintf(stderr, "worker: --ple-fp8: short read at %zu of %llu\n", got,
                             (unsigned long long) bytes);
                std::fclose(f);
                return 1;
            }
            got += r;
        }
        // Same page-cache hygiene as the arena load (51.2 GiB of dead file pages on a unified-memory host).
        posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
        std::fclose(f);
        uint32_t ple_rkey = 0;
        if (!stage.expose_region(ple, (size_t) bytes, true, false, ple_rkey, err)) {
            std::fprintf(stderr, "worker: --ple-fp8 registration: %s\n", err.c_str());
            return 1;
        }
        RDMAExpertTier::PleInfo info{};
        info.magic = RDMAExpertTier::PLE_INFO_MAGIC;
        info.version = 2;
        info.addr = (uint64_t) ple;
        info.rkey = ple_rkey;
        info.row_bytes = 160;
        info.rows = rows;
        info.format = RDMAExpertTier::kPleFmtFp8E4m3;
        info.fp8_scale = a.ple_fp8_scale;
        if (!stage.ring_publish(0, &info, sizeof info, RDMAExpertTier::PLE_INFO_SEQ, err)) {
            std::fprintf(stderr, "worker: --ple-fp8 publish: %s\n", err.c_str());
            return 1;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "worker: PLE table %llu rows x 160 B FP8 E4M3 (%.1f GiB, scale %.9g) resident + published in %.1f s\n",
                     (unsigned long long) rows, (double) bytes / (1ull << 30), (double) a.ple_fp8_scale, s);
    }
    if (!a.ple_gguf.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        strata::GgufFile gf(a.ple_gguf);
        const strata::TensorInfo* t = gf.find("per_layer_token_embd.weight");
        if (t == nullptr || t->shape.size() != 2 || t->shape[0] != 160) {
            std::fprintf(stderr, "worker: --ple-gguf: per_layer_token_embd.weight missing or misshapen\n");
            return 1;
        }
        const uint64_t rows = t->shape[1];
        const uint64_t bytes = rows * 90ull;
        const uint64_t at = gf.data_start() + t->offset;
        uint8_t* ple = nullptr;
        if (!cuda_ok(cudaHostAlloc((void**) &ple, (size_t) bytes, cudaHostAllocDefault), "pin ple", err)) {
            std::fprintf(stderr, "worker: --ple-gguf: %s\n", err.c_str());
            return 1;
        }
        FILE* f = std::fopen(a.ple_gguf.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "worker: --ple-gguf: cannot open %s\n", a.ple_gguf.c_str()); return 1; }
        if (fseeko(f, (off_t) at, SEEK_SET) != 0) {
            std::fprintf(stderr, "worker: --ple-gguf: seek failed\n"); std::fclose(f); return 1;
        }
        size_t got = 0;
        while (got < bytes) {
            const size_t want = (size_t) std::min<uint64_t>(bytes - got, 256ull << 20);
            const size_t r = std::fread(ple + got, 1, want, f);
            if (r == 0) {
                std::fprintf(stderr, "worker: --ple-gguf: short read at %zu of %llu\n", got,
                             (unsigned long long) bytes);
                std::fclose(f);
                return 1;
            }
            got += r;
        }
        // Same page-cache hygiene as the arena load: the table lives in pinned memory now, and 28.8 GiB of
        // dead file pages would squeeze every later allocation on this unified-memory host.
        posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
        std::fclose(f);
        uint32_t ple_rkey = 0;
        if (!stage.expose_region(ple, (size_t) bytes, true, false, ple_rkey, err)) {
            std::fprintf(stderr, "worker: --ple-gguf registration: %s\n", err.c_str());
            return 1;
        }
        RDMAExpertTier::PleInfo info{};
        info.magic = RDMAExpertTier::PLE_INFO_MAGIC;
        info.version = 1;
        info.addr = (uint64_t) ple;
        info.rkey = ple_rkey;
        info.row_bytes = 90;
        info.rows = rows;
        if (!stage.ring_publish(0, &info, sizeof info, RDMAExpertTier::PLE_INFO_SEQ, err)) {
            std::fprintf(stderr, "worker: --ple-gguf publish: %s\n", err.c_str());
            return 1;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "worker: PLE table %llu rows x 90 B (%.1f GiB) resident + published in %.1f s\n",
                     (unsigned long long) rows, (double) bytes / (1ull << 30), s);
    }
    std::fprintf(stderr, "worker: READY rdma=true slots=%d slot_bytes=%lld - waiting for doorbells\n",
                 a.slots, a.slot_bytes);

    // Push-channel I/O buffers: pinned and registered up front, so that after a bind record the worker can
    // one-sided-read the engine's request block and activation and RDMA-write the result rows plus the
    // doorbell flag.  The legacy pull path below does not use them.  The request block is polled by its
    // head (p_req); the entries follow in a second read that only runs when the head's seq moved.
    float* p_out = nullptr;
    float* p_x = nullptr;
    uint16_t* p_out16 = nullptr;
    RDMAExpertTier::PushReqHead* p_req = nullptr;
    int32_t* p_ent = nullptr;
    uint32_t* p_hdr = nullptr;   // W1: the v3 contiguous block's [count][pad][rows...] header staging
    if (!cuda_ok(cudaHostAlloc((void**) &p_out, (size_t) CAP_ENTRIES * H * sizeof(float),
                               cudaHostAllocDefault), "pin out", err) ||
        !cuda_ok(cudaHostAlloc((void**) &p_x, (size_t) CAP_TOK * H * sizeof(float), cudaHostAllocDefault),
                 "pin x", err) ||
        !cuda_ok(cudaHostAlloc((void**) &p_out16, (size_t) CAP_ENTRIES * H * sizeof(uint16_t),
                               cudaHostAllocDefault), "pin out16", err) ||
        !cuda_ok(cudaHostAlloc((void**) &p_req, sizeof(RDMAExpertTier::PushReqHead), cudaHostAllocDefault),
                 "pin req", err) ||
        !cuda_ok(cudaHostAlloc((void**) &p_ent, (size_t) CAP_ENTRIES * 2 * sizeof(int32_t),
                               cudaHostAllocDefault), "pin ent", err) ||
        !cuda_ok(cudaHostAlloc((void**) &p_hdr, (size_t) RDMAExpertTier::kPushYCDataOff, cudaHostAllocDefault),
                 "pin hdr", err)) {
        std::fprintf(stderr, "worker: %s\n", err.c_str());
        return 1;
    }
    uint32_t io_rkey = 0;
    if (!stage.expose_region(p_out, (size_t) CAP_ENTRIES * H * sizeof(float), false, false, io_rkey, err) ||
        !stage.expose_region(p_x, (size_t) CAP_TOK * H * sizeof(float), false, false, io_rkey, err) ||
        !stage.expose_region(p_out16, (size_t) CAP_ENTRIES * H * sizeof(uint16_t), false, false, io_rkey,
                             err) ||
        !stage.expose_region(p_req, sizeof(RDMAExpertTier::PushReqHead), false, false, io_rkey, err) ||
        !stage.expose_region(p_ent, (size_t) CAP_ENTRIES * 2 * sizeof(int32_t), false, false, io_rkey, err) ||
        !stage.expose_region(p_hdr, (size_t) RDMAExpertTier::kPushYCDataOff, false, false, io_rkey, err)) {
        std::fprintf(stderr, "worker: push buffer registration: %s\n", err.c_str());
        return 1;
    }

    std::vector<float> h_out((size_t) CAP_ENTRIES * H);
    uint64_t last = 0;
    int served = 0;
    bool push_mode = false;
    RDMAExpertTier::PushBind bind{};
    RDMAExpertTier::PushBind2 bind2{};
    bool have_bind2 = false;
    RDMAExpertTier::PushBind3 bind3{};
    bool have_bind3 = false;
    uint64_t last_push = 0;
    int served_push = 0;
    // STRATA_PF_PROBE=1: per-channel-1-window service breakdown (idle between windows / entries+activation
    // read / compute / result write), printed every 512 windows and at exit.
    const bool probe = std::getenv("STRATA_PF_PROBE") != nullptr;
    std::vector<double> pb_idle, pb_read, pb_compute, pb_write;
    // NVFP4 kernel split (gu/swiglu/down device ms) + the window's geometry, same cadence as above
    std::vector<double> pb_gu, pb_swiglu, pb_down, pb_ent, pb_exp, pb_grp;
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
        std::fprintf(stderr, "worker probe: %zu push windows | %s | %s | %s | %s\n", pb_read.size(),
                     pb_line("idle", pb_idle, b1, sizeof b1), pb_line("read", pb_read, b2, sizeof b2),
                     pb_line("compute", pb_compute, b3, sizeof b3), pb_line("write", pb_write, b4, sizeof b4));
        if (!pb_gu.empty()) {
            char c1[96], c2[96], c3[96], c4[96], c5[96], c6[96];
            std::fprintf(stderr, "worker probe: kernels | %s | %s | %s || %s | %s | %s\n",
                         pb_line("gu", pb_gu, c1, sizeof c1), pb_line("swiglu", pb_swiglu, c2, sizeof c2),
                         pb_line("down", pb_down, c3, sizeof c3), pb_line("entries", pb_ent, c4, sizeof c4),
                         pb_line("experts", pb_exp, c5, sizeof c5), pb_line("groups", pb_grp, c6, sizeof c6));
        }
    };
    while (a.max_windows <= 0 || served + served_push < a.max_windows) {
        const uint64_t seq = stage.load_local_doorbell();
        if (seq > last) {
            const uint8_t* inbox = stage.local_inbox();
            if (!inbox) { std::fprintf(stderr, "worker: no inbox\n"); return 1; }
            uint32_t magic = 0;
            std::memcpy(&magic, inbox, sizeof magic);
            if (magic == RDMAExpertTier::kMagicBind) {
                // Push mode: the engine registered its request block, activation, y_miss and doorbell flag;
                // from here the worker drives BOTH directions and the engine's host leaves the wire.
                std::memcpy(&bind, inbox, sizeof bind);
                // Bind v3 (W1): the engine advertises the contiguous result block the channel-0 write below
                // targets; a v2 peer has no such region, so it is refused here (both sides rebuilt together).
                if (bind.version != 3 || bind.n_embd != (uint32_t) H || bind.k_cap == 0 ||
                    bind.n_layers != (uint32_t) a.layers || !bind.req_addr || !bind.req_rkey || !bind.x_addr ||
                    !bind.x_rkey || !bind.y_addr || !bind.y_rkey || !bind.flag_addr || !bind.flag_rkey ||
                    !bind.yc_addr || !bind.yc_rkey || bind.yc_rows < bind.k_cap ||
                    bind.expert_format != (want_nvfp4 ? 1u : 0u)) {
                    std::fprintf(stderr, "worker: bad push bind record (version, geometry or expert format "
                                         "mismatch: bind=%u, worker=%u) - refusing, fail-closed\n",
                                 bind.expert_format, want_nvfp4 ? 1u : 0u);
                    return 1;
                }
                push_mode = true;
                last_push = 0;
                last = seq;
                std::fprintf(stderr, "worker: push mode bound (y_miss %u rows, %u layers) - the engine's host "
                                     "verbs are off the wire\n", bind.k_cap, bind.n_layers);
                continue;
            }
            if (magic == RDMAExpertTier::kMagicBind2) {
                // Push channel 1: the prefill path's activation and result blocks.  Kept alongside the RXE2
                // bind; the request block's channel field picks the region pair.  Bind v2 declares the wire
                // dtypes (the engine owns the buffers); anything else is refused, never misread.
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
                    bind2.x2_row_bytes != xrb || bind2.y2_row_bytes != yrb) {
                    std::fprintf(stderr, "worker: bad push channel-1 bind record - refusing, fail-closed\n");
                    return 1;
                }
                have_bind2 = true;
                last = seq;
                std::fprintf(stderr, "worker: push channel 1 bound (x %u tokens %s, y %u rows %s) - prefill "
                                     "windows are one window per layer per chunk\n", bind2.t_cap,
                             bind2.x2_dtype == RDMAExpertTier::kPushXq8 ? "q8_0" : "f32", bind2.y2_rows,
                             bind2.y2_dtype == RDMAExpertTier::kPushYf16 ? "f16" : "f32");
                continue;
            }
            if (magic == RDMAExpertTier::kMagicBind3) {
                // Push channel 2: the verify window's activation and result blocks plus a doorbell flag of
                // its own.  All rows are f32 on this channel; a request carries x_off/y_off, the offset of
                // the window's segment (a split window's two groups are two requests).
                std::memcpy(&bind3, inbox, sizeof bind3);
                if (bind3.version != 1 || bind3.t3_cap == 0 || bind3.y3_rows == 0 || !bind3.x3_addr ||
                    !bind3.x3_rkey || !bind3.y3_addr || !bind3.y3_rkey || !bind3.flag3_addr ||
                    !bind3.flag3_rkey) {
                    std::fprintf(stderr, "worker: bad push channel-2 bind record - refusing, fail-closed\n");
                    return 1;
                }
                have_bind3 = true;
                last = seq;
                std::fprintf(stderr, "worker: push channel 2 bound (x %u tokens, y %u rows, own flag) - "
                                     "verify windows are push windows\n", bind3.t3_cap, bind3.y3_rows);
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            RDMAExpertTier::ReqHeader h;
            std::memcpy(&h, inbox, sizeof h);
            if (h.magic != RDMAExpertTier::kMagic || h.version != RDMAExpertTier::kVersion ||
                h.layer < 0 || h.layer >= a.layers || h.n_tok < 1 || h.n_tok > CAP_TOK ||
                h.k < 1 || h.k > 32 || h.n_entries < 1 || h.n_entries > h.n_tok * h.k) {
                std::fprintf(stderr, "worker: bad request header (magic %08x) - refusing, fail-closed\n", h.magic);
                return 1;
            }
            const size_t x_bytes = (size_t) h.n_tok * H * sizeof(float);
            const size_t ent_bytes = (size_t) h.n_entries * 2 * sizeof(int32_t);
            if (sizeof h + x_bytes + ent_bytes > (size_t) a.slot_bytes) {
                std::fprintf(stderr, "worker: request exceeds the inbox\n");
                return 1;
            }
            const float* x = reinterpret_cast<const float*>(inbox + sizeof h);
            const int32_t* entries = reinterpret_cast<const int32_t*>(inbox + sizeof h + x_bytes);
            if (!compute_request(arena, g, h.layer, h.n_tok, h.k, x, nullptr, entries, h.n_entries,
                                 h_out.data(), nullptr, err, pcpu)) {
                std::fprintf(stderr, "worker: compute failed: %s\n", err.c_str());
                return 1;
            }
            const uint32_t slot = (uint32_t) ((seq - 1) % (uint64_t) a.slots);
            if (!stage.ring_publish(slot, h_out.data(), (size_t) h.n_entries * H * sizeof(float), seq, err)) {
                std::fprintf(stderr, "worker: publish failed: %s\n", err.c_str());
                return 1;
            }
            last = seq;
            ++served;
            if (a.verbose || served <= 3 || (served % 480) == 0) {
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "worker: seq=%" PRIu64 " layer=%d tok=%d entries=%d %.0f us\n", seq, h.layer,
                             h.n_tok, h.n_entries, us);
            }
            continue;
        }
        if (push_mode) {
            // Poll only the 32-byte head of the engine's request block: one small one-sided read per
            // iteration, no peer involvement.  The entries follow in a second read once the seq moved (the
            // publisher stored them before the seq, release).
            if (!stage.rdma_read_remote(bind.req_addr, bind.req_rkey, p_req,
                                        sizeof(RDMAExpertTier::PushReqHead), err)) {
                std::fprintf(stderr, "worker: push request read failed: %s\n", err.c_str());
                return 1;
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
            if (layer < 0 || layer >= a.layers || k < 1 || k > 32 || ne < 0 || ne > n_tok * k ||
                ne > CAP_ENTRIES) {
                std::fprintf(stderr, "worker: bad push request header - refusing, fail-closed\n");
                return 1;
            }
            uint64_t x_addr = 0, y_addr = 0, flag_addr = bind.flag_addr;
            uint32_t x_rkey = 0, y_rkey = 0, flag_rkey = bind.flag_rkey, flag_val = 0;
            if (channel == 0) {
                // decode: one token, rows written CONTIGUOUS in entry order into the v3 y_contig block
                // (the engine's graph scatters them on the device), flag = layer + 1
                if (n_tok != 1 || (uint32_t) k > bind.k_cap) {
                    std::fprintf(stderr, "worker: bad decode push request - refusing, fail-closed\n");
                    return 1;
                }
                x_addr = bind.x_addr; x_rkey = bind.x_rkey;
                y_addr = bind.yc_addr; y_rkey = bind.yc_rkey;
                flag_val = (uint32_t) (layer + 1);
            } else if (channel == 1 && have_bind2) {
                // prefill: a whole chunk, rows contiguous in entry order, flag = the request's seq
                if ((uint32_t) n_tok > bind2.t_cap || (uint32_t) ne > bind2.y2_rows) {
                    std::fprintf(stderr, "worker: push channel-1 request beyond the bound buffers - refusing\n");
                    return 1;
                }
                x_addr = bind2.x2_addr; x_rkey = bind2.x2_rkey;
                y_addr = bind2.y2_addr; y_rkey = bind2.y2_rkey;
                flag_val = (uint32_t) rseq;
            } else if (channel == 2 && have_bind3) {
                // verify: the window's segment (x3 + x_off tokens), rows contiguous in entry order at
                // y3 + y_off, the window's own flag = the request's seq.  All f32.
                if (x_off < 0 || y_off < 0 || n_tok < 1 || (uint32_t) (x_off + n_tok) > bind3.t3_cap ||
                    (uint32_t) (y_off + ne) > bind3.y3_rows) {
                    std::fprintf(stderr, "worker: push channel-2 request beyond the bound buffers - refusing\n");
                    return 1;
                }
                x_addr = bind3.x3_addr + (uint64_t) x_off * (uint64_t) (H * sizeof(float));
                x_rkey = bind3.x3_rkey;
                y_addr = bind3.y3_addr + (uint64_t) y_off * (uint64_t) (H * sizeof(float));
                y_rkey = bind3.y3_rkey;
                flag_addr = bind3.flag3_addr; flag_rkey = bind3.flag3_rkey;
                flag_val = (uint32_t) rseq;
            } else {
                std::fprintf(stderr, "worker: push request on an unbound channel - refusing, fail-closed\n");
                return 1;
            }
            const bool x_q8 = channel == 1 && bind2.x2_dtype == RDMAExpertTier::kPushXq8;
            const bool y_f16 = channel == 1 && bind2.y2_dtype == RDMAExpertTier::kPushYf16;
            RemoteStage::RdmaRowSeg segs[80];   // a verify window's worst case is kVerifyMaxT * k = 80 rows
            RemoteStage::RdmaRowSeg one{y_f16 ? (const void*) p_out16 : (const void*) p_out, y_addr,
                                        (uint32_t) ((size_t) ne * (channel == 1 ? bind2.y2_row_bytes
                                                                                : (uint32_t) (H * sizeof(float))))};
            std::chrono::steady_clock::time_point t1 = t0, t2 = t0;
            float pb_ms[3] = {-1.0f, -1.0f, -1.0f};   // kernel split; <0 = not an NVFP4 window
            int32_t pb_sh[3] = {0, 0, 0};             // entries / experts / groups
            if (ne > 0) {
                if (!stage.rdma_read_remote(bind.req_addr + offsetof(RDMAExpertTier::PushReq, entries),
                                            bind.req_rkey, p_ent, (size_t) ne * 2 * sizeof(int32_t), err)) {
                    std::fprintf(stderr, "worker: push entries read failed: %s\n", err.c_str());
                    return 1;
                }
                for (int32_t j = 0; j < ne; ++j) {
                    const int32_t row = p_ent[2 * j];
                    if (row < 0 || row >= n_tok * k || p_ent[2 * j + 1] < 0 ||
                        p_ent[2 * j + 1] >= a.experts) {
                        std::fprintf(stderr, "worker: push request entry out of range - refusing\n");
                        return 1;
                    }
                }
                if (!stage.rdma_read_remote(x_addr, x_rkey, p_x,
                                            (size_t) n_tok * (channel == 1 ? bind2.x2_row_bytes
                                                                           : (uint32_t) (H * sizeof(float))),
                                            err)) {
                    std::fprintf(stderr, "worker: push activation read failed: %s\n", err.c_str());
                    return 1;
                }
                if (probe) t1 = std::chrono::steady_clock::now();
                if (!compute_request(arena, g, layer, n_tok, k,
                                     x_q8 ? nullptr : (const float*) p_x,
                                     x_q8 ? (const uint8_t*) p_x : nullptr, p_ent, ne,
                                     y_f16 ? nullptr : p_out, y_f16 ? p_out16 : nullptr, err, pcpu,
                                     probe ? pb_ms : nullptr, probe ? pb_sh : nullptr)) {
                    std::fprintf(stderr, "worker: push compute failed: %s\n", err.c_str());
                    return 1;
                }
                if (probe) t2 = std::chrono::steady_clock::now();
                if (channel == 0) {
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
                    // verify scatters the same way inside the window's segment (y3 + y_off + row) - the
                    // dispatch expects each row AT its routing position, only channel 1's consumer reads
                    // them contiguous in entry order
                    for (int32_t j = 0; j < ne; ++j)
                        segs[j] = RemoteStage::RdmaRowSeg{
                            p_out + (size_t) j * H,
                            y_addr + (uint64_t) p_ent[2 * j] * (uint64_t) H * sizeof(float),
                            (uint32_t) ((size_t) H * sizeof(float))};
            }
            if (ne == 0 && channel == 0) {
                // W1: an empty decode window still rewrites the header (count = 0), or the engine's scatter
                // would replay the previous layer's rows.
                p_hdr[0] = 0;
                p_hdr[1] = 0;
                segs[0] = RemoteStage::RdmaRowSeg{p_hdr, y_addr, 8};
            }
            // Rows first, flag last, one ordered chain: the consumer passes only once every row it will read
            // has landed.  Channel 0's flag is the layer's ring sequence (layer + 1); channels 1/2's is the
            // request's seq (channel 2 raises the window's OWN flag, not the token graph's).
            const RemoteStage::RdmaRowSeg* sp = channel == 1 ? &one : segs;
            const size_t ns = channel == 1 ? (ne > 0 ? 1 : 0)
                                           : (channel == 0 ? (ne > 0 ? 2 : 1) : (size_t) ne);
            if (!stage.rdma_write_scatter(sp, ns, y_rkey, flag_addr, flag_rkey, flag_val, err)) {
                std::fprintf(stderr, "worker: push result write failed: %s\n", err.c_str());
                return 1;
            }
            last_push = rseq;
            ++served_push;
            if (probe) {
                const auto t3 = std::chrono::steady_clock::now();
                pb_idle.push_back(std::chrono::duration<double, std::milli>(t0 - pb_prev).count());
                pb_read.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                pb_compute.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
                pb_write.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
                if (pb_ms[0] >= 0.0f) {
                    pb_gu.push_back(pb_ms[0]);
                    pb_swiglu.push_back(pb_ms[1]);
                    pb_down.push_back(pb_ms[2]);
                    pb_ent.push_back(pb_sh[0]);
                    pb_exp.push_back(pb_sh[1]);
                    pb_grp.push_back(pb_sh[2]);
                }
                pb_prev = t3;
                if (pb_read.size() % 512 == 0) pb_dump();
            }
            if (a.verbose || served_push <= 3 || (served_push % 480) == 0) {
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "worker: push seq=%" PRIu64 " ch=%d layer=%d tok=%d entries=%d %.0f us\n",
                             rseq, channel, layer, n_tok, ne, us);
            }
            continue;
        }
        std::this_thread::yield();
    }
    pb_dump();
    std::fprintf(stderr, "worker: served %d windows + %d push windows, done\n", served, served_push);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (s == "--pack") a.pack = next("--pack");
        else if (s == "--peer") a.peer = next("--peer");
        else if (s == "--ple-gguf") a.ple_gguf = next("--ple-gguf");
        else if (s == "--ple-fp8") a.ple_fp8 = next("--ple-fp8");
        else if (s == "--ple-fp8-scale") a.ple_fp8_scale = std::strtof(next("--ple-fp8-scale").c_str(), nullptr);
        else if (s == "--bind") a.bind = next("--bind");
        else if (s == "--device") a.device = next("--device");
        else if (s == "--port") a.port = std::atoi(next("--port").c_str());
        else if (s == "--gid") a.gid = std::atoi(next("--gid").c_str());
        else if (s == "--slots") a.slots = std::atoi(next("--slots").c_str());
        else if (s == "--slot-bytes") a.slot_bytes = std::atoll(next("--slot-bytes").c_str());
        else if (s == "--layers") a.layers = std::atoi(next("--layers").c_str());
        else if (s == "--experts") a.experts = std::atoi(next("--experts").c_str());
        else if (s == "--timeout-ms") a.timeout_ms = std::atoi(next("--timeout-ms").c_str());
        else if (s == "--wait-ms") a.wait_ms = std::atoi(next("--wait-ms").c_str());
        else if (s == "--max-windows") a.max_windows = std::atoi(next("--max-windows").c_str());
        else if (s == "--windows") a.windows = std::atoi(next("--windows").c_str());
        else if (s == "--selftest") a.selftest = true;
        else if (s == "--probe") a.probe = true;
        else if (s == "--verbose") a.verbose = true;
        else if (s == "--format") a.format = next("--format");
        else if (s == "--cpu-threads") a.cpu_threads = std::atoi(next("--cpu-threads").c_str());
        else if (s == "--cpu-share") a.cpu_share = std::strtof(next("--cpu-share").c_str(), nullptr);
        else { usage(); return 2; }
    }
    if (a.cpu_threads < 0 || a.cpu_threads > 64 || a.cpu_share < 0.0f || a.cpu_share > 1.0f) {
        std::fprintf(stderr, "--cpu-threads must be 0..64 and --cpu-share 0..1\n");
        return 2;
    }
    if (a.pack.empty()) { usage(); return 2; }
    // Format resolution: explicit --format wins; auto sniffs the NVFP4 manifest.  Anything else
    // is a hard error - a worker that guesses the wrong format would compute garbage, so fail here.
    bool want_nvfp4;
    if (a.format == "nvfp4") want_nvfp4 = true;
    else if (a.format == "q2") want_nvfp4 = false;
    else if (a.format == "auto") {
        FILE* f = std::fopen((a.pack + "/manifest-cold.json").c_str(), "rb");
        want_nvfp4 = f != nullptr;
        if (f) std::fclose(f);
    } else {
        std::fprintf(stderr, "--format must be q2, nvfp4 or auto (got '%s')\n", a.format.c_str());
        return 2;
    }
    std::fprintf(stderr, "worker: pack format %s\n", want_nvfp4 ? "nvfp4" : "q2");
    if (a.selftest) return run_selftest(a, want_nvfp4);
    if (a.probe) {
        if (a.peer.empty()) { usage(); return 2; }
        if (want_nvfp4) {
            // the probe's reference side reads Q2_0 experts.bin blobs; P3's reference lives in the
            // selftest/harness oracle instead
            std::fprintf(stderr, "worker: --probe is q2-only in P3 (use --selftest)\n");
            return 2;
        }
        return run_probe(a);
    }
    return run_serve(a, want_nvfp4);
}
