// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the header first.
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <cstring>

namespace strata::core {

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    (void) version;   // a future format bumps it; the layout check above is what protects this reader today
    return true;
}

ExpertCache::~ExpertCache() { close(); }

#if defined(STRATA_USE_HIP)
bool ExpertCache::ensure_blocking_staging(std::size_t bytes, std::string& err) {
    if (bytes <= blocking_staging_bytes_) return true;
    void* next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        err = std::string("ExpertCache: HIP blocking staging allocation: ") + cudaGetErrorString(status);
        return false;
    }
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = static_cast<uint8_t*>(next);
    blocking_staging_bytes_ = bytes;
    return true;
}
#endif

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    // On unified-memory targets (DGX Spark GB10) cudaMemGetInfo counts reclaimable page cache as
    // used, so this pre-check under-reports what cudaMalloc can actually get (the kernel reclaims
    // cache under pressure).  STRATA_SKIP_VRAM_PRECHECK=1 lets the real allocation below decide.
    const bool skip_precheck = std::getenv("STRATA_SKIP_VRAM_PRECHECK") != nullptr;
    if (!skip_precheck && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (long long) n_slots, (long long) blob_bytes, (double) want / 1073741824.0,
                          (double) free_b / 1073741824.0, (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

    if (cudaMalloc((void**) &base_, (size_t) want) != cudaSuccess) {
        base_ = nullptr;
        char buf[256];
        std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
                      (double) want / 1073741824.0, cudaGetErrorString(cudaGetLastError()));
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {
        err = "ExpertCache: cudaMemset of the slot arena failed";
        close();
        return false;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    // R4.2g: each layer starts at the bottom of its own range.  Built here rather than lazily so `admit`
    // stays allocation-free on the token path.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it
        off[i + 1] = off[i] + ((uint64_t) slot_bytes[i] + 255) / 256 * 256;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    // one allocation of the summed size, through the uniform path's checks: n "slots" of 1 byte
    if (!open((int64_t) off.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    blob_ = mx;
    off_ = std::move(off);
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    return true;
}

void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    planar_planes_ = 0;
    for (int p = 0; p < kMaxPlanes; ++p) { planar_bytes_[p] = 0; planar_region_[p] = 0; }
    if (base_ != nullptr) {
        cudaFree(base_);
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    // R4.2g: THE PER-LAYER PATH.  Same "no eviction" rule, but the ceiling is this layer's own range rather
    // than one counter shared by all 48 - which is what confined the measured hit rate to 2.97%.
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;   // this layer's quota is full
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;   // full: no eviction, deliberately - see the header
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (planar()) return nullptr;   // planar cache: use device_plane_slot
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (planar()) return nullptr;
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::open_planar(int64_t n_slots, int64_t n_layers, int64_t n_expert,
                              const int64_t* plane_bytes, int n_planes, std::string& err) {
    if (plane_bytes == nullptr || n_planes < 1 || n_planes > kMaxPlanes) {
        err = "ExpertCache::open_planar: plane count outside 1.." + std::to_string(kMaxPlanes);
        return false;
    }
    int64_t total = 0;
    for (int p = 0; p < n_planes; ++p) {
        if (plane_bytes[p] <= 0) {
            err = "ExpertCache::open_planar: plane " + std::to_string(p) + " has no bytes";
            return false;
        }
        if (plane_bytes[p] > std::numeric_limits<int64_t>::max() - total) {
            err = "ExpertCache::open_planar: the plane sizes overflow";
            return false;
        }
        total += plane_bytes[p];
    }
    // The flat open() does the VRAM check, the allocation, the zeroing and the residency tables; the
    // planar layout only re-interprets WHERE a plane lives inside that one allocation.
    if (!open(n_slots, n_layers, n_expert, total, err)) return false;
    planar_planes_ = n_planes;
    uint64_t region = 0;
    for (int p = 0; p < n_planes; ++p) {
        planar_bytes_[p] = plane_bytes[p];
        planar_region_[p] = region;
        region += (uint64_t) n_slots * (uint64_t) plane_bytes[p];
    }
    return true;
}

uint8_t* ExpertCache::device_plane_slot(int p, int32_t slot) {
    if (!planar() || p < 0 || p >= planar_planes_ || slot < 0 || slot >= slots_) return nullptr;
    return base_ + planar_region_[p] + (size_t) slot * (size_t) planar_bytes_[p];
}

const uint8_t* ExpertCache::device_plane_slot(int p, int32_t slot) const {
    if (!planar() || p < 0 || p >= planar_planes_ || slot < 0 || slot >= slots_) return nullptr;
    return base_ + planar_region_[p] + (size_t) slot * (size_t) planar_bytes_[p];
}

bool ExpertCache::fill_slot_planar(int32_t slot, const uint8_t* const* plane_src, void* stream,
                                   std::string& err) {
    if (!planar()) {
        err = "ExpertCache::fill_slot_planar: the cache is not planar";
        return false;
    }
    if (plane_src == nullptr || slot < 0 || slot >= slots_) {
        err = "ExpertCache::fill_slot_planar: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    for (int p = 0; p < planar_planes_; ++p) {
        if (plane_src[p] == nullptr) {
            err = "ExpertCache::fill_slot_planar: plane " + std::to_string(p) + " is null";
            return false;
        }
        const cudaError_t e = cudaMemcpyAsync(device_plane_slot(p, slot), plane_src[p],
                                              (size_t) planar_bytes_[p], cudaMemcpyHostToDevice,
                                              (cudaStream_t) stream);
        if (e != cudaSuccess) {
            err = std::string("ExpertCache::fill_slot_planar: ") + cudaGetErrorString(e);
            return false;
        }
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice,
                                          (cudaStream_t) stream);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
#if defined(STRATA_USE_HIP)
    // Bound HIP's pageable-source staging to one expert instead of repeatedly
    // registering regions of the mmap. The blocking copy completes before reuse.
    if (!blocking_staging_ || n > blocking_staging_bytes_) {
        err = "ExpertCache::fill_slot_blocking: HIP staging buffer is too small";
        return false;
    }
    std::memcpy(blocking_staging_, host_blob, n);
    const cudaError_t e = cudaMemcpy(dst, blocking_staging_, n, cudaMemcpyHostToDevice);
#else
    const cudaError_t e = cudaMemcpy(dst, host_blob, n, cudaMemcpyHostToDevice);
#endif
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice, (cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_queued: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::sync_queued(std::string& err) {
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::sync_queued: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const cudaError_t e = cudaMemcpy(got.data(), src, (size_t) nb, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::verify_slot: ") + cudaGetErrorString(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
