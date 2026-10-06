// include/strata/core/expert_cache.hpp - R4: the VRAM-resident expert tier.
//
// **WHY THIS EXISTS, IN ONE PARAGRAPH.** The CPU expert pool is the engine's largest single cost: measured,
// **663.6 MB of expert bytes per token at ~40 GB/s = 16.2 ms**, on a token of ~53 ms, and `test_expert_pool`
// says the pipeline hides **1.055 ms of 19.035** - the CPU work is 96% exposed because the residual chain is
// strictly serial. The only way past it is to stop reading those bytes: put some of the 48 x 512 experts in
// VRAM and let the GPU compute them, so the CPU reads only the misses.
//
// **THE DESIGN IS SIZED AGAINST MEASUREMENTS, NOT AGAINST THE PLAN'S ASSUMPTIONS.**
//
//   * `h_expert` at 4,105 slots, **ten-fold leave-one-out: 0.6447 [0.6110, 0.6750]** (`tools/routing_kfold.py`).
//     The 0.6573 the README published is in-sample and 0.4720 measured a single-prompt CORPUS, not the metric.
//   * VRAM, measured on a live run rather than at startup: **6414 MiB free** with the engine at steady state,
//     against **5413 MiB** for 4,105 slots. It fits, with ~1 GB spare. `strata-device`'s planner agrees and says
//     FITS, but the planner runs before the dense arena exists, so the live figure is the one that counts.
//   * `h_layer` held-out is **0.0456** - only 4.6% of (layer, token) pairs have all ten experts resident - so
//     this cannot be a per-layer grouped kernel and the three-way split is not an optimisation.
//
// **WHAT THIS FILE IS AND IS NOT, TODAY.** It is the SLOT STORAGE and the RESIDENCY TABLE: it allocates the
// VRAM, fills it from the host arena, and answers `(layer, expert) -> slot or -1`. It does **not** yet compute
// anything - `moe_hit_grouped_s2` does not exist and the hit/miss split is not wired into the graph - so with
// the cache on and nothing consuming it, **the engine is slower by the fill cost and faster by nothing.**
// That is stated here rather than discovered from a benchmark, and it is why `--expert-cache` defaults to 0.
//
// The order of work is `Memory/R4-design-note.md` §7: slots and residency first, then the split, then the
// kernel. This is that first step, and the step it unblocks is the one that can be measured.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Where one expert's VRAM lives, or -1.  `int32_t` and not a bool because the slot index is what the kernel
/// needs, and a table that has to be re-derived from a bitmask is a table that will be re-derived differently
/// in the two places that use it.
inline constexpr int32_t kNotResident = -1;

/// **THE STATIC RESIDENCY PLAN, READ BACK FROM THE FILE `tools/make_profile.py` WROTE.**
///
/// `profile.bin` is `STRP`, then `version, n_layers, n_expert, slots, n_ranked`, then `n_ranked` `(layer,
/// expert)` pairs ranked by descending routing frequency, then a `n_layers x n_expert` frequency table.
///
/// **WHY A PROFILE AND NOT THE COMPULSORY-MISS POLICY THIS ENGINE ALREADY HAD.**  Compulsory-miss fills with
/// the first `S` DISTINCT experts the run happens to route, which on this workload is whatever the prompt
/// touched first - it measured `h = 0.4864`.  A profile ranked by frequency over a whole trace is what the
/// plan's `0.6447` refers to, and the CPU's half of the drain is linear in `h`, so the gap is worth about
/// 4 ms of drain.  The point of reading it from a file is that a profile built on ONE prompt can be scored on
/// ANOTHER, which is the only way to know whether it transfers.
///
/// The frequency table is not returned: it is the profile's own working, and the engine measures `h` by
/// running rather than by re-deriving what the file claims.
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    /// Allocates `n_slots * blob_bytes` of DEVICE memory and a `n_layers * n_expert` residency table, and
    /// **CHECKS THE ALLOCATION AGAINST WHAT THE CARD ACTUALLY HAS** rather than assuming the planner was right.
    /// A cache that silently allocates less than it was asked for would report a hit rate for slots it does
    /// not have, which is the shape of error this project keeps paying for.
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err);
    /// Plan v0.3 P6: slots of the given sizes, back to back (a native pack's blobs differ per layer, and a
    /// profile-filled tier never moves an expert to another layer's slot, so each slot keeps its first size).
    bool open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert, std::string& err);
    /// Byte offset of each slot in the arena (null for uniform slots).
    const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
    void close();

    /// **P2 (NVFP4): THE SAME RESIDENCY TABLE OVER PLANAR STORAGE.**  The marlin kernel hard-requires each
    /// weight/scale array contiguous across experts (`b_q_weight.is_contiguous()`), which a per-slot blob
    /// arena cannot provide: expert `e`'s w13 must sit at `base + e * w13_bytes`, not inside a 2.7 MB blob.
    /// `open_planar` allocates the same total VRAM as `open(n_slots, ..., sum(plane_bytes))` but lays it
    /// out as `n_planes` back-to-back regions, region `p` holding every slot's plane `p` contiguously.
    /// Residency, admission and the per-layer policy are byte-for-byte the flat cache's; only storage
    /// addressing changes, and the flat `fill_slot`/`verify_slot`/`device_slot` refuse a planar cache.
    static constexpr int kMaxPlanes = 4;
    bool open_planar(int64_t n_slots, int64_t n_layers, int64_t n_expert, const int64_t* plane_bytes,
                     int n_planes, std::string& err);
    bool planar() const { return planar_planes_ > 0; }
    int plane_count() const { return planar_planes_; }
    int64_t plane_bytes(int p) const { return (p >= 0 && p < planar_planes_) ? planar_bytes_[p] : 0; }
    /// The device address of `slot`'s plane `p` (null when out of range or not planar).
    uint8_t* device_plane_slot(int p, int32_t slot);
    const uint8_t* device_plane_slot(int p, int32_t slot) const;
    /// Copies the `plane_count()` host planes into `slot`, one async copy per plane on `stream`, in plane
    /// order.  The copies are stream-ordered before any kernel that reads the slot, exactly like
    /// `fill_slot`'s single copy.  `plane_src` entries must each hold `plane_bytes(p)` readable bytes.
    bool fill_slot_planar(int32_t slot, const uint8_t* const* plane_src, void* stream, std::string& err);

    bool valid() const { return base_ != nullptr; }
    int64_t slots() const { return slots_; }
    /// Slots actually claimed.  Not the same as `slots()` - the cache does not evict, so a run that routes
    /// fewer distinct experts than there are slots leaves the rest empty.
    int64_t resident() const { return per_layer_ ? admitted_ : next_free_; }
    int64_t bytes() const { return off_.empty() ? slots_ * blob_ : (int64_t) off_.back(); }
    double gib() const { return (double) bytes() / 1073741824.0; }

    /// `(layer, expert)` -> slot index, or `kNotResident`.  Bounds-checked: a bad layer or expert returns
    /// `kNotResident` rather than reading whatever is adjacent in the table.
    int32_t slot_of(int64_t layer, int64_t expert) const;
    /// Claims the next free slot for `(layer, expert)`.  Returns the slot, or `kNotResident` when the cache is
    /// full - **it never evicts**, because eviction policy is a measured question (`R4.1`'s LFU-decay vs LRU
    /// sweep) and a placeholder policy would set the hit rate that everything downstream is then sized against.
    int32_t admit(int64_t layer, int64_t expert);

    /// **R4.2g: GIVE EACH LAYER ITS OWN SLOTS.  ROUND 328 MEASURED WHY THE GLOBAL FORM CANNOT WORK.**
    ///
    /// `admit`'s original policy hands out slots in ARRIVAL ORDER from one counter shared by every layer
    /// (`if (next_free_ >= slots_) return kNotResident;`).  A position looks at 10 experts in each of 48
    /// layers, so the first `n_slots` distinct (layer, expert) pairs are **about 26 layers of POSITION 0** -
    /// the cache fills entirely inside the first position it ever sees and never changes again, so hits are
    /// confined to those few layers for the rest of the run.
    ///
    /// Measured, `--expert-cache 256`: **1781 of 60000 = 2.97%**.  `Memory/cache_allocation.py` scores the
    /// same run's routing and says **8 slots per layer is worth 21.4%** and 64 is worth **70.4%**.  The cache
    /// was never short of capacity - it was handing all of it to two layers.
    ///
    /// With this on, layer `l` may only use slots `[l*q, (l+1)*q)` where `q = slots()/n_layers`, the last
    /// layer taking the remainder.  **EVICTION IS STILL ABSENT** - the property the comment above calls
    /// deliberate is preserved exactly; only WHICH slot a layer may take changes.  Default off, so every
    /// caller that predates this flag is bit-for-bit unaffected.
    void set_per_layer_admission(bool on) { per_layer_ = on; }
    bool per_layer_admission() const { return per_layer_; }
    /// The slot range layer `l` may admit into under per-layer admission.  Exposed so a test can check it.
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    /// The device address of one slot.
    uint8_t* device_slot(int32_t slot);
    const uint8_t* device_slot(int32_t slot) const;

    /// Copies `(layer, expert)`'s blob from `host_blob` into `slot` on `stream`.  Asynchronous: the caller
    /// orders it.  Returns false if the indices are out of range rather than reading past the arena.
    /// `bytes` (plan v0.3 P6): the blob's own size when it is smaller than the slot (a native pack); 0 = the slot.
    bool fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes = 0);

    /// **THE SAME COPY, BUT BLOCKING, AND THE PROFILE FILL NEEDS IT.**  `fill_slot` is asynchronous because on
    /// the token path the fill and the kernel that reads it are ordered by one stream and waiting would be a
    /// synchronisation per layer.  At STARTUP there is no such ordering to lean on: the source is pageable host
    /// memory, `verify_slot` reads the slot back on the LEGACY stream, and a legacy-stream copy is not ordered
    /// against a `cudaStreamNonBlocking` one.  The first version used the async form and `verify_slot` refused
    /// the whole run with "slot 0 differs from the arena at byte 0" - which is the check doing its job.
    bool fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    /// perf-review D-4: `fill_slot_blocking`'s copy on the same (legacy) stream, but queued: many slots are refilled
    /// with one `sync_queued` at the end instead of a wait per slot. Same ordering against earlier work, same bytes.
    bool fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    bool sync_queued(std::string& err);

    /// Reads `slot` back to the host and compares it to `host_blob`, byte for byte.  **THE ONLY THING THAT SAYS
    /// THE CACHE HOLDS THE EXPERT IT CLAIMS TO.**  A slot table that is right about indices and wrong about
    /// bytes produces a plausible token, which is exactly the failure this project has paid for most often.
    bool verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);

    /// Slots filled so far, for the startup report.
    int64_t fills() const { return fills_; }

private:
#if defined(STRATA_USE_HIP)
    bool ensure_blocking_staging(std::size_t bytes, std::string& err);
    uint8_t* blocking_staging_ = nullptr;
    std::size_t blocking_staging_bytes_ = 0;
#endif
    uint8_t* base_ = nullptr;
    std::vector<int32_t> residency_;   ///< [n_layers * n_expert] -> slot or kNotResident
    int64_t slots_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t blob_ = 0;
    int64_t next_free_ = 0;
    int64_t fills_ = 0;
    /// R4.2g.  `per_layer_` off (the default) leaves `next_free_` as the only admission counter, so the
    /// pre-existing path is untouched.
    bool per_layer_ = false;
    std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
    std::vector<uint64_t> off_;         ///< plan v0.3 P6: slot offsets (slots + 1 entries) when sized
    int64_t admitted_ = 0;
    int planar_planes_ = 0;             ///< P2: >0 = planar storage (region p holds all slots' plane p)
    int64_t planar_bytes_[kMaxPlanes] = {};
    uint64_t planar_region_[kMaxPlanes] = {};   ///< region p's byte offset inside base_
};

}  // namespace strata::core
