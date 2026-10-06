// include/strata/core/pinned.hpp - P2.S1: the host arena the expert weights live in.
//
// 33.97 GB of expert weights cannot fit in a 12 GB card, so they stay in host memory and are streamed.  That
// makes this arena the engine's real working set: it must be PAGE-LOCKED for the copy engine to reach full
// bandwidth, and it must say which backing it got, because the two options differ by more than twice in TLB
// reach:
//
//   * LARGE PAGES (2 MB) - `mmap(MAP_HUGETLB)` / hugetlbfs on Linux, `VirtualAlloc(MEM_LARGE_PAGES)` on
//     Windows.  33.97 GB at 4 KB pages is 8.3 million TLB entries, which does not fit in any TLB, so every
//     block of every expert matvec takes TLB misses.
//   * NORMAL PAGES - the fallback.  Correct, slower, and it must be REPORTED rather than silently accepted:
//     "the engine adapts to the machine it is on" is only true if the engine says what it got.  On Windows
//     large pages additionally need SeLockMemoryPrivilege, which a normal user account does not have, so this
//     fallback is the common case and not an error path.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

enum class PageBacking { LargePages, NormalPages, PinnedByCuda };

struct PinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    PageBacking backing = PageBacking::NormalPages;
    std::string note;              // why the backing is what it is, for the startup print
    uint64_t locked_bytes = 0;     // resident via the working-set lock when CUDA could not pin it
    /// Plan v0.3 P5: when the whole arena cannot be registered, it is registered in `slice`-byte pieces from the
    /// start; this is the pinned prefix (a copy that stays inside one slice can then DMA straight from the arena).
    uint64_t registered_bytes = 0;
    uint64_t slice_bytes = 0;
    int registered_slices = 0;

    PinnedArena() = default;
    /// `slice`: the piece size for the per-slice registration fallback (0 = none).
    explicit PinnedArena(uint64_t bytes, uint64_t slice = 0);
    /// Plan v0.3 P6: slices of different sizes (one per layer of a native pack), given as their start offsets
    /// followed by the end of the last one.  `slice_starts` holds the registered ones.
    /// `max_pinned_bytes`: optional cap on CUDA registration. 0 preserves the normal unrestricted path.
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t max_pinned_bytes = 0);
    std::vector<uint64_t> slice_starts;
    ~PinnedArena();
    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return (uint8_t*) base; }
};

struct LoadStats {
    double seconds = 0.0;
    // Loader fix: the two halves of `seconds`, both SUMMED OVER THE READER THREADS rather than wall clock, so
    // either can exceed `seconds` when the threads overlap.  They exist to separate the read from the copy and
    // the hash: an 8 MiB chunk that reaches the disk as ~2048 4 KiB reads spends its time in the first one.
    double read_seconds = 0.0;                  // inside the read call only: no seek, no copy, no hash
    double copy_seconds = 0.0;                  // memcpy + FNV-1a
    // A short read is not a slow load, it is a WRONG one: the caller must refuse the pack rather than run
    // on an arena whose tail was never written.  `seconds` alone cannot say that (it is -1.0 on failure, but
    // the caller's own check is on the REQUESTED byte count, which a refused read does not change).
    bool ok = true;                             // false: the load failed, see `error`
    std::string error;                          // why it failed, for the caller's message
    uint64_t bytes = 0;
    uint64_t layers = 0;
    std::vector<uint64_t> layer_checksums;      // one FNV-1a per layer
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
    // Aggregate rate of the readers WHILE THEY WERE INSIDE THE READ CALL: `bytes` over the mean per-thread
    // read time.  `gib_per_second()` above divides the same bytes by the wall clock of the whole loop, which
    // also contains the copy and the hash, so the two differ by how much of the loop was not I/O.
    double read_gib_per_second(int threads) const {
        if (read_seconds <= 0.0 || threads < 1) return 0.0;
        return (double) bytes / (1024.0 * 1024 * 1024) / (read_seconds / (double) threads);
    }
};

// Load `layers` layers of the expert arena into `dst` with `threads` readers, `chunk` bytes at a time.
// Each thread opens its OWN handle and seeks, which is the portable form of parallel pread: a shared handle
// needs a lock around the seek and defeats the parallelism on Windows.
//
// The reads go through `fread()` on a per-thread `FILE*`, NOT through `std::ifstream`.  MSVC's
// `basic_filebuf::xsgetn` splits every request larger than `_INTERNAL_BUFSIZ - 1` into 4095-byte `fread()`
// calls, so an 8 MiB `chunk` reached the disk as ~2048 4 KiB operations and the chunking did nothing at all
// (measured on a 42.9 GB pack: 1222 s, 4096 bytes per operation, 0.03 GiB/s).  `fread()` hands a request
// larger than the stream buffer straight to `_read()`/`ReadFile()`, which is what `chunk` is for.  The
// control flow, the copy, the per-layer FNV-1a and therefore the checksums are unchanged, so a pack loaded
// by the old and the new reader must produce identical `layer_checksums`.
LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk);
/// Plan v0.3 P6: the same with one byte range per layer (`layer_off[L]`, `layer_bytes[L]`).
LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk);

// FNV-1a 64.  Per layer, so a corrupt or short read names WHICH layer rather than just failing a whole-file
// comparison - the same reason the Phase 1 tools report the first differing element.
uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = 1469598103934665603ull);

// ---- the expert stream -----------------------------------------------------
//
// THIS IS THE NUMBER THE OBJECTIVE DEPENDS ON.  With one expert per stream, the design's own arithmetic is:
//
//     per token: E = 663.6 MB of expert weights, of which only the MISSES cross the bus
//     miss fraction (1 - h) x 663.6 MB / measured bandwidth = milliseconds per token
//
// So `h` (the VRAM hit rate, ~0.167 by design) is only worth anything if a per-expert transfer can actually
// reach the bus's bandwidth.  A 1,382,400-byte copy is SMALL, and if small copies run at a fraction of large
// ones then the architecture's central constant is wrong and no kernel optimisation can fix it.  That is what
// this measures: the same total bytes at several granularities, so the difference is visible.
struct StreamStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t chunk = 0;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};

// Copy `bytes` from pinned host memory at `src` to a device buffer of `chunk` bytes, in `chunk`-sized
// transfers, `iters` times.  The destination is REUSED, so this measures the bus and not allocation.
StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters);

}  // namespace strata::core
