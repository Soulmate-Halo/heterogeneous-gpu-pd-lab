// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

// Loader fix: `fseek`/`ftell` are 32-bit on Windows by default (and the pack is 42.9 GB), and the 64-bit
// spelling is not the same on the two platforms the engine builds for.
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#endif

namespace strata::core {

namespace {

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note) {
#ifdef _WIN32
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege.  Having it assigned to the account is not enough: the
    // PROCESS must enable it in its own token (AdjustTokenPrivileges) before VirtualAlloc, or the call fails.
    // An account without the assignment, or a failure to enable, leaves the process as it was: VirtualAlloc
    // then refuses and the 4 KB fallback below runs - that is the EXPECTED outcome on a desktop.
    {
        HANDLE tok = nullptr;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp{};
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                if (!AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() != ERROR_NOT_ALL_ASSIGNED)
                    (void) 0;   // nothing actionable: the large-page attempt below reports the outcome
            }
            CloseHandle(tok);
        }
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege; a normal account does not have it and VirtualAlloc then
    // fails with ERROR_PRIVILEGE_NOT_HELD.  That is the EXPECTED outcome on a desktop, not an error.
    SIZE_T large = GetLargePageMinimum();
    // A/B switch: STRATA_NO_LARGEPAGES=1 skips the large-page attempt, same run, same boot.
    if (large > 0 && std::getenv("STRATA_NO_LARGEPAGES") == nullptr) {
        // MEM_LARGE_PAGES requires the allocation size to be an exact multiple of the large page size -
        // anything else is ERROR_INVALID_PARAMETER (87), which reads like a privilege problem but is not.
        // Round up: the slack is under 2 MB and the tail stays unused.
        const SIZE_T lbytes = (SIZE_T) (((SIZE_T) bytes + large - 1) / large * large);
        void* p = VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) {
            got = PageBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long) large) + " B)";
            return p;
        }
        // 1450 (ERROR_NO_SYSTEM_RESOURCES) is the large-page pool saying no, 87 is a size that is not a
        // multiple of the minimum, 1314 is the privilege: without the byte count the three read as one bug.
        note = "large pages refused for " + std::to_string((unsigned long long) lbytes) + " B (GetLargePageMinimum=" +
               std::to_string((unsigned long long) large) + ", VirtualAlloc error " +
               std::to_string((unsigned long long) GetLastError()) + "); using 4 KB pages";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = PageBacking::NormalPages;
    return p;
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
    if (p != MAP_FAILED) {
        got = PageBacking::LargePages;
        note = "hugetlb 2 MB pages";
        return p;
    }
    note = "MAP_HUGETLB unavailable (no hugetlb pool configured?); using 4 KB pages";
    p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = PageBacking::NormalPages;
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
#ifdef _WIN32
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }
}  // namespace

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         uint64_t max_pinned_bytes) : capacity(bytes) {
    if (bytes == 0) return;
    base = reserve(bytes, backing, note);

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        const bool capped = max_pinned_bytes > 0 && max_pinned_bytes < bytes && bounds.size() >= 2;
        const cudaError_t e = capped ? cudaSuccess :
            cudaHostRegister(base, (size_t) bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (!capped && e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // Plan v0.3 P5: the whole range is refused, so pin it slice by slice from the start.  The rest stays
            // resident through the working-set lock below.  (P6: slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (capped && (off > max_pinned_bytes || n > max_pinned_bytes - off)) break;
                if (cudaHostRegister((uint8_t*) base + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            note = (capped ? "cudaHostRegister limited to " + std::to_string(max_pinned_bytes >> 30) +
                             " GiB for CUDA1; " :
                             "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); ") +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) {
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                if (env == nullptr || std::string(env) != "0") {
                    const strata::platform::LockResult lr =
                        strata::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                    locked_bytes = lr.locked_bytes;
                    note = lr.note + "; " + note;
                }
            }
        } else {
            note = std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Plan v0.3 P0.1: keep it RESIDENT instead. Unpinned, Windows trims the arena under memory pressure
            // and the CPU pool's rate then depends on the OS; locking it through the working set needs no
            // special privilege. STRATA_ARENA_LOCK=0 is the A/B arm.
            const char* env = std::getenv("STRATA_ARENA_LOCK");
            if (env == nullptr || std::string(env) != "0") {
                const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            } else {
                note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
            }
        }
    }
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) strata::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else {
            cudaHostUnregister(base);
        }
        release(base, capacity);
        base = nullptr;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: see the header - MSVC's `basic_filebuf::xsgetn` splits any request larger than
        // `_INTERNAL_BUFSIZ - 1` into 4095-byte freads, which turned one 8 MiB chunk into ~2048 4 KiB reads.
        // `fread` sees a request bigger than the stream buffer and passes it to `_read()`/
        // `ReadFile()` unchanged, so the chunk size reaches the disk.  Buffered, not `FILE_FLAG_NO_BUFFERING`:
        // the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace strata::core
