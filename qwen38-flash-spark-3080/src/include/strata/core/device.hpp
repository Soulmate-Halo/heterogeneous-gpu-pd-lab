// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE cudaMalloc per planner region with bump sub-allocation below it and no frees.  That is
// not a simplification for the first version: the memory plan from P1.S9 is fixed at startup, so the set of
// regions and their sizes is known before anything is allocated, and an allocator that can free would be
// solving a problem the engine does not have while adding fragmentation and failure modes it does.
//
// The reason to get this in early is that the VRAM budget is the binding constraint of the whole design
// (5.95 GB pooled between KV and the expert cache, 33.97 GB of experts in DRAM).  A runtime that discovers at
// token 4000 that it has overcommitted has already lost; the plan is printed against `cudaMemGetInfo` at
// startup so the discrepancy is visible immediately.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
};

// Throws when there is no CUDA device.  The engine targets sm_120 specifically and must say so rather than
// run slowly on something else: `CMakeLists.txt` already refuses to COMPILE for another architecture, and
// this is the matching check at run time (a binary can be carried to a different machine).
DeviceInfo device_info(int ordinal = 0);

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// One cudaMalloc, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a debug
// build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same reasoning as
// the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

}  // namespace strata::core
