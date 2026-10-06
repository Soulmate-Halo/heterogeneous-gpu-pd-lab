// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

}  // namespace strata::platform
