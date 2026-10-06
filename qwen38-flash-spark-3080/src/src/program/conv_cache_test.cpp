// src/program/conv_cache_test.cpp - the conversation cache's retention policy: the shared root survives,
// the leaves rotate by least recent use.
//
// The scenarios, in the serve loop's terms (generate.cpp): checkpoints are a prefix chain - stamps[0] is
// the root, the deepest point every request so far has shared (the end of the system prompt, in practice).
// A stamp advances on creation, when a new checkpoint lands on an existing length, and when a request
// mounts through the checkpoint.
//
//   1. one conversation over its budget: the oldest leaf leaves - with fresh stamps in creation order this
//      is exactly the previous first-in-first-out behaviour;
//   2. the root survives arbitrarily many newer checkpoints however old its stamp is;
//   3. a mount advances a checkpoint's stamp and the rotation then takes a different leaf;
//   4. a one-slot budget has no room for root and leaf, so the pin is off and the oldest leaves (the
//      newest point stays, as before);
//   5. the policy is pure: the same stamps, the same victim.
#include "strata/program/conv_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using strata::program::conv_cache::eviction_victim;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("conv_cache_test\n");
    {
        const std::vector<uint64_t> stamps = {1, 2, 3, 4, 5, 6, 7};   // chat A: root + six turns, one over
        check(eviction_victim(stamps.data(), stamps.size(), 6) == 1,
              "over budget with fresh stamps: the oldest leaf leaves (the FIFO it was)");
    }
    {
        // chat B mounts through the root long after it was created; its stamp stays ancient
        const std::vector<uint64_t> stamps = {1, 100, 101, 102, 103, 104, 105};
        const size_t v = eviction_victim(stamps.data(), stamps.size(), 6);
        check(v == 1, "the root's stale stamp never makes it the victim");
        check(v != 0, "the root - the shared prefix - is never the victim");
    }
    {
        // the same chain after mounting checkpoint 1: its stamp jumps, the rotation moves to the next leaf
        const std::vector<uint64_t> stamped = {1, 8, 3, 4, 5, 6, 7};
        check(eviction_victim(stamped.data(), stamped.size(), 6) == 2,
              "after a mount, the rotation takes the next least recently used leaf");
    }
    {
        const std::vector<uint64_t> stamps = {1, 90, 91, 92};
        check(eviction_victim(stamps.data(), stamps.size(), 2) == 1,
              "a two-slot budget keeps root + newest and drops the leaf in between");
        check(eviction_victim(stamps.data(), stamps.size(), 1) == 0,
              "a one-slot budget has no pin: the oldest leaves, the newest point stays");
    }
    {
        const std::vector<uint64_t> stamps = {7, 3, 3, 5};
        const size_t a = eviction_victim(stamps.data(), stamps.size(), 4);
        const size_t b = eviction_victim(stamps.data(), stamps.size(), 4);
        check(a == 1 && b == 1, "ties among equally stale leaves: the earlier index, deterministically");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
