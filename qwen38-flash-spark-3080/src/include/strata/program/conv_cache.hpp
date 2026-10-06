// include/strata/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in generate.cpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
//   * the root is PINNED: dropping it frees one slot's ~118 MB and costs every future conversation that
//     shares the prefix a full re-read of it (a 30K system prompt is ~30 s at ~1,000 tok/s);
//   * every other slot rotates by least recent use.  A checkpoint's stamp advances when it is created,
//     when a new checkpoint lands on its length, and when a request mounts through it.
//
// With a budget below two slots there is no room to keep root and leaf apart, so the pin switches off and
// the oldest checkpoint leaves - the previous behaviour, one slot = the newest point only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::program::conv_cache {

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    size_t v = 1;                     // the root (0) is pinned; least recent use among the rest
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

}  // namespace strata::program::conv_cache
