#pragma once

// include/strata/core/remote_tier.hpp - the dispatch-facing interface every remote expert tier implements.
//
// `ExpertDispatch::remote[]` (`expert_source.hpp`) holds these pointers, so a tier can be a second CUDA device
// in the same process (`RemoteExperts`) or a machine at the other end of an RDMA flag ring (`RDMAExpertTier`)
// without the dispatch loop knowing which.  The contract is exactly the one `expert_pool_dispatch` already
// exercises:
//
//   * `begin` is called once per layer (single-token) or per verify-window layer (multi-token) BEFORE the CPU
//     pool runs.  `kind[i] == -1` marks rows nobody has claimed yet (0 = CUDA0 hit, 1 = PCIe share, 2 = a
//     previously attached remote tier); a tier claims rows by marking them in its own bookkeeping, reported
//     through `owns`.
//   * `owns(i)` is consulted AFTER `begin`: claimed rows are zeroed by the caller and skipped by the CPU pool.
//   * `finish(out)` is called AFTER the CPU pool drains and writes each claimed row of `out` (n_tok*k rows of
//     `n_embd` floats, ROUTING order - `moe_combine` weights against that order).
//
// The router weight is NEVER applied by any tier (`moe_combine` does it on the device).

#include <cstdint>
#include <string>

namespace strata::core {

class RemoteTier {
public:
    virtual ~RemoteTier() = default;

    /// `kind` is the primary verifier's classification (-1 = unclaimed), or null on the one-token path.
    /// `primary_res` is the CUDA0 static residency row table (n_layers x n_expert), or null.
    virtual bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                       const int32_t* kind, const int32_t* primary_res, std::string& err) = 0;
    virtual bool owns(int64_t index) const = 0;
    virtual bool finish(float* out, std::string& err) = 0;

    /// True when begin() arranged for the claimed rows of `out` to be written by the PEER itself (an RDMA
    /// write straight into the staging buffer) rather than by a later finish().  The dispatch must not zero
    /// such rows first - the zeroing would race the peer's write.  Default false: finish() writes the rows.
    virtual bool direct_write() const { return false; }
};

} // namespace strata::core
