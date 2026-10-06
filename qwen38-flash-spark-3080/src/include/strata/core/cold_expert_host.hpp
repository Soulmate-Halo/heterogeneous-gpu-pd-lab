// include/strata/core/cold_expert_host.hpp - P1②: the cold-expert worker's RDMA doorbell as an IN-PROCESS
// thread of the engine, computing on the engine's OWN resident NVFP4 marlin plane.
//
// **WHAT THIS IS.**  The dual-machine stack's Spark side used to be a separate process
// (tools/cold_expert_worker.cpp) with its own 63.3 GiB arena - which made a standalone Spark prefill
// engine (P1①'s --expert-resident-all, ~90 GiB) and the decode peer's cold tier physically mutually
// exclusive on one 121 GiB GB10.  This host runs the worker's exact wire protocol (rdma_expert_tier.hpp's
// RXE1 pull ring and the RXE2/3/4 push channels, byte for byte) on a thread inside the engine process,
// and computes every window with `moe_prefill_grouped_nvfp4` against the planar tier the engine already
// filled: ONE copy of the weights covers local prefill AND the 3080 peer's cold windows (~96 GiB total,
// they coexist).  The peer cannot tell the difference, and the standalone worker stays as the Q2_0 /
// fallback path - this file touches none of it.
//
// **SEMANTICS KEPT.**  Unweighted expert outputs, f32 rows in ENTRY order; the marlin entry's h_dst writes
// each row at its entry index.  Entries are grouped by expert first-appearance (contiguous runs, what the
// marlin batch path's consecutive-slot grouping wants); the channel-0 W1 header, flag values and write
// chains are the worker's, verbatim.  Fail-closed like the worker: a malformed bind or request kills the
// process rather than serving a plausible wrong row.
//
// **WHY THE STAGE OPENS ON THE THREAD.**  RemoteStage::open as Receiver blocks on the accept poll until
// the peer connects (remote_stage.cpp:303).  Doing that in start() would make the engine's whole boot wait
// for the 3080 - the wrong dependency direction (the engine must be up and serving prefill first).  So
// start() synchronously does everything that can fail fast and cheap (arena liveness, a port-taken probe,
// the CUDA allocations) and the thread does the accept; stop() wakes a pending accept with a self-connect
// so engine shutdown never hangs on the accept window.
//
// **DAEMON, NOT WORKER.**  The standalone worker exits when its peer's engine disconnects (it has nothing
// else to do).  This host must NOT: the same process serves the prefill leg, and a 3080 restart is a
// routine event.  Transport loss ends the SESSION (buffers and binds reset) and the thread re-listens;
// only protocol violations and compute failures stay fail-closed (process exit), exactly the rows the
// worker refuses too.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::kernels {
struct Nvfp4HotArena;
}

namespace strata::core {

struct ColdExpertHostConfig {
    std::string bind = "0.0.0.0";
    std::string device;                  ///< HCA name; empty = first verbs device
    int port = 39580;
    int gid = 0;
    int slots = 4;                       ///< result ring depth (must match the engine peer)
    int64_t slot_bytes = 4 << 20;        ///< ring slot bytes (must match the engine peer)
    int timeout_ms = 600000;             ///< the accept window: the peer may take a while to start
    bool verbose = false;
};

class ColdExpertHost {
public:
    /// `shared_arena` and `slot_of` are NOT owned: both must outlive the host (the engine's nv4arena and a
    /// vector it keeps in main scope).  slot_of[layer * n_expert + expert] = the marlin plane slot.
    ColdExpertHost(const kernels::Nvfp4HotArena* shared_arena, const int32_t* slot_of, int n_layers,
                   int n_expert);
    ~ColdExpertHost();
    ColdExpertHost(const ColdExpertHost&) = delete;
    ColdExpertHost& operator=(const ColdExpertHost&) = delete;

    /// Validates the arena, probes the port, allocates the pinned/device buffers and spawns the doorbell
    /// thread (which then blocks in accept until the peer connects).  Synchronous failures come back here
    /// so the engine refuses to start half-armed; protocol failures after that are fail-closed (the thread
    /// logs and exits the process, exactly like the standalone worker).
    bool start(const ColdExpertHostConfig& cfg, std::string& err);
    /// Signals the doorbell loop, wakes a pending accept and joins the thread.  Idempotent; the destructor
    /// calls it.
    void stop();
    uint64_t served_windows() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core
