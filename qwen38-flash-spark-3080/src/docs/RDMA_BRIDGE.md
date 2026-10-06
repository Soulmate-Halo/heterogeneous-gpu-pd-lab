# Strata remote-stage RDMA boundary

`strata_remote_stage` is the transport boundary for a contiguous layer split.
The sender owns layers `[0, K)` and the receiver owns `[K, n_layers)`. One
complete verify-window activation is sent at the split; the transport does not
turn the model into a per-layer RPC pipeline. The data path is an RC QP using
`libibverbs`. A short TCP control connection only exchanges QP attributes; it
is never used for activation bytes. If verbs are not present, opening the
transport fails closed.

Two data planes share the same QP:

1. **Window messages (legacy)** — two-sided SEND/RECV with ACK, kept for the
   contiguous-layer hand-off path (`send_window`/`recv_window`).
2. **One-sided flag ring (Nico-style publish/poll)** — the cross-machine
   equivalent of the same-process pinned-memory flag exchange. Neither side
   ever blocks on a request/response round trip.

## Flag-ring layout

The receiver (cold-expert host, e.g. Spark) registers three regions and
exports their `rkey`/address in the `QpInfo` exchange at open time:

| region    | size                | access       | purpose                                   |
|-----------|---------------------|--------------|-------------------------------------------|
| doorbell  | 8 bytes             | REMOTE_WRITE | activation sequence written by the sender |
| inbox     | `slot_bytes`        | REMOTE_WRITE | one activation payload                    |
| ring      | `slots * slot_bytes`| REMOTE_READ  | result slots, 8-byte seq header + payload |

The sender (hot-expert GPU host, e.g. 3080) registers one local work buffer.

Per verify window (sequence `i`, slot `(i-1) % slots`):

- sender: `rdma_write_doorbell(slot, payload, len, i)` — chained WRs:
  RDMA_WRITE activation → peer inbox, then RDMA_WRITE 8-byte `i` → peer
  doorbell (same QP, ordered; only the doorbell WR is signaled).
- receiver: busy-polls its own doorbell with an acquire load (no wire
  traffic), computes the cold-expert share, then `ring_publish(slot, out,
  len, i)` — copies the payload into the ring slot and release-stores the
  seq header, so a remote read observes payload and header together.
- sender: `rdma_read_slot(slot, buf, 8+len)` — one RDMA_READ pulls header
  and payload; when the header equals `i` the window is complete. The only
  wait is `max(0, T_cold - T_hot)`, identical to the same-process design.

Geometry (`slots`, `slot_bytes`, 8-byte-aligned) is validated on both ends at
open; any mismatch fails closed. `max_(dest_)rd_atomic` is 16 to allow
concurrent reads. TCP is never a data-plane fallback.

## Build and local check

```bash
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build --target strata_remote_stage_tool -j
build/strata-remote-stage --selftest
```

The `--selftest` checks the wire header, geometry limits, and CRC without
needing a second host. The binary prints `rdma=true` only after an active
verbs port and RC QP have been established.

## Two-host flag-ring selftest

Start the receiver on the Spark side, then the sender on the 3080 side:

```bash
# Spark (receiver, result-ring owner)
build/strata-remote-stage --role receiver --mode selftest \
  --slots 4 --slot-bytes 65536 --windows 64 \
  --device <spark-hca> --port 39580 --timeout-ms 120000

# 3080 (sender, one-sided writer/reader)
build/strata-remote-stage --role sender --mode selftest --peer <spark-rdma-host> \
  --slots 4 --slot-bytes 65536 --windows 64 \
  --device <3080-hca> --port 39580 --timeout-ms 120000
```

Both sides must print `remote_stage OPEN PASS rdma=true`; the sender finishes
with `remote_stage RING PASS windows=64 ... mean_us=... p99_us=...` (per-window
doorbell→result latency) and the receiver with the same `RING PASS` line. The
sender verifies a byte-wise XOR transform plus CRC on every window.

## Two-host window smoke test (legacy)

```bash
# Spark (receiver)
build/strata-remote-stage --role receiver --bind :: --port 39580 \
  --device <spark-hca> --split-layer 24 --windows 4 --timeout-ms 30000

# 3080 (sender)
build/strata-remote-stage --role sender --peer <spark-rdma-host> --port 39580 \
  --device <3080-hca> --split-layer 24 --windows 4 --timeout-ms 30000
```

Both sides must finish with `remote_stage PASS`. Wiring either data plane into
the model executable still requires the caller to place the CUDA graph and
expert-cache work for each contiguous layer segment around the boundary; the
existing same-process `strata generate` path is not changed implicitly by
this target.
