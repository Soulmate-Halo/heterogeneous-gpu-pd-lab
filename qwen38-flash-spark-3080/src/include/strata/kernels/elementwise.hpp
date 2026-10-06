// include/strata/kernels/elementwise.hpp - the small ops a LAYER needs between its GEMVs, P2.S5.
//
// These are not interesting kernels and that is why they are in one file with one parity test: they are the
// glue in the per-layer graph, and glue that is written inline at each call site is glue with several
// slightly different float conventions.  Three of them exist because a projection returns a DIFFERENT
// quantity from the one the next kernel wants:
//
//   * `gdn_gate` turns (alpha, dt, ssm_a) into the recurrence's `gate`.  `ref/gdn.py` L265:
//     `gate = softplus(w_alpha . x + dt) * ssm_a`, with `softplus(x) = log1p(exp(x))` and the large-x branch
//     `ggml_compute_softplus_f32` uses.  `ssm_a` is NEGATIVE, which is what puts `exp(gate)` in (0,1).
//   * `scale_inplace` applies the delta net's `1/sqrt(S)` to q.  It is a separate op because
//     `ref/gdn.py` says the two llama.cpp paths apply it in DIFFERENT places, so the caller has to own the
//     choice rather than have it buried in a kernel.
//   * `f32_to_f16_bulk` is what lets the f32 arithmetic of the mixer feed the fp16-activation GEMVs.
//     `docs/activation-contract.md` is explicit that this is NOT the contract for a quantized weight - it is
//     the bridge between two stages that both already exist, and the contract decision is a separate one.
//
// `silu_inplace` is here because `shared_expert` has its own copy inline and `gdn` needs one for the conv
// output; a third copy would be two too many.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Decode one canonical embedding row already selected by the caller. Codes are
/// packed low bits first (2, 4 or 8 bits); scales/offsets are FP32 per group.
/// The optional offset defaults to +0. Multiplication and addition round separately,
/// matching the former CPU decoder. Launches asynchronously on the supplied stream.
/// The caller validates positive, divisible dimensions and row/plane bounds.
void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream);

/// `y[t][h] = softplus(alpha[t][h] + dt[h]) * ssm_a[h]`, one thread per (t, h).  In place is not possible:
/// `alpha` is read and `gate` is written, and they are different buffers by construction.
///
/// `h_v` is the number of v heads (48) and `n_tokens` the leading dimension; the real call is (1, 48).
void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream);

/// `x[i] *= s`, in place.
void scale_inplace(float* x, int64_t n, float s, void* stream);
/// `dst[i] += src[i]`.  **ADDED FOR R4's HIT/MISS SPLIT, WHERE THE TWO HALVES COME FROM DIFFERENT
/// ENGINES.**  The CPU writes the misses into `parts` and the GPU writes the resident experts into a
/// buffer of their own, so that the GPU's work can run CONCURRENTLY with the CPU's instead of after it -
/// and the two are summed here, on the stream, immediately before `moe_combine` reads the result.
///
/// The alternative - having the hit kernel write straight into `parts` - costs the whole overlap, because
/// the copy of the misses into `parts` cannot happen until the pool has produced them, and the kernel is
/// then ordered behind that copy.  Measured: the drain fell 18.2 -> 10.2 ms and the token did not move.
void add_inplace(float* dst, const float* src, int64_t n, void* stream);

/// `y[i] = f16(x[i])`, round-to-nearest-even, using the shared conversion in `f16_bits.hpp`.
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);

/// `y[i] = bf16(x[i])`, for the weights whose contract is a bf16 activation - the BF16 `hc_*`, `ssm_alpha`,
/// `ssm_beta`, `indexer.*` and `ple_value` tensors.
///
/// A SEPARATE ENTRY POINT FROM `f32_to_f16_bulk` and not a flag on it, because the two conversions are not
/// variants of one idea: fp16 has 5 exponent bits and 10 mantissa bits, bf16 has 8 and 7.  A caller that
/// picked the wrong one would get a plausible tensor at the wrong precision - 2^-11 against 2^-9, a factor of
/// four - and nothing downstream could tell which was intended.
void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);

/// `x[i] = x[i] / (1 + exp(-x[i]))`, in place.
void silu_inplace(float* x, int64_t n, void* stream);

/// `build_norm`: `y[r][c] = x[r][c] / sqrt(MEAN_c(x[r]^2) + eps) * w[c]`, over the LAST axis.
///
/// QSA's norm, used on `attn_q` (24x256), `attn_k` (2x256) and `indexer.q_proj` (4x128). `w` may be null.
///
/// **IT IS NOT `gdn_l2_norm` AND THE DIFFERENCE IS A FACTOR OF sqrt(cols) = 16 HERE.** `gdn_l2_norm` divides by
/// `sqrt(SUM(x^2) + eps)` and applies no weight; this divides by `sqrt(MEAN(x^2) + eps)` and multiplies by `w`.
/// Two kernels whose names both say "norm" and whose bodies differ in one `/ cols` is exactly the pair that
/// gets swapped, and swapping them produces a well-scaled, plausible tensor - so they are separate entry
/// points and the parity test asserts the two readings are OBSERVABLE apart rather than trusting the names.
///
/// There is NO `(1 + w)` folding: `ref/qsa.py::rms_norm` is `y * w`, and `build_norm` in the source is
/// `ggml_rms_norm` followed by a plain multiply. Some norms in this artifact DO fold (the GDN path records
/// which); this one does not, and the same `sqrt(cols)`-sized argument applies to getting it wrong.
void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream);

/// *seq = v in MAPPED PINNED memory - THE DOORBELL.  A KERNEL and not cudaEventRecord, because round 199
/// found that an event record inside a capture is SILENTLY DROPPED while a kernel is captured normally, and
/// round 209 measured the host seeing this one's result 0.050 ms into a 39.8 ms graph.
///
/// It lives here because the caller that needs it - moe_layer - is a HOST translation unit, where
/// __global__ and <<<>>> do not exist.
/// *seq += 1 - AN INCREMENT, NOT A STORE OF A CALLER-SUPPLIED VALUE.  A value passed in is evaluated on the
/// HOST at CAPTURE time, so a captured graph would ring the same number on every replay.  Reading and
/// incrementing in memory is what makes the count advance under cudaGraphLaunch.
void doorbell_ring(uint32_t* d_seq, void* stream);

/// Plan v0.3 P3 token graph: one thread spins until the host-written `*d_flag` equals the ring value `*d_seq`
/// (both mapped pinned memory), then fences so everything the host wrote before the flag is visible to the
/// nodes that follow.  Captured once per layer between `pre[l]` and the parts copy; it is what lets one graph
/// hold the whole token (bench/micro/device_wait.cu: 5.2 us per handoff, no driver call).
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream);

/// Plan v0.3 P3 token graph: copy `n` floats from MAPPED pinned host memory (`src` is its device pointer) into
/// device memory with a kernel, so the handoff stays on the compute queue (a memcpy node is a copy-engine
/// operation, which WDDM submits separately and which measured 67 flushes per token).
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream);
/// W1 (push bind v3): the worker RDMA-writes its rows CONTIGUOUS in entry order into a mapped block
/// ([u32 count][u32 pad][i32 rows[count]] header, data at offset 256) instead of scattering one SGE per
/// routing row.  This kernel - captured where copy_from_mapped used to sit, after doorbell_wait - places
/// each row at its routing position and zero-fills the rows the worker did not send.
void scatter_contig_mapped(float* dst, const void* blk, int64_t rows, int64_t width, void* stream);
/// `rows` rows of `width` floats from mapped memory, except the rows listed in hit_rows[0, *count) (device),
/// which are written +0.0 instead (a verify window's GPU-computed entries: the pool leaves zeros there).
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream);

/// Plan v0.3 P3: the doorbell's payload and its ring in ONE kernel.  Copies `x` (n floats), `ids` and `weights`
/// (k each) into the mapped host regions, fences, and increments the mapped sequence number - replacing three
/// device-to-host memcpy nodes (copy-engine operations in the middle of the layer chain) and the ring kernel.
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream);

/// Plan v0.3 P3: copy `n` int32 from mapped pinned host memory into device memory with a kernel (the QSA
/// per-token step and positions), instead of a host-to-device memcpy node in the middle of a layer.
void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream);

}  // namespace strata::kernels
