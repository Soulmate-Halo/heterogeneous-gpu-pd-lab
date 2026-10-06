// include/strata/kernels/verify_kernels.hpp - plan v0.3 P6: kernels for the speculative VERIFY window, where T
// tokens (the last accepted token and T-1 drafts) go through a layer at once.
//
// The GDN recurrence is the one piece of state that cannot simply be overwritten by the next window, so its
// kernels come in two halves:
//
//   * the verify half reads the conv history and the recurrent state, runs all T tokens in order (bitwise the
//     single-token kernels' arithmetic per token) and writes ONLY the per-token outputs - the state is left
//     untouched;
//   * the commit half, launched once acceptance is known, replays the first `n_keep` tokens from the inputs the
//     verify half stored and writes the state.  Rejected tokens therefore never touch the state and there is no
//     snapshot of the 113 MB recurrent state to keep.
//
// Everything that varies per window (token ids, `n_keep`) is read from DEVICE memory so the kernels can be captured.
#pragma once

#include <cstdint>

namespace strata::kernels {

inline constexpr int kVerifyMaxT = 8;

/// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
/// `history` is NOT written.  Bitwise `fused_gdn_conv_l2` per token.
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0);
/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// alpha/beta for T columns of x (T, n_embd): gate (T, h_v), beta (T, h_v).  Bitwise `fused_gdn_ab` per column.
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// The recurrence + output norm for T tokens (h = (T, conv_channels) as q|k|v, gate/beta (T, h_v), z/y
/// (T, value_dim)).  With `n_keep == nullptr` the state is read and NOT written (verify); otherwise the first
/// *n_keep tokens are run and the state is written (commit; `y` may be scratch).  Bitwise `fused_gdn_step_norm`.
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin = 0);
/// Spin until *flag >= value (a mapped host flag).  The value is fixed at capture, so several rings can be
/// outstanding at once (the split verify window keeps two).
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream);
/// the GPU's %globaltimer (ns) into buf[i] (a one-thread kernel: the verify window's stage profiler).
void gpu_stamp(unsigned long long* buf, int i, void* stream);

// ---- perf-review E-6: a layer whose routed experts are all in VRAM needs nothing from the host
/// One group's plan, built on the device when every routed expert of its n*k entries is resident: the host pool's
/// layout (counts | start | dst | tok | pad | ptr | ptr2 | start2, `capx` entries) and order (distinct experts in
/// routing order, their entries ascending), no PCIe groups.  *skip = ring when it did, else 0.
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream);
/// wait_flag_ge that also returns when *skip == value (device memory).
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream);
/// copy_i32_from_mapped unless *skip == value.
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream);
/// copy_from_mapped, or zeros when *skip == value (n a multiple of 4, 16-byte aligned).
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream);

/// Rows of the S2/S4/S8 embedding for T token ids read from DEVICE memory; out (T, n).  Bitwise `embedding_gather`.
void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream);
/// R[t][c][:] = x[t][:] for the hc streams.
void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// dst[:n] = src[(*index) * stride + :n]  (index read from device memory; a negative index copies nothing).
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream);

/// Plan v0.3 P6: copy *n (device memory) blobs of `blob_bytes` from mapped host memory (src[k], device aliases)
/// into dst + k * blob_bytes with coalesced 16-byte loads - the PCIe share of a layer's missed experts, staged
/// into VRAM before the grouped expert kernel reads them.  Launched for a capacity of `cap` blobs.
void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream);
/// ptr[k] = base + k * blob_bytes for k < *n (the staged copies `fetch_blobs` made).
void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream);

// ---- the MTP draft layer (src/core/mtp.cpp)
/// R[t][c][:] = h[t][c][:] + e[t][:]  (the embedding branch added to every stream).
void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// Every routed expert is resident (slot = expert id): slot[i] = ids[i], dst[i] = i, *count = n.
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
/// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
/// row = *row_dev (device memory).  `out` may be mapped host memory.
void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs = nullptr,
                float* out_p = nullptr);
/// dst row i = src row ids[i] (row_bytes each, multiple of 16), for n rows.
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream);
/// ids[t] = table[ids[t]] for n entries (a subset index back to a token id).
void map_ids(int32_t* ids, const int32_t* table, int n, void* stream);
/// probs[t] = softmax(logits[t])[ids[t]] for n_rows rows of n_vocab (the probability of each row's argmax).
void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);
/// Dense-attention step records for `n` cells: [cell, cell+1, (cell+1)/4, cell+1] from cells[i] (device memory).
void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream);
/// A sliding attention window: for `n` step records (kStepCount ints each, n_kv at [1]) the selection becomes the
/// last `window` cells: ids[q * ids_stride + j] = max(0, n_kv - window) + j and the record's width = the count.
void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream);

}  // namespace strata::kernels
