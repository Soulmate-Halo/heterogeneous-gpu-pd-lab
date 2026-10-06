// include/strata/kernels/ngram.hpp - P2.S4: the PLE n-gram hash and the per-layer token table.
//
// Two halves that share nothing but the row indices they exchange:
//
//   * `ngram_rows` - the HOST hash.  Sixteen row indices per token from the last three token ids, computed
//     with 64-bit multiply/xor and no tensor op at all.  `qwen4exp.cpp` L1092-1124, where the source says
//     outright that it is host-side because "ggml has no int64 and no xor".
//   * `PleTable`  - the row gather.  `per_layer_token_embd.weight` is IQ4_NL and is NOT IN THE PACK: it is
//     51.2e9 elements (28.8 GB) in the ORIGINAL second GGUF shard, and it is the only tensor this engine
//     reads from the GGUF rather than from the canonical pack.
//
// THE HASH IS THE ONE PLACE IN THIS MODEL WHERE A WRONG BIT IS SILENTLY PLAUSIBLE.  Every one of its
// properties has a rival reading that produces a valid index in range:
//
//   * XOR, not sum.  Two equal tokens with equal multipliers cancel to zero under XOR and double under sum.
//   * `%` vocab_size, not `& (vocab_size-1)`.  The vocab sizes are 20000003, 20000023, 20000033, ... - not
//     powers of two, so a mask would be a different function.
//   * the EOS cut propagates FORWARD: an EOS at any position replaces that position AND every older one.
//   * a missing predecessor is `LLAMA_TOKEN_NULL = -1`, tested as `t < 0`.  Token id 0 is a REAL token.
//   * the token's OWN eos does not cut its own context - only the predecessors are examined.
//
// All of those are asserted observable in `ple_parity.cpp` rather than trusted.
#pragma once

namespace strata::ngram { struct PleRemoteRows; }

#include <cstdint>
#include <string>

namespace strata::kernels {

// ---- geometry, every value confirmed against the artifact (docs/gguf-metadata.txt) ----
inline constexpr int NG_N_EMBD = 2560;          // n_embd
inline constexpr int PLE_HEAD_DIM = 160;        // qwen4exp.embedding_length_per_layer_input
inline constexpr int NGRAM_SIZE = 3;            // qwen4exp.ple.ngram_size
inline constexpr int HEADS_PER_NGRAM = 8;       // qwen4exp.ple.heads_per_ngram
inline constexpr int PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM;   // 16
inline constexpr int PLE_CONV_KERNEL = 4;       // qwen4exp.ple.conv_kernel
inline constexpr int32_t PLE_EOS_TOKEN_ID = 248044;
inline constexpr int NG_HC = 4;                 // hyper_connection.count
inline constexpr int NG_HC_DIM = NG_HC * NG_N_EMBD;                      // 10240
inline constexpr int NG_HIST = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE;       // 9
inline constexpr int32_t TOKEN_NULL = -1;       // LLAMA_TOKEN_NULL
inline constexpr float NG_RMS_EPS = 1e-6f;

// The table: [160, 320001536] IQ4_NL.  ne0 = 160 is the FAST axis, so one row is 160 contiguous elements =
// 5 blocks of 32 at 18 bytes = 90 bytes.  The head-slowest flatten then makes 16 rows exactly n_embd = 2560.
inline constexpr uint64_t PLE_TABLE_ROWS = 320001536ull;
inline constexpr int PLE_ROW_BYTES = (PLE_HEAD_DIM / 32) * 18;           // 90
// The NVFP4 stack's PLE table (ple/qwen3.8-flash-next-ple-fp8.bin): the SAME 320,001,536 rows in the SAME
// order, one e4m3 byte per element instead of IQ4_NL blocks, times ONE global scale (the checkpoint's
// model-plefp8-scale.safetensors, 0.00019931793212890625). Row-level parity against the IQ4_NL decode is the
// latter's own quantization error (~0.08 rel_l2, measured 2026-10-02, evidence/nvfp4-p4b-ple-fp8-*.json).
inline constexpr int PLE_ROW_BYTES_FP8 = PLE_HEAD_DIM;                   // 160

/// The row encoding of the served table. IQ4_NL is the Q2 stack's; Fp8E4m3 is the NVFP4 stack's (and the
/// higher-fidelity one of the two - FP8 is the checkpoint's native storage, IQ4_NL a requantization of it).
enum class PleRowFormat { Iq4Nl = 0, Fp8E4m3 = 1 };

/// The artifact's own hash constants, transcribed from `docs/gguf-dump-shard1.txt`:
///
///     layer_multipliers = [23703573157769, 20109073645365, 8052911324071]
///     head_vocab_sizes  = [20000003, 20000023, ...]          16 entries, NOT an arithmetic sequence
///     head_offsets      = [0, 20000003, 40000026, ...]       the running sum of head_vocab_sizes
///
/// `ref/ngram.py::selftest` calls its own copies "synthetic but STRUCTURALLY REAL"; they are in fact
/// `20000003 + 10*i` for the vocab sizes, which the artifact's are not (20000003, 20000023, 20000033 - the
/// differences are 20, 10, ...).  A selftest is free to use whatever constants it likes, but a PARITY test
/// against the artifact must use these, and the difference is recorded because it is exactly the kind of
/// thing that gets copied from the reference into an engine.
struct PleConsts {
    uint64_t mult[NGRAM_SIZE];
    uint64_t vocab[PLE_N_HEADS];
    uint64_t offset[PLE_N_HEADS];
};
PleConsts ple_artifact_consts();

/// `mixed_n = (ctx[0]*m[0]) ^ (ctx[1]*m[1]) ^ ... ^ (ctx[n-1]*m[n-1])` in uint64.
///
/// Exposed separately so a test can assert the XOR STRUCTURE directly rather than only its consequences.
uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n);

/// The sixteen row indices per token.  `qwen4exp.cpp` L1092-1124.
///
///   tokens  (n_tokens,)      the ubatch's own tokens
///   prev    (n_tokens, 2)    predecessors OLDEST FIRST, `TOKEN_NULL` where there is none
///   out     (n_tokens, 16)   row indices into the PLE table
///
/// The window is built once per token and then only its PREFIXES are used, which is what makes the bigram
/// heads (0..7) and the trigram heads (8..15) agree about the recent history.
void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out);

/// The IQ4_NL codebook: `{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}`,
/// which is `kvalues_iq4nl` from `ggml-common.h` L1121 and also the manifest's `codebooks.IQ4NL`.
///
/// There is NO `-8` in this version of ggml.  Some releases of `dequantize_row_iq4_nl` subtract 8 from the
/// codebook value; at 3cf03257 they do not, and the difference would shift every one of the 51.2e9 elements.
int iq4nl_code(int code);

/// One 90-byte row -> 160 floats.  The canonical rule `value = cb[code] * scale + offset` with offset 0,
/// because IQ4_NL has no min term (`has_offset` is false in the manifest).
///
/// THE NIBBLE ORDER IS SPLIT-HALF, NOT INTERLEAVED, and that is the trap here.  `dequantize_row_iq4_nl` is
///     y[j]           = d * kvalues[qs[j] & 0xf];
///     y[j + QK4_NL/2] = d * kvalues[qs[j] >> 4];
/// so byte j carries elements j and j+16 - NOT 2j and 2j+1, which is the natural reading and which produces
/// a perfectly plausible embedding of the wrong 160 values.
void iq4nl_dequant_row(const uint8_t* row, float* out160);

/// One 160-byte FP8 row -> 160 floats: `out[i] = e4m3(row[i]) * scale`. e4m3 is 1S/4E(bias 7)/3M with no
/// inf and NaN at 0x7f/0xff; the artifact's table holds neither (NaN would poison the PLE sum, not hide).
void fp8e4m3_dequant_row(const uint8_t* row160, float scale, float* out160);

/// How the table's rows are read (plan v0.3 P2). `Direct` is the default: unbuffered 4 KiB reads from the SSD,
/// so the table never occupies RAM or the OS file cache. `Mmap` is the earlier memory-mapped path, kept as the
/// A/B arm; it returns the same bytes.
///
/// NEVER KEEP THE SHARD MAPPED WHILE READING IT DIRECT: a live section on the same file serializes the unbuffered
/// reads (311 -> 1,575 us per token, bench/results/2026-09-23-p2-ssd-direct). Direct mode drops its own mapping
/// after the header parse; nothing else in the process may hold one.
enum class PleIo { Direct, Mmap, Rdma };

struct PleIoOptions {
    PleIo mode = PleIo::Direct;
    uint32_t max_inflight = 64;      ///< outstanding SSD reads (decode needs 16; prefill chunks use more)
    uint64_t cache_rows = 1u << 20;  ///< bounded row cache: 1,048,576 rows x 90 B ~ 95 MB; 0 disables
    bool io_thread = true;           ///< reads submitted by a worker thread, not the caller
    /// Rdma mode only: the peer's registered table, read one-sided (the engine's RDMA tier builds this).
    /// The GGUF is still parsed locally for the shape check; no table byte comes from the local disk.
    strata::ngram::PleRemoteRows* remote = nullptr;
    /// Rdma mode only: the row encoding of the REMOTE table (the worker publishes it in PleInfo). FP8 needs
    /// the table's global scale; IQ4_NL ignores it. Direct/Mmap are always Iq4Nl - the local shard is.
    PleRowFormat format = PleRowFormat::Iq4Nl;
    float fp8_scale = 0.0f;
    /// Bare FP8 table (the NVFP4 stack's ple-fp8.bin): no GGUF header at all - row 0 starts at byte 0
    /// and the row count is file_size / PLE_ROW_BYTES_FP8.  Only legal with mode = Direct and
    /// format = Fp8E4m3 (the local NVMe read of the same table Rdma serves out of the worker's RAM).
    bool bare_table = false;
};

/// The PLE table.  Held by pointer-to-impl so this header does not drag `<windows.h>` into every
/// translation unit that wants the hash.
class PleTable {
public:
    PleTable();
    ~PleTable();
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;

    /// Open with an explicit I/O mode. The two-argument `open` below is the default (Direct).
    bool open(const std::string& gguf_path, std::string& err, const PleIoOptions& io);

    /// The split the plan asks for: `issue` as soon as the token id is known, `collect` just before layer 1
    /// needs the rows. `gather` is `issue` followed by `collect`. In Mmap mode `issue` only prefetches.
    bool issue(const uint32_t* rows16);
    bool collect(float* out2560, std::string& err);
    /// Plan v0.3 P5: the rows of `n_tokens` tokens (16 each, `rows` token-major) into `out` (2560 floats per token),
    /// as ONE reader request - page dedupe and sort across the whole batch, the reader's full queue depth.  Not
    /// while a single-token `issue` is pending.  The mapped mode gathers row by row.
    bool gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err);

    /// Fault injection (Direct mode): every row read completes no earlier than `us` after issue.
    void set_injected_delay_us(double us);

    /// Maps the ORIGINAL second GGUF shard read-only.  The tensor's data does NOT start at file offset 0:
    /// the manifest's `shard2_tensor.offset` is relative to the GGUF's DATA SECTION, and the header before it
    /// is 192 bytes.  Reading at 0 would silently decode the header plus 192 bytes of shifted rows - still
    /// plausible IQ4_NL, which is why the reader takes the offset from a real GGUF parse (`GgufFile`, the
    /// project's validated reader) and checks that the tensor exactly fills the rest of the file.
    bool open(const std::string& gguf_path, std::string& err);
    void close();
    bool is_open() const;
    uint64_t rows() const;

    /// 16 row indices -> 2560 floats.  The gathered rows are flattened HEAD-SLOWEST: row h's 160 values
    /// occupy `out[h*160, (h+1)*160)`, which is what `ggml_get_rows` does and what makes the result a plain
    /// n_embd-wide vector for the block that consumes it.
    void gather(const uint32_t* rows16, float* out2560) const;

    /// One row -> 160 floats, for diagnostics and for the parity test's oracle comparison.
    void read_row(uint32_t row, float* out160) const;

    /// Bytes actually touched since `open`, for the "how often is it not ready" measurement P2.S4 asks for.
    uint64_t bytes_read() const;

    /// One line of reader statistics (Direct mode), e.g. for --stats. Empty in Mmap mode.
    std::string io_report() const;
    PleIo mode() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// ================================ THE PLE GATHER'S PREFETCH, AND ITS A/B ARM ================================
//
// `gather` issues its sixteen row reads as sixteen SEPARATE page faults into a 26.8 GB mapping, which measured
// **2.10-2.61 ms per token** and is second only to the layer loop among the token's avoidable terms. The table
// is 90 B per row, so the sixteen rows are 1,440 B - 0.5 MB/s, which is latency and not bandwidth, and it is
// because the sixteen pages are taken one at a time. `PrefetchVirtualMemory` issues them in one call.
//
// **THIS SWITCH EXISTS SO THE CLAIM CAN BE MEASURED RATHER THAN ASSERTED.** Cross-build comparisons in this
// project have repeatedly turned out to be machine drift - `LAYERS` alone moves 48.9 -> 52.3 between runs of
// one binary - so the two arms are alternated inside one session. Off is `--no-ple-prefetch`.
void ple_prefetch_enable(bool on);
bool ple_prefetch_enabled();

// ---- NOTE ON `madvise(MADV_RANDOM)` ------------------------------------------------------------------
// P2.S4 asks for `madvise(MADV_RANDOM)` on the mapping, and on Linux that is exactly right: a token gathers
// 16 rows scattered over 28.8 GB, so read-ahead is pure waste and would evict useful pages.
//
// THERE IS NO LINUX, SO THERE IS NO MADV_RANDOM HERE.  Windows has no equivalent of the advice - the closest
// is `FILE_FLAG_RANDOM_ACCESS` at CreateFile time, which suppresses the cache-manager's read-ahead for the
// whole handle, and `PrefetchVirtualMemory` for the positive case.  `GgufFile` opens with neither, so this
// build gets the DEFAULT sequential read-ahead on a random access pattern.  Saying so is the point: a
// comment claiming MADV_RANDOM here would be a comment about code that does not run.  On Linux the advice
// belongs in `GgufFile::open`, which this module does not own.
// ------------------------------------------------------------------------------------------------------

}  // namespace strata::kernels
