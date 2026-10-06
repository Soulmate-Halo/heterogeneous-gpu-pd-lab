// include/strata/core/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768);
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    uint64_t vram_bytes() const { return vram_; }
    /// The draft layer's K/V state (read-only: --serve's STRATA_STATE_HASH check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// KV streaming: refill the ring of the drafter's window from its host copy for a sequence that continues at
    /// `upto` (a conversation-cache resume). No-op unless the drafter's K/V is a ring.
    void kv_restore(int64_t upto);
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err);

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< E-4: without the mapped staging (inputs copied on device)
    bool capture_round(int T, std::string& err);
    bool capture_step(int j, std::string& err);
    cudaGraphExec_t step_exec_[9] = {};
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int max_drafts_ = 1 << 30;
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t prefill_dev_exec_[9] = {};
    int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaGraphExec_t round_exec_[9] = {};

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
    float* probs_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
