// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in the model's shard 1, so
    /// the arena can be filled from the GGUF itself when the pack has no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer, the GGUF file (a name beside the --native shard) that holds its experts when the model's
    /// shards split the layers (Swift's GGUFs: layers 13-47 in shard 2).  Empty = the --native shard itself.
    std::vector<std::string> gguf_file;
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load` finds a native pack).
const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), else sets the canonical layout.
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);

}  // namespace strata::kernels::cpu
